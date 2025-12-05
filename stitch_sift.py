import zarr
import cv2
import numpy as np
import dask.array as da
import deconvolution 
import flat_field_correction

# =========================================================
# Helper Functions (SIFT & Image Processing)
# =========================================================

def to_uint8(img):
    """OpenCV 연산을 위한 8비트 변환"""
    if img.dtype == np.uint8:
        return img
    img_float = img.astype(np.float32)
    # 0으로 나누기 방지
    div = (img_float.max() - img_float.min())
    if div == 0: div = 1.0
    img_norm = (img_float - img_float.min()) / div
    return (img_norm * 255).astype(np.uint8)

def apply_enhanced_contrast(img):
    """
    Deconvolution 대신 SIFT 정확도를 높이기 위한 고속 대비 향상 (CLAHE)
    작은 패턴(15x15)을 찾기 위해 국소 대비 극대화
    """
    if img is None: return None
    
    img_8u = to_uint8(img)
        
    # CLAHE 적용 (tileGridSize=(8,8)은 작은 패턴 검출에 유리함)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_8u)
    
    return enhanced

def find_shift_sift(img1, img2):
    """SIFT Feature Matching (기존 로직 유지)"""
    if img1 is None or img2 is None: return np.array([0, 0])

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
        return np.array([0, 0])

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        return np.array([0, 0])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None: return np.array([0, 0])

    diff = dst_pts[:, 0, :] - src_pts[:, 0, :]
    valid_diff = diff[mask.ravel() == 1]
    
    if len(valid_diff) == 0: return np.array([0, 0])
        
    shift_x = np.median(valid_diff[:, 0])
    shift_y = np.median(valid_diff[:, 1])

    return np.array([-shift_y, -shift_x])

def get_raw_overlap(tile_dask, side, ov_x, ov_y, dummy_flat, dummy_dark):
    """
    Dask Array에서 필요한 경계면만 즉시 로드(compute)
    - Full Resolution 유지 (축소 X)
    - FFC 적용 (매칭 정확도 향상)
    - MIP (Max Projection) 적용
    """
    try:
        # tile_dask shape: (1, 1, Z, Y, X)
        
        # 1. FFC 적용 (전체 하지 않고 필요한 부분만 슬라이싱해서 적용하면 더 빠르지만,
        # 코드 복잡도를 줄이기 위해 Dask 그래프 상에서 먼저 적용함)
        if dummy_flat is not None:
            tile_corr = flat_field_correction.apply_flat_field(tile_dask, dummy_flat, dummy_dark)
        else:
            tile_corr = tile_dask

        # 2. MIP (Z축 Max) -> (1, 1, Y, X)
        mip = tile_corr.max(axis=2)
        mip_2d = mip[0, 0] # (Y, X) Lazy Array

        # 3. 필요한 영역만 슬라이싱 (Lazy)
        if side == "right":   patch = mip_2d[:, -ov_x:]
        elif side == "left":  patch = mip_2d[:, :ov_x]
        elif side == "bottom":patch = mip_2d[-ov_y:, :]
        elif side == "top":   patch = mip_2d[:ov_y, :]
        else: return None
        
        # 4. Compute (여기서 메모리로 로드 - 용량 작아서 빠름)
        return patch.compute()
        
    except Exception as e:
        print(f"Patch extraction error: {e}")
        return None

# =========================================================
# Phase 1: Shift Calculation (CPU Sequential / Fast IO)
# =========================================================

def calculate_shifts_phase(tiles, cfg, dummy_flat, dummy_dark):
    """
    모든 타일의 위치 관계를 미리 계산
    - Deconvolution 수행 안 함 (속도 확보)
    - CLAHE 적용 (15x15 미세 패턴 검출력 확보)
    """
    rows = cfg['preprocess']['rows']
    cols = cfg['preprocess']['cols']
    overlap_pct = cfg['preprocess']['overlap_pct']
    
    # 메타데이터만 확인
    _, _, _, H, W = tiles[0].shape
    ov_x = int(W * overlap_pct)
    ov_y = int(H * overlap_pct)
    
    shifts_h = np.zeros((rows, cols), dtype=object) # 가로 (X축) 이동량
    shifts_v = np.zeros((rows, cols), dtype=object) # 세로 (Y축) 이동량 (Row간)

    print(f"\n[Phase 1] Analyzing Shifts with CLAHE (No Deconv)... Grid: {rows}x{cols}")

    # 1. 가로 방향 스캔
    for r in range(rows):
        # 진행 상황 표시 (줄 단위)
        print(f"  - Analyzing Row {r+1} Horizontal Overlaps...")
        for c in range(1, cols):
            idx_prev = r * cols + (c - 1)
            idx_curr = r * cols + c
            
            # 원본에서 경계면 추출 (Full Res)
            p_L = get_raw_overlap(tiles[idx_prev], "right", ov_x, ov_y, dummy_flat, dummy_dark)
            p_R = get_raw_overlap(tiles[idx_curr], "left", ov_x, ov_y, dummy_flat, dummy_dark)
            
            # CLAHE로 미세 패턴 강조
            p_L_enh = apply_enhanced_contrast(p_L)
            p_R_enh = apply_enhanced_contrast(p_R)
            
            # SIFT
            shift = find_shift_sift(p_L_enh, p_R_enh)
            shifts_h[r, c] = int(shift[1]) # X shift
            
            # (옵션) 디버깅용: 시프트 값이 너무 크거나 0이면 경고
            # if shift[1] == 0: print(f"    Warning: No shift found at R{r} C{c-1}->C{c}")

    # 2. 세로 방향 스캔 (각 컬럼별로 위아래 비교 후 중앙값 사용)
    if rows > 1:
        print(f"  - Analyzing Vertical Overlaps...")
        for r in range(1, rows):
            col_shifts = []
            for c in range(cols):
                idx_top = (r - 1) * cols + c
                idx_bot = r * cols + c
                
                p_T = get_raw_overlap(tiles[idx_top], "bottom", ov_x, ov_y, dummy_flat, dummy_dark)
                p_B = get_raw_overlap(tiles[idx_bot], "top", ov_x, ov_y, dummy_flat, dummy_dark)
                
                p_T_enh = apply_enhanced_contrast(p_T)
                p_B_enh = apply_enhanced_contrast(p_B)
                
                shift = find_shift_sift(p_T_enh, p_B_enh)
                col_shifts.append(int(shift[0])) # Y shift
            
            # 해당 Row 연결부의 대표 이동값 (Median)
            median_shift = int(np.median(col_shifts))
            # 모든 컬럼에 동일 적용 (또는 컬럼별 적용 가능, 여기선 Grid 정렬 유지 위해 Median)
            for c in range(cols):
                shifts_v[r, c] = median_shift
                
    return shifts_h, shifts_v, ov_x, ov_y

# =========================================================
# Phase 2: Graph Building (GPU Parallel / Lazy)
# =========================================================

def build_graph_phase(tiles, shifts_h, shifts_v, ov_x, ov_y, cfg, g_min, g_max, dummy_flat, dummy_dark):
    """
    계산된 Shift를 적용하여 최종 실행 그래프 생성.
    ★ 중요: 여기서는 compute()를 안함.
    """
    rows = cfg['preprocess']['rows']
    cols = cfg['preprocess']['cols']
    
    print("\n[Phase 2] Building Deconvolution & Stitching Graph (Lazy)...")
    
    # 1. 가로 스티칭 (Row Stitching)
    stitched_rows = []
    
    for r in range(rows):
        row_tiles_processed = []
        for c in range(cols):
            idx = r * cols + c
            tile = tiles[idx]
            
            # 1-1. Flat Field Correction (Lazy)
            if dummy_flat is not None:
                tile = flat_field_correction.apply_flat_field(tile, dummy_flat, dummy_dark)
            
            # 1-2. Deconvolution (Lazy)
            # Dask가 알아서 GPU에 할당하고 병렬 처리함
            tile_deconv = tile.map_blocks(
                deconvolution.deconv_wrapper_torch,
                psf_cpu=deconvolution.get_psf_numpy(),
                iterations=cfg['algorithm']['iterations'],
                global_min=g_min,
                global_max=g_max,
                pad_width=cfg['algorithm']['pad_width'],
                dtype=np.float32
            )

            # 1-3. Cropping (Lazy)
            if c == 0:
                row_tiles_processed.append(tile_deconv)
            else:
                shift_x = shifts_h[r, c]
                cut_amount = ov_x - shift_x
                
                # 안전장치
                if cut_amount < 0: cut_amount = 0
                if cut_amount >= tile_deconv.shape[-1]: cut_amount = ov_x
                
                # 왼쪽 부분을 잘라내고 붙임
                tile_cropped = tile_deconv[..., :, :, cut_amount:]
                row_tiles_processed.append(tile_cropped)
        
        # Row 합치기
        row_image = da.concatenate(row_tiles_processed, axis=4)
        stitched_rows.append(row_image)

    # 너비 맞추기 (최소 너비 기준)
    min_width = min([row.shape[-1] for row in stitched_rows])
    stitched_rows = [row[..., :min_width] for row in stitched_rows]

    # 2. 세로 스티칭 (Col Stitching)
    final_col_processed = []
    
    for r in range(rows):
        row_img = stitched_rows[r]
        if r == 0:
            final_col_processed.append(row_img)
        else:
            # Row간 대표 shift 값 (아까 계산한 Median 사용)
            shift_y = shifts_v[r, 0] 
            cut_amount = ov_y - shift_y
            
            if cut_amount < 0: cut_amount = 0
            
            row_cropped = row_img[..., :, cut_amount:, :]
            final_col_processed.append(row_cropped)

    final_image = da.concatenate(final_col_processed, axis=3) 
    return final_image

def get_zarr_array_safe(z_path):
    """(기존 유지) Zarr 파일을 안전하게 Dask Array로 로드"""
    try:
        store = zarr.open(str(z_path), mode='r')
        if isinstance(store, zarr.core.Array): return da.from_zarr(store)
        if "0" in store:
            if isinstance(store["0"], zarr.core.Array): return da.from_zarr(store["0"])
            if "0" in store["0"]: return da.from_zarr(store["0"]["0"])
            keys = list(store["0"].keys())
            if keys: return da.from_zarr(store["0"][keys[0]])
    except Exception as e:
        print(f"Failed to load {z_path}: {e}")
        return None
    return None

def scan_global_min_max_arrays(dask_arrays):
    """
    [수정] 경로가 아닌 Dask Array 리스트를 받아 Min/Max 계산
    """
    valid_arrays = [a for a in dask_arrays if a is not None]
    if not valid_arrays: return 0.0, 1.0
    
    # Lazy하게 전체 연결 후 min/max 계산
    mins = [da.min(a) for a in valid_arrays]
    maxs = [da.max(a) for a in valid_arrays]
    
    # compute
    g_min, g_max = da.compute(da.min(da.stack(mins)), da.max(da.stack(maxs)))
    return float(g_min), float(g_max)

def generate_pyramid(base_zarr_path, levels=3):
    """
    기존에 생성된 Level 0 Zarr를 읽어서 다운샘플링된 피라미드(Level 1, 2...) 생성
    및 OME-NGFF 메타데이터 작성
    """
    store = zarr.DirectoryStore(str(base_zarr_path))
    try:
        root = zarr.open_group(store=store, mode='r+') 
    except:
        print("Failed to open Zarr for pyramid generation.")
        return

    # Level 0 (원본) 읽기
    base_array = da.from_zarr(root["0"])
    
    datasets = [{"path": "0"}]
    current_array = base_array
    
    print(f"Generating Image Pyramid (Levels: {levels})...")

    for i in range(1, levels + 1):
        print(f" - Processing Level {i}...")
        
        # 1. 다운샘플링 (Y, X 축만 2배씩 축소)
        # 5차원 (T, C, Z, Y, X) -> axis 3(Y), 4(X)만 축소
        # np.mean을 사용하여 픽셀 평균값으로 축소
        downsampled = da.coarsen(np.mean, current_array, {3: 2, 4: 2}, trim_excess=True)
        
        # 2. 저장 (Level i)
        downsampled.to_zarr(
            store, 
            component=str(i), 
            overwrite=True, 
            compute=True,
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )
        
        datasets.append({"path": str(i)})
        current_array = downsampled

    # 3. OME-NGFF 메타데이터 (.zattrs) 작성
    multiscales = [{
        "version":  "0.4",
        "name": "stitched_image",
        "datasets": datasets,
        "type": "gaussian", 
        "metadata": {
            "method": "mean_downsampling",
            "version": "1.0"
        }
    }]
    
    root.attrs["multiscales"] = multiscales
    print("Pyramid generation complete with Metadata.")
    