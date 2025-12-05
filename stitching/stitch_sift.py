import zarr
import cv2
import numpy as np
import dask.array as da
from preprocessing import deconvolution, flat_field_correction

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

def extract_patch_numpy(tile, side, ov_x, ov_y):
    """
    Tile shape: (T, C, Z, Y, X) -> Z축 Max Projection -> Y, X 평균
    SIFT 특징점 검출을 위해 가장 선명한 이미지를 만드는 MIP 방식을 사용합니다.
    """
    try:
        if tile.ndim != 5:
            return None

        if side == "right":   patch = tile[..., :, :, -ov_x:]
        elif side == "left":  patch = tile[..., :, :, :ov_x]
        elif side == "bottom":patch = tile[..., :, -ov_y:, :]
        elif side == "top":   patch = tile[..., :, :ov_y, :]
        else: return None
        
        # (T, C, Z, Y, X) -> Z(2) Max (MIP) -> T(0), C(1) Mean
        # MIP를 해야 세포 특징점이 가장 뚜렷하게 잡힙니다.
        patch_mip = patch.max(axis=2)
        patch_2d = patch_mip.mean(axis=(0,1))
        
        return patch_2d.compute() 
    except Exception as e:
        print(f"Patch extraction failed: {e}")
        return None

def to_uint8(img):
    """OpenCV는 uint8 타입만 받으므로 변환해주는 헬퍼 함수"""
    if img.dtype == np.uint8:
        return img
    # 정규화 (0~255)
    img_float = img.astype(np.float32)
    img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
    return (img_norm * 255).astype(np.uint8)

def find_shift_sift(img1, img2):
    """
    SIFT를 이용한 정밀 위치 보정 (Feature Matching)
    """
    if img1 is None or img2 is None: return np.array([0, 0])

    # 1. 8비트 변환 (OpenCV 필수)
    img1_8 = to_uint8(img1)
    img2_8 = to_uint8(img2)

    # 2. SIFT 생성
    sift = cv2.SIFT_create()

    # 3. 키포인트 & 디스크립터 검출
    kp1, des1 = sift.detectAndCompute(img1_8, None)
    kp2, des2 = sift.detectAndCompute(img2_8, None)

    # [디버깅 로그 1]
    print(f"  [SIFT Debug] Found Keypoints: img1={len(kp1)}, img2={len(kp2)}")

    # 매칭할 특징점이 없으면 실패 처리
    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
        return np.array([0, 0])

    # 4. 매칭 (BFMatcher)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 5. 좋은 매칭 필터링 (Lowe's ratio test)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # [디버깅 로그 2]
    print(f"  [SIFT Debug] Good Matches: {len(good)}")

    # 매칭 점이 너무 적으면 신뢰할 수 없음 -> 기계 좌표 믿음
    if len(good) < 4:
        print(f"  [SIFT Fail] Not enough matches ({len(good)} < 4). Returns (0,0)")
        return np.array([0, 0])

    # 6. 매칭된 점들의 좌표 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 7. 변환 행렬(Homography) 찾기 (RANSAC으로 이상치 제거)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        return np.array([0, 0])

    # 8. 이동량(Shift) 추출
    # 매칭된 점들의 좌표 차이의 중앙값(Median) 사용 (이상치에 강함)
    diff = dst_pts[:, 0, :] - src_pts[:, 0, :] # (x, y) 차이들
    
    # RANSAC 마스크가 1인(정상) 점들만 골라서 계산
    valid_diff = diff[mask.ravel() == 1]
    
    if len(valid_diff) == 0:
        return np.array([0, 0])
        
    shift_x = np.median(valid_diff[:, 0])
    shift_y = np.median(valid_diff[:, 1])

    # (Y, X) 순서로 리턴 (부호는 상황에 따라 반대일 수 있음)
    return np.array([-shift_y, -shift_x]) 

def build_graph(tiles, cfg, g_min, g_max):
    rows = cfg['preprocess']['rows']
    cols = cfg['preprocess']['cols']
    overlap_pct = cfg['preprocess']['overlap_pct']
    
    # FFC 사용 여부 확인
    use_ffc = cfg['preprocess'].get('use_flat_field', False)

    if len(tiles) != rows * cols:
        print(f"Error: Tile count ({len(tiles)}) does not match grid ({rows}x{cols})")
        return None

    _, _, _, H, W = tiles[0].shape
    
    # [수정] correction 모듈을 사용하여 더미 데이터 생성
    if use_ffc:
        print(" [Info] Applying Flat Field Correction (Dummy Data)...")
        dummy_flat, dummy_dark = flat_field_correction.generate_dummy_references((H, W))
    
    default_ov_x = int(W * overlap_pct)
    default_ov_y = int(H * overlap_pct)

    print(f"Stitching {rows}x{cols} tiles (3D Stacked)...")

    # ---- Row Stitching (가로) ----
    stitched_rows = []
    for r in range(rows):
        row_tiles_processed = []
        for c in range(cols):
            idx = r * cols + c
            tile = tiles[idx] 
            
            # FFC 적용
            if use_ffc:
                tile = flat_field_correction.apply_flat_field(tile, dummy_flat, dummy_dark)
            
            # Deconvolution 적용
            tile_deconv = tile.map_blocks(
                deconvolution.deconv_wrapper_torch,
                psf_cpu=deconvolution.get_psf_numpy(),
                iterations=cfg['algorithm']['iterations'],
                global_min=g_min,
                global_max=g_max,
                pad_width=cfg['algorithm']['pad_width'],
                dtype=np.float32
            )

            if c == 0:
                row_tiles_processed.append(tile_deconv)
            else:
                prev_tile = tiles[idx - 1] 
                
                # Shift 계산
                patch_L = extract_patch_numpy(prev_tile, "right", default_ov_x, default_ov_y)
                patch_R = extract_patch_numpy(tile, "left", default_ov_x, default_ov_y)
                
                print(f"   [Horizontal] Stitching Row {r+1}: Tile {c} <-> {c+1}")
                shift = find_shift_sift(patch_L, patch_R)
                shift_x = int(shift[1])
                
                cut_amount = default_ov_x - shift_x
                cut_amount = max(1, min(cut_amount, W - 1))
                
                tile_cropped = tile_deconv[..., :, :, cut_amount:]
                row_tiles_processed.append(tile_cropped)
                
        row_image = da.concatenate(row_tiles_processed, axis=4) 
        stitched_rows.append(row_image)

    min_width = min([row.shape[-1] for row in stitched_rows])
    stitched_rows = [row[..., :min_width] for row in stitched_rows]

    # ---- Col Stitching (세로) ----
    final_col_processed = []
    for r in range(rows):
        row_img = stitched_rows[r]
        if r == 0:
            final_col_processed.append(row_img)
        else:
            prev_row = stitched_rows[r-1]
            
            try:
                # [수정] 세로 스티칭 시 전체 너비를 사용하여 SIFT 정확도 향상
                # 메모리 절약을 위해 가로축 다운샘플링(::2) 수행
                
                # 위쪽 행의 밑바닥 (전체 너비)
                patch_T_dask = prev_row[..., :, -default_ov_y:, :]
                # 아래쪽 행의 윗부분 (전체 너비)
                patch_B_dask = row_img[..., :, :default_ov_y, :]
                
                # MIP & Mean -> 2D 변환 및 다운샘플링
                patch_T = patch_T_dask.max(axis=2).mean(axis=(0,1))[:, ::2].compute()
                patch_B = patch_B_dask.max(axis=2).mean(axis=(0,1))[:, ::2].compute()
                
                print(f"   [Vertical] Stitching Row {r} <-> {r+1}")
                shift = find_shift_sift(patch_T, patch_B)
                
                shift_y = int(shift[0])
                # shift_x = int(shift[1] * 2) # X축 이동은 여기서 무시하거나 필요 시 보정
                
                cut_amount = default_ov_y - shift_y
                cut_amount = max(1, min(cut_amount, H - 1))
            except Exception as e:
                print(f"Warning: Vertical shift calc error: {e}")
                cut_amount = default_ov_y

            row_cropped = row_img[..., :, cut_amount:, :]
            final_col_processed.append(row_cropped)

    final_image = da.concatenate(final_col_processed, axis=3) 
    return final_image

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
    