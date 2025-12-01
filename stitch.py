import zarr
import json
import numpy as np
import dask.array as da
import deconvolution 

def get_zarr_array_safe(z_path):
    """
    Zarr 파일을 안전하게 Dask Array로 로드
    """
    try:
        store = zarr.open(str(z_path), mode='r')
        # 계층 구조에 따라 데이터 위치 찾기
        if isinstance(store, zarr.core.Array): return da.from_zarr(store)
        if "0" in store:
            if isinstance(store["0"], zarr.core.Array): return da.from_zarr(store["0"])
            # OME-Zarr 구조 ("0"/"0")인 경우
            if "0" in store["0"]: return da.from_zarr(store["0"]["0"])
            keys = list(store["0"].keys())
            if keys: return da.from_zarr(store["0"][keys[0]])
    except Exception as e:
        print(f"Failed to load {z_path}: {e}")
        return None
    return None

def scan_global_min_max(zarr_dirs):
    """
    모든 타일의 Min/Max를 스캔 (Contrast Normalization용)
    """
    temp_tiles = [get_zarr_array_safe(z) for z in zarr_dirs]
    temp_tiles = [t for t in temp_tiles if t is not None]
    
    if not temp_tiles: return None, None
    
    # Lazy하게 전체 연결 후 min/max 계산
    full_lazy = da.concatenate([da.concatenate(temp_tiles, axis=0)], axis=0)
    g_min, g_max = da.compute(full_lazy.min(), full_lazy.max())
    return float(g_min), float(g_max)

def extract_patch_numpy(tile, side, ov_x, ov_y):
    """
    Dask Array에서 오버랩 영역만큼 패치를 추출하고, 
    즉시 .compute()하여 메모리(NumPy)로 가져옴.
    -> 이것이 Dask FFT 에러를 방지하는 핵심입니다.
    """
    # tile shape 가정: (T, C, Z, Y, X)
    try:
        if side == "right":
            patch = tile[..., :, :, -ov_x:]
        elif side == "left":
            patch = tile[..., :, :, :ov_x]
        elif side == "bottom":
            patch = tile[..., :, -ov_y:, :]
        elif side == "top":
            patch = tile[..., :, :ov_y, :]
        else:
            return None
        
        # [수정 포인트] 3D -> 2D 변환 전략
        # 기존: patch.mean(axis=(0,1,2)) -> 전체 평균 (흐릿함)
        # 변경: patch.max(axis=2).mean(axis=(0,1)) -> Z축(axis 2)은 Max Projection
        
        # T(0), C(1)은 평균을 내거나 첫 번째 채널만 쓰기도 하지만, 
        # Z(2)는 max를 취하는 것이 국룰(MIP)입니다.
        
        # (T, C, Z, Y, X) -> Z축(2번)에 대해 Max를 취해 깊이 정보를 압축
        patch_mip = patch.max(axis=2)
        
        # 남은 T, C 축은 평균을 내서 하나의 2D 이미지(Y, X)로 만듦
        patch_2d = patch_mip.mean(axis=(0,1))
        
        return patch_2d.compute() # Numpy로 변환
        
    except Exception as e:
        print(f"Patch extraction failed: {e}")
        return None

def phase_corr(a, b, search_px):
    """
    두 NumPy 패치(a, b) 사이의 이동량을 계산 (위상 상관법)
    """
    if a is None or b is None: return np.array([0, 0])

    # FFT 수행
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * B.conj()
    # 0으로 나누기 방지
    R /= (np.abs(R) + 1e-8)
    corr = np.fft.ifft2(R).real
    
    # 가장 높은 상관관계(Peak) 찾기
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    
    # 좌표 중심 보정 (FFT 결과는 순환적이므로)
    if y > corr.shape[0]//2: y -= corr.shape[0]
    if x > corr.shape[1]//2: x -= corr.shape[1]
    
    # [Robustness] 너무 큰 이동(튀는 값)은 노이즈로 간주하고 무시
    # 배경이 없는 Chip 이미지에서 엉뚱하게 100px씩 튀는 것을 막음
    if abs(y) > search_px or abs(x) > search_px:
        # 보정 포기 (기계 좌표 신뢰)
        return np.array([0, 0], dtype=float)
        
    return np.array([y, x], dtype=float)

def build_graph(zarr_dirs, cfg, g_min, g_max):
    """
    동적 크롭(Dynamic Crop) 방식의 스티칭 그래프 생성
    """
    rows = cfg['preprocess']['rows']
    cols = cfg['preprocess']['cols']
    overlap_pct = cfg['preprocess']['overlap_pct']
    # search_px: 보정 허용 범위 (기본 30px, config에 없으면 30 사용)
    search_px = cfg['preprocess'].get('search_px', 30)

    # 1. Tile 로딩
    tiles = []
    # 파일명 순서대로 정렬되어 있다고 가정
    for z_dir in zarr_dirs:
        raw = get_zarr_array_safe(z_dir)
        if raw is not None: tiles.append(raw)

    if len(tiles) != rows * cols:
        print(f"Error: Tile count ({len(tiles)}) does not match grid ({rows}x{cols})")
        return None

    # Shape 추출 (T, C, Z, Y, X)
    _, _, _, H, W = tiles[0].shape
    
    # 기본 오버랩 픽셀 수 (기계 설정값)
    default_ov_x = int(W * overlap_pct)
    default_ov_y = int(H * overlap_pct)

    print(f"Stitching {rows}x{cols} tiles with Shift Correction (Range: ±{search_px}px)...")

    # ---- 2. 가로(Row) 방향 스티칭 & 보정 ----
    stitched_rows = []
    
    for r in range(rows):
        row_tiles_processed = []
        
        for c in range(cols):
            idx = r * cols + c
            tile = tiles[idx]
            
            if c == 0:
                # 첫 번째 열은 그대로 사용
                row_tiles_processed.append(tile)
            else:
                # 이전 타일(왼쪽)과 현재 타일(오른쪽) 비교
                prev_tile = tiles[idx - 1]
                
                # 패치 추출 (메모리에 로드됨)
                patch_L = extract_patch_numpy(prev_tile, "right", default_ov_x, default_ov_y)
                patch_R = extract_patch_numpy(tile, "left", default_ov_x, default_ov_y)
                
                # 이동량 계산
                shift = phase_corr(patch_L, patch_R, search_px)
                shift_x = int(shift[1]) # X축 이동량
                
                # [핵심] 잘라낼 양 결정 (기본 오버랩 - 이동량)
                # 예: 이동량이 -5px (왼쪽으로 밀림) -> 겹침이 5px 늘어남 -> 5px 더 잘라야 함
                cut_amount = default_ov_x - shift_x
                
                # 안전장치: 최소 1px은 남기고, 타일 전체를 날리진 않도록 제한
                cut_amount = max(1, min(cut_amount, W - 1))
                
                # 타일의 왼쪽 부분을 잘라냄 (Dask Slicing - 지연 연산)
                tile_cropped = tile[..., :, :, cut_amount:]
                row_tiles_processed.append(tile_cropped)
                
        # 가로 한 줄 합치기
        row_image = da.concatenate(row_tiles_processed, axis=4)
        stitched_rows.append(row_image)

    # ---- 3. 줄(Row) 길이 맞추기 (Trimming) ----
    # 보정으로 인해 각 줄의 길이가 다를 수 있으므로 가장 짧은 줄에 맞춤
    min_width = min([row.shape[-1] for row in stitched_rows])
    stitched_rows = [row[..., :min_width] for row in stitched_rows]
    print(f"  -> All rows trimmed to width: {min_width} px")

    # ---- 4. 세로(Col) 방향 스티칭 & 보정 ----
    final_col_processed = []
    
    for r in range(rows):
        row_img = stitched_rows[r]
        
        if r == 0:
            final_col_processed.append(row_img)
        else:
            prev_row = stitched_rows[r-1]
            
            # 속도 향상을 위해 이미지의 중앙 부분(ROI)만 사용하여 오차 계산
            center_x = min_width // 2
            roi_w = min(2048, min_width) # 최대 2048px 폭만 사용
            x_start = center_x - (roi_w // 2)
            x_end = x_start + roi_w

            # ROI 패치 추출 (Dask -> Numpy compute)
            # 주의: 여기서 row_img는 이미 연결된 Dask Array이므로 인덱싱 후 compute 필요
            try:
                patch_T = prev_row[..., :, -default_ov_y:, x_start:x_end].mean(axis=(0,1,2)).compute()
                patch_B = row_img[..., :, :default_ov_y, x_start:x_end].mean(axis=(0,1,2)).compute()
                
                shift = phase_corr(patch_T, patch_B, search_px)
                shift_y = int(shift[0]) # Y축 이동량
                
                cut_amount = default_ov_y - shift_y
                cut_amount = max(1, min(cut_amount, H - 1))
            except Exception as e:
                print(f"Warning: Vertical shift calc failed at row {r}, using default. ({e})")
                cut_amount = default_ov_y

            # 위쪽을 잘라내고 붙임
            row_cropped = row_img[..., :, cut_amount:, :]
            final_col_processed.append(row_cropped)

    # ---- 5. 최종 합치기 ----
    final_image = da.concatenate(final_col_processed, axis=3)
    
    return final_image

def generate_pyramid(base_zarr_path, levels=3):
    """
    기존에 생성된 Level 0 Zarr를 읽어서 다운샘플링된 피라미드(Level 1, 2...) 생성
    및 OME-NGFF 메타데이터 작성
    """
    store = zarr.DirectoryStore(str(base_zarr_path))
    try:
        root = zarr.open_group(store=store, mode='r+') # 읽기/쓰기 모드
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
        # np.mean을 사용하여 픽셀 평균값으로 축소 (부드럽게)
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
