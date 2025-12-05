import zarr
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
    # 메모리 터짐 방지를 위해 샘플링하거나, map_blocks로 min/max만 계산 후 reduce 추천
    # 여기서는 간단히 전체 min/max (데이터가 아주 크다면 수정 필요함)
    mins = [da.min(a) for a in valid_arrays]
    maxs = [da.max(a) for a in valid_arrays]
    
    # compute
    g_min, g_max = da.compute(da.min(da.stack(mins)), da.max(da.stack(maxs)))
    return float(g_min), float(g_max)

def extract_patch_numpy(tile, side, ov_x, ov_y):
    """
    (기존 유지) 
    Tile shape: (T, C, Z, Y, X) -> Z축 Max Projection -> Y, X 평균
    """
    try:
        # shape 확인
        if tile.ndim != 5:
            # (1, 1, Z, Y, X)로 가정하므로 5차원이 아니면 에러 또는 reshape 필요
            return None

        if side == "right":   patch = tile[..., :, :, -ov_x:]
        elif side == "left":  patch = tile[..., :, :, :ov_x]
        elif side == "bottom":patch = tile[..., :, -ov_y:, :]
        elif side == "top":   patch = tile[..., :, :ov_y, :]
        else: return None
        
        # (T, C, Z, Y, X) -> Z(2) Max -> T(0), C(1) Mean
        patch_mip = patch.max(axis=2)
        patch_2d = patch_mip.mean(axis=(0,1))
        
        return patch_2d.compute() 
    except Exception as e:
        print(f"Patch extraction failed: {e}")
        return None

def phase_corr(a, b, search_px):
    """위상 상관법"""
    if a is None or b is None: return np.array([0, 0])
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * B.conj()
    R /= (np.abs(R) + 1e-8)
    corr = np.fft.ifft2(R).real
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    if y > corr.shape[0]//2: y -= corr.shape[0]
    if x > corr.shape[1]//2: x -= corr.shape[1]
    if abs(y) > search_px or abs(x) > search_px:
        return np.array([0, 0], dtype=float)
    return np.array([y, x], dtype=float)

def build_graph(tiles, cfg, g_min, g_max):
    rows = cfg['preprocess']['rows']
    cols = cfg['preprocess']['cols']
    overlap_pct = cfg['preprocess']['overlap_pct']
    search_px = cfg['preprocess'].get('search_px', 30)
    
    # FFC 사용 여부 확인
    use_ffc = cfg['preprocess'].get('use_flat_field', False)

    if len(tiles) != rows * cols:
        print(f"Error: Tile count ({len(tiles)}) does not match grid ({rows}x{cols})")
        return None

    _, _, _, H, W = tiles[0].shape
    
    # [수정] correction 모듈을 사용하여 더미 데이터 생성
    if use_ffc:
        print(" [Info] Applying Flat Field Correction (Dummy Data)...")
        # correction.py의 함수 호출
        dummy_flat, dummy_dark = flat_field_correction.generate_dummy_references((H, W))
    
    default_ov_x = int(W * overlap_pct)
    default_ov_y = int(H * overlap_pct)

    print(f"Stitching {rows}x{cols} tiles (3D Stacked)...")

    # ---- Row Stitching ----
    stitched_rows = []
    for r in range(rows):
        row_tiles_processed = []
        for c in range(cols):
            idx = r * cols + c
            tile = tiles[idx] 
            
            # [수정] correction 모듈을 사용하여 보정 적용
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
                
                shift = phase_corr(patch_L, patch_R, search_px)
                shift_x = int(shift[1])
                
                cut_amount = default_ov_x - shift_x
                cut_amount = max(1, min(cut_amount, W - 1))
                
                tile_cropped = tile_deconv[..., :, :, cut_amount:]
                row_tiles_processed.append(tile_cropped)
                
        row_image = da.concatenate(row_tiles_processed, axis=4) 
        stitched_rows.append(row_image)

    min_width = min([row.shape[-1] for row in stitched_rows])
    stitched_rows = [row[..., :min_width] for row in stitched_rows]

    # ---- Col Stitching ----
    final_col_processed = []
    for r in range(rows):
        row_img = stitched_rows[r]
        if r == 0:
            final_col_processed.append(row_img)
        else:
            prev_row = stitched_rows[r-1]
            center_x = min_width // 2
            roi_w = min(2048, min_width)
            x_start = center_x - (roi_w // 2)
            x_end = x_start + roi_w

            try:
                patch_T = prev_row[..., :, -default_ov_y:, x_start:x_end].max(axis=2).mean(axis=(0,1)).compute()
                patch_B = row_img[..., :, :default_ov_y, x_start:x_end].max(axis=2).mean(axis=(0,1)).compute()
                
                shift = phase_corr(patch_T, patch_B, search_px)
                shift_y = int(shift[0])
                
                # [중요] 세로 자르는 양 강제 조정이 필요하면 여기서 수정
                cut_amount = default_ov_y - shift_y
                # cut_amount = 100 # 강제 지정 시 주석 해제
                
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
