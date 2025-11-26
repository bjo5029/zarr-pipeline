import zarr
import json
import numpy as np
import dask.array as da
import deconvolution 

def get_zarr_array_safe(z_path):
    try:
        store = zarr.open(str(z_path), mode='r')
        if isinstance(store["0"], zarr.core.Array): return da.from_zarr(store["0"])
        elif "0" in store["0"]: return da.from_zarr(store["0"]["0"])
        else:
            keys = list(store["0"].keys())
            if keys: return da.from_zarr(store["0"][keys[0]])
            raise KeyError("Empty")
    except: return None

def scan_global_min_max(zarr_dirs):
    """Global min/max computation across all tiles"""
    temp_tiles = [get_zarr_array_safe(z) for z in zarr_dirs]
    temp_tiles = [t for t in temp_tiles if t is not None]
    
    if not temp_tiles: return None, None
    
    full_lazy = da.concatenate([da.concatenate(temp_tiles, axis=0)], axis=0)
    g_min, g_max = da.compute(full_lazy.min(), full_lazy.max())
    return float(g_min), float(g_max)

def build_graph(zarr_dirs, cfg, g_min, g_max):
    """Build the final Dask graph"""
    rows = cfg['preprocess']['rows']
    cols = cfg['preprocess']['cols']
    
    psf = deconvolution.get_psf_numpy(sigma=cfg['algorithm']['psf_sigma'])
    grid = [[None for _ in range(cols)] for _ in range(rows)]

    for idx, z_dir in enumerate(zarr_dirs):
        if idx >= rows * cols: break
        row, col = idx // cols, idx % cols
        
        raw_dask = get_zarr_array_safe(z_dir)
        if raw_dask is None: continue

        # 1. Deconv Mapping
        deconv_dask = raw_dask.map_blocks(
            deconvolution.deconv_wrapper_torch,
            psf_cpu=psf,
            iterations=cfg['algorithm']['iterations'],
            global_min=g_min,
            global_max=g_max,
            pad_width=cfg['algorithm']['pad_width'],
            dtype=np.float32
        )

        # 2. Stitching Crop
        _, _, _, y_size, x_size = raw_dask.shape
        overlap = cfg['preprocess']['overlap_pct']
        cut_y = int(y_size * overlap * 0.5)
        cut_x = int(x_size * overlap * 0.5)
        
        y_s = 0 if row == 0 else cut_y
        y_e = None if row == rows - 1 else -cut_y
        x_s = 0 if col == 0 else cut_x
        x_e = None if col == cols - 1 else -cut_x
        
        grid[row][col] = deconv_dask[..., y_s:y_e, x_s:x_e]

    # 3. Merge
    row_arrays = [da.concatenate([g for g in r if g is not None], axis=4) for r in grid if any(g is not None for g in r)]
    final_image = da.concatenate(row_arrays, axis=3)
    
    return final_image

def generate_pyramid(base_zarr_path, levels=3):
    """
    기존에 생성된 Level 0 Zarr를 읽어서 다운샘플링된 피라미드(Level 1, 2...)를 생성하고
    OME-Zarr 메타데이터(.zattrs)를 작성함.
    """
    store = zarr.DirectoryStore(str(base_zarr_path))
    root = zarr.open_group(store=store, mode='r+') # 읽기/쓰기 모드로 열기
    
    # Level 0 (원본) 읽기
    base_array = da.from_zarr(root["0"])
    
    datasets = [{"path": "0"}]
    current_array = base_array
    
    print(f"Generating Pyramid (Levels: {levels})...")

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
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )
        
        datasets.append({"path": str(i)})
        
        # 다음 레벨을 위해 현재 결과를 갱신
        current_array = downsampled

    # 3. OME-NGFF 메타데이터 (.zattrs) 작성
    # 이것이 있어야 Napari 등에서 피라미드로 인식함
    multiscales = [{
        "version": "0.4",
        "name": "stitched_image",
        "datasets": datasets,
        "type": "gaussian", # 혹은 "local_mean"
        "metadata": {
            "method": "mean_downsampling",
            "version": "1.0"
        }
    }]
    
    root.attrs["multiscales"] = multiscales
    print("Pyramid generation complete with Metadata.")
