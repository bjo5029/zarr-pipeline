import os
import sys
import yaml
import zarr
import torch
from pathlib import Path
from dask.distributed import Client

# import local modules
import tiff_to_zarr
import stitch

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Config & GPU Setup
    if not Path("config.yaml").exists():
        print("config.yaml not found.")
        return
    cfg = load_config()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['system']['gpu_id'])
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not found.")
        sys.exit(1)

    # 2. Dask Client
    client = Client(
        processes=False,
        n_workers=cfg['system']['dask_workers'],
        threads_per_worker=cfg['system']['dask_threads']
    )
    print(f"Dashboard: {client.dashboard_link}")

    # 3. Run Step 1: Convert
    print("\n[Step 1] Converting TIFF to Zarr...")
    zarr_dirs = tiff_to_zarr.run_conversion(cfg)
    if not zarr_dirs:
        print("No data to process.")
        sys.exit(1)

    # 4. Run Step 2: Min/Max Scan
    print("\n[Step 2] Scanning Global Min/Max...")
    g_min, g_max = stitch.scan_global_min_max(zarr_dirs)
    if g_min is None:
        print("Error reading Zarrs.")
        sys.exit(1)
    print(f"Range: {g_min:.2f} ~ {g_max:.2f}")

    # 5. Run Step 3: Build Graph & Execute
    print("\n[Step 3] Building Graph & Executing...")
    final_image = stitch.build_graph(zarr_dirs, cfg, g_min, g_max)
    
    save_path = Path(cfg['paths']['output_root']) / cfg['paths']['final_filename']
    print(f"Saving to {save_path}...")

    final_image.to_zarr(
        str(save_path),
        component="0",
        overwrite=True,
        compute=True,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    )

    # [추가] 피라미드 생성 단계
    print("\n[Step 4] Generating Image Pyramid...")
    # levels=4 정도로 설정하면 (원본 -> 1/2 -> 1/4 -> 1/8 -> 1/16)까지 생성됨
    stitch.generate_pyramid(save_path, levels=4)
    
    print("FINISH")

if __name__ == "__main__":
    main()
    