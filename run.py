import os
import sys
import re
import yaml
import zarr
import torch
import numpy as np
import dask.array as da
import time
from pathlib import Path
from dask.distributed import Client

import conversion.tiff_to_zarr_parallel as tiff_to_zarr 

# # mist 폴더를 라이브러리 경로에 추가
# current_dir = Path(__file__).resolve().parent
# mist_dir = current_dir / "stitching/mist"
# if str(mist_dir) not in sys.path:
#     sys.path.append(str(mist_dir))
import stitching.stitch_mist as stitch 

# import stitching.stitch_sift as stitch

import preprocessing.flat_field_correction

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def parse_filename(filepath):
    name = filepath.stem
    # W{well}F{field}T{time}Z{z}C{ch} 패턴 파싱
    pattern = r"W(\d+)F(\d+)T(\d+)Z(\d+)C(\d+)"
    match = re.search(pattern, name)
    if match:
        return {
            "W": int(match.group(1)),
            "F": int(match.group(2)),
            "T": int(match.group(3)),
            "Z": int(match.group(4)),
            "C": int(match.group(5)),
            "path": filepath
        }
    return None

def main():
    total_start_time = time.time()

    # 1. Config & Setup
    if not Path("config.yaml").exists():
        print("config.yaml not found.")
        return
    cfg = load_config()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['system']['gpu_id'])
    
    # Dask Client
    client = Client(
        processes=cfg['system'].get('use_processes', True), # 기본값 True 추천
        n_workers=cfg['system']['dask_workers'],
        threads_per_worker=cfg['system']['dask_threads']
    )
    print(f"Dashboard: {client.dashboard_link}")

    # =========================================================================
    # Step 1: Zarr 변환 단계 
    # =========================================================================
    SKIP_CONVERSION = True 

    output_root = Path(cfg['paths']['output_root'])
    zarr_dirs = []

    if SKIP_CONVERSION and output_root.exists():
        print("\n[Step 1] Skipping Conversion (Loading existing Zarrs)...")
        zarr_dirs = sorted([p for p in output_root.glob("*_zarr") if p.is_dir()])
        print(f"Loaded {len(zarr_dirs)} existing Zarr directories.")
    
    if not zarr_dirs:
        if SKIP_CONVERSION:
            print("Warning: SKIP_CONVERSION is True but no files found. Running conversion anyway.")
        print("\n[Step 1] Converting TIFF to Zarr...")
        zarr_dirs = tiff_to_zarr.run_conversion(cfg)
    
    if not zarr_dirs:
        print("No data to process.")
        sys.exit(1)

    # 3. 데이터 그룹화
    print("\n[Step 2] Organizing Data Structure...")
    data_map = {} 

    for z_path in zarr_dirs:
        original_stem = z_path.stem.replace("_zarr", "")
        meta = parse_filename(Path(original_stem))
        
        if meta:
            c, f, z = meta['C'], meta['F'], meta['Z']
            if c not in data_map: data_map[c] = {}
            if f not in data_map[c]: data_map[c][f] = []
            data_map[c][f].append((z, z_path))
    
    sorted_channels = sorted(data_map.keys())
    print(f"Found Channels: {sorted_channels}")

    # 4. 채널별 처리
    for ch in sorted_channels:
        print(f"\n=== Processing Channel {ch} ===")
        fields_dict = data_map[ch]
        required_fields = cfg['preprocess']['rows'] * cfg['preprocess']['cols']
        field_arrays = [] 
        
        # Field 순서대로 처리
        for i in range(1, required_fields + 1):
            if i not in fields_dict:
                print(f"Warning: Field {i} is missing in Channel {ch}. Filling with zeros not implemented yet.")
                continue
            
            z_list = fields_dict[i]
            z_list.sort(key=lambda x: x[0]) 
            
            slices = []
            for z_idx, z_p in z_list:
                arr = stitch.get_zarr_array_safe(z_p)
                if arr is None: continue
                
                while arr.ndim > 2:
                    arr = arr[0]
                
                slices.append(arr)
            
            if not slices:
                print(f"Warning: No valid slices for Field {i}")
                continue

            stack_z = da.stack(slices, axis=0) 
            full_stack = stack_z[None, None, ...] 
            field_arrays.append(full_stack)

        if not field_arrays:
            print(f"Skipping Channel {ch} (No data)")
            continue

        # Min/Max Scan
        print(f"Scanning Min/Max for Channel {ch}...")
        g_min, g_max = stitch.scan_global_min_max_arrays(field_arrays)
        print(f"Channel {ch} Range: {g_min:.2f} ~ {g_max:.2f}")

        # Stitching
        print(f"Stitching Channel {ch}...")
        try:
            # =================================================================
            # [수정됨] MIST 방식 호출 (통합된 build_graph 하나만 호출하면 됨)
            # =================================================================
            
            # FFC 더미 데이터 생성 등은 stitch_mist.py 내부에서 처리하거나
            # 필요하다면 여기서 넘겨줄 수 있으나, 현재 stitch_mist 구조상 
            # 내부에서 생성하므로 인자만 잘 넘기면 됩니다.
            
            final_image = stitch.build_graph(field_arrays, cfg, g_min, g_max)
            
        except Exception as e:
            print(f"Stitching Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if final_image is None:
            continue

        # Save
        out_filename = f"Channel_{ch}_stitched.zarr"
        save_path = Path(cfg['paths']['output_root']) / out_filename
        print(f"Saving to {save_path}...")
        
        final_image.to_zarr(
            str(save_path),
            component="0",
            overwrite=True,
            compute=True,
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )
        
        # Pyramid
        stitch.generate_pyramid(save_path, levels=4)
        print(f"Channel {ch} Done.")

    print("\nALL FINISHED")

    total_end_time = time.time()
    elapsed_time = total_end_time - total_start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print("="*50)
    print(f"Total Execution Time: {elapsed_time:.2f} seconds ({minutes}m {seconds}s)")
    print("="*50)

if __name__ == "__main__":
    main()
