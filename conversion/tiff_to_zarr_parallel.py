import os
import shutil
import subprocess
import time
from pathlib import Path
import dask
from tqdm import tqdm
from dask.distributed import get_client, as_completed

def get_dir_size(path):
    """디렉토리 크기 계산"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except FileNotFoundError:
        return 0
    return total

def run_cmd(cmd):
    """subprocess 실행 및 에러 처리"""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"Error output: {e.stderr.decode()}")
        raise e

def convert_single_tiff_to_zarr(tiff_path: Path, out_root: Path, bioformats_path: str) -> Path:
    """하나의 TIFF를 bioformats2raw로 OME-Zarr로 변환."""
    name = tiff_path.stem
    out_dir = out_root / f"{name}_zarr"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        bioformats_path,
        str(tiff_path),
        str(out_dir),
        "--resolutions", "1",
        "--downsample-type", "GAUSSIAN",
        "--compression", "blosc",
        "--compression-properties", "cname=zstd",
        "--compression-properties", "clevel=5",
    ]

    try:
        run_cmd(cmd)
    except Exception as e:
        print(f"Failed to convert {name}: {e}")
        return None
        
    return out_dir

def list_tiff_files(input_root_str):
    """중복 제거된 TIFF 파일 목록 반환"""
    input_root = Path(input_root_str)
    files = sorted(list(set([f.resolve() for f in input_root.glob("*.tif") if "._" not in f.name] + 
                            [f.resolve() for f in input_root.glob("*.tiff") if "._" not in f.name])))
    return files

def run_conversion(cfg):
    """전체 변환 실행 함수 (수정됨: 확실한 ProgressBar 적용)"""
    input_root = cfg['paths']['input_root']
    output_root = cfg['paths']['output_root']
    bioformats_path = cfg['paths'].get('bioformats_path', 'bioformats2raw')
     
    Path(output_root).mkdir(exist_ok=True, parents=True)

    files = list_tiff_files(input_root)
    print(f"Found {len(files)} TIFF files. Starting PARALLEL conversion...")
    
    # 1. 지연 작업(Delayed) 생성
    lazy_results = []
    for f in files:
        task = dask.delayed(convert_single_tiff_to_zarr)(f, Path(output_root), bioformats_path)
        lazy_results.append(task)
    
    if not lazy_results:
        return []

    # dask.compute() 대신 client.compute()와 tqdm 사용
    try:
        # main.py에서 만든 Client를 가져옴
        client = get_client()
        
        # 작업을 클러스터에 던짐 (즉시 리턴)
        futures = client.compute(lazy_results)
        
        results = []
        # as_completed: 작업이 끝나는 순서대로 하나씩 뱉어냄
        for future in tqdm(as_completed(futures), total=len(futures), desc="TIFF to Zarr", unit="file"):
            res = future.result() # 결과값 받기
            results.append(res)
            
    except Exception as e:
        # Client 연결 실패하면 기존 방식으로 폴백
        print(f"Progress bar error (fallback to standard): {e}")
        results = dask.compute(*lazy_results)
    
    # 3. 결과 필터링
    zarr_dirs = [res for res in results if res is not None]
    zarr_dirs.sort()
    
    return zarr_dirs
