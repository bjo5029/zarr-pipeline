import os
import shutil
import subprocess
import time
from pathlib import Path

def get_dir_size(path):
    """디렉토리 크기 계산 (MB 단위용)"""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total

def run_cmd(cmd):
    """subprocess 실행 및 에러 처리"""
    try:
        # 로그가 너무 길어지는 것을 방지하기 위해 stdout은 숨김 (필요시 제거)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"Error output: {e.stderr.decode()}")
        raise e

def convert_single_tiff_to_zarr(tiff_path: Path, out_root: Path, bioformats_path: str) -> Path:
    """
    하나의 TIFF를 bioformats2raw로 OME-Zarr로 변환.
    resolutions=1, zstd 압축 사용.
    """
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

    start = time.time()
    try:
        run_cmd(cmd)
    except Exception:
        return None
        
    end = time.time()

    tiff_size = os.path.getsize(tiff_path)
    zarr_size = get_dir_size(out_dir)

    print(f"[{name}] TIFF: {tiff_size / (1024**2):.2f} MB -> "
          f"Zarr: {zarr_size / (1024**2):.2f} MB "
          f"({end - start:.2f} s)")

    return out_dir

def list_tiff_files(input_root_str):
    """중복 제거된 TIFF 파일 목록 반환"""
    input_root = Path(input_root_str)
    files = sorted(list(set([f.resolve() for f in input_root.glob("*.tif") if "._" not in f.name] + 
                            [f.resolve() for f in input_root.glob("*.tiff") if "._" not in f.name])))
    return files

def run_conversion(cfg):
    """전체 변환 실행 함수"""
    input_root = cfg['paths']['input_root']
    output_root = cfg['paths']['output_root']
    bioformats_path = cfg['paths'].get('bioformats_path', 'bioformats2raw') # 설정 없으면 기본값
    
    Path(output_root).mkdir(exist_ok=True, parents=True)

    files = list_tiff_files(input_root)
    print(f"Found {len(files)} TIFF files. Starting conversion...")
    
    zarr_dirs = []
    for f in files:
        z = convert_single_tiff_to_zarr(f, Path(output_root), bioformats_path)
        if z: zarr_dirs.append(z)
    
    zarr_dirs.sort()
    return zarr_dirs