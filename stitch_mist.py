import numpy as np
import dask.array as da
import zarr
import logging
import sys
import os

# sys.path.append("mist")가 main.py에 되어 있으므로 바로 임포트
import pciam
import stage_model
import translation_refinement
import img_tile
import mle_estimator

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(message)s")

# =========================================================
# 1. 필수 헬퍼 함수들 (get_zarr_array_safe 등)
# =========================================================

def get_zarr_array_safe(z_path):
    """Zarr 파일을 안전하게 Dask Array로 로드"""
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
    """Dask Array 리스트를 받아 Min/Max 계산"""
    valid_arrays = [a for a in dask_arrays if a is not None]
    if not valid_arrays: return 0.0, 1.0
    
    # Lazy하게 전체 연결 후 min/max 계산
    mins = [da.min(a) for a in valid_arrays]
    maxs = [da.max(a) for a in valid_arrays]
    
    g_min, g_max = da.compute(da.min(da.stack(mins)), da.max(da.stack(maxs)))
    return float(g_min), float(g_max)

def generate_pyramid(base_zarr_path, levels=3):
    """이미지 피라미드 생성"""
    store = zarr.DirectoryStore(str(base_zarr_path))
    try:
        root = zarr.open_group(store=store, mode='r+') 
    except:
        return
    
    base_array = da.from_zarr(root["0"])
    datasets = [{"path": "0"}]
    current_array = base_array
    
    print(f"Generating Image Pyramid (Levels: {levels})...")
    for i in range(1, levels + 1):
        # 3D (T, C, Z, Y, X) -> Y, X 축소
        # 만약 2D라면 (Y, X) 축소
        downsample_factor = {axis: 2 for axis in range(current_array.ndim) if axis >= current_array.ndim - 2}
        
        downsampled = da.coarsen(np.mean, current_array, downsample_factor, trim_excess=True)
        downsampled.to_zarr(
            store, 
            component=str(i), 
            overwrite=True, 
            compute=True, 
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )
        datasets.append({"path": str(i)})
        current_array = downsampled

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "stitched_image",
        "datasets": datasets,
        "type": "gaussian",
        "metadata": {"method": "mean_downsampling"}
    }]

# =========================================================
# 2. MIST 어댑터 클래스 (InMemoryTile, InMemoryGrid)
# =========================================================

class InMemoryTile:
    """
    MIST의 img_tile.Tile 클래스 호환용 어댑터
    """
    def __init__(self, r, c, image_data, name):
        self.r = r
        self.c = c
        self.name = name
        self.image_data = image_data # 2D Numpy Array (MIP 결과)
        
        # MIST 알고리즘용 속성
        self.west_translation = None
        self.north_translation = None
        self.abs_x = 0
        self.abs_y = 0
        self.ncc = np.nan

    def get_image(self) -> np.ndarray:
        return self.image_data

    def get_translation(self, direction):
        return self.west_translation if direction == 'HORIZONTAL' else self.north_translation
    
    def get_max_translation_ncc(self):
        wt = self.west_translation
        nt = self.north_translation
        nccs = []
        if wt and not np.isnan(wt.ncc): nccs.append(wt.ncc)
        if nt and not np.isnan(nt.ncc): nccs.append(nt.ncc)
        return max(nccs) if nccs else np.nan

    # 위치 관계 확인 메서드들 (typing.Self 이슈 해결됨)
    def north_of(self, other: "InMemoryTile") -> bool:
        if other is None: raise RuntimeError("Other Tile is None")
        return self.r + 1 == other.r and self.c == other.c

    def south_of(self, other: "InMemoryTile") -> bool:
        if other is None: raise RuntimeError("Other Tile is None")
        return self.r - 1 == other.r and self.c == other.c

    def east_of(self, other: "InMemoryTile") -> bool:
        if other is None: raise RuntimeError("Other Tile is None")
        return self.r == other.r and self.c - 1 == other.c

    def west_of(self, other: "InMemoryTile") -> bool:
        if other is None: raise RuntimeError("Other Tile is None")
        return self.r == other.r and self.c + 1 == other.c

    def get_peak(self, other: "InMemoryTile"):
        if self.north_of(other): return other.north_translation
        if self.south_of(other): return self.north_translation
        if self.east_of(other): return self.west_translation
        if self.west_of(other): return other.west_translation
        return None

    def update_absolute_position(self, other: "InMemoryTile"):
        if other is None:
            raise RuntimeError("Other Tile is None")

        x = other.abs_x
        y = other.abs_y

        if self.north_of(other):
            peak = other.north_translation
            self.abs_x = x - peak.x
            self.abs_y = y - peak.y
        elif self.south_of(other):
            peak = self.north_translation
            self.abs_x = x + peak.x
            self.abs_y = y + peak.y
        elif self.west_of(other):
            peak = other.west_translation
            self.abs_x = x - peak.x
            self.abs_y = y - peak.y
        elif self.east_of(other):
            peak = self.west_translation
            self.abs_x = x + peak.x
            self.abs_y = y + peak.y

class InMemoryGrid:
    """MIST의 img_grid.TileGrid 호환"""
    def __init__(self, tiles_list, rows, cols):
        self.width = cols
        self.height = rows
        self.tiles = [[None for _ in range(cols)] for _ in range(rows)]
        for t in tiles_list:
            self.tiles[t.r][t.c] = t

    def get_tile(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.tiles[r][c]
        return None

    def get_image_shape(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.tiles[r][c] is not None:
                    return self.tiles[r][c].get_image().shape
        return (0, 0)
    
    def get_image_size_per_direction(self, direction):
        h, w = self.get_image_shape()
        return w if direction == 'HORIZONTAL' else h
        
    def get_num_valid_tiles(self):
        return sum([1 for r in range(self.height) for c in range(self.width) if self.tiles[r][c]])

class MockArgs:
    def __init__(self, rows, cols, overlap_pct):
        self.grid_width = cols
        self.grid_height = rows
        self.num_fft_peaks = 2
        self.overlap_uncertainty = 5.0
        self.valid_correlation_threshold = 0.5
        self.stage_repeatability = None
        self.horizontal_overlap = None
        self.vertical_overlap = None   
        self.num_hill_climbs = 0

# =========================================================
# 3. 메인 로직 (build_graph)
# =========================================================

def build_graph(tiles, cfg, g_min, g_max):
    rows = cfg['preprocess']['rows']
    cols = cfg['preprocess']['cols']
    overlap_pct = cfg['preprocess']['overlap_pct']
    
    if len(tiles) != rows * cols:
        print(f"Error: Tile count ({len(tiles)}) does not match grid ({rows}x{cols})")
        return None

    _, _, Z_dim, H, W = tiles[0].shape
    print(f"Starting MIST Stitching for {rows}x{cols} tiles (Size: {H}x{W}, Depth: {Z_dim})...")

    # A. 2D MIP 생성
    print(" [MIST] Generating 2D MIPs for alignment calculation...")
    mip_tasks = []
    for tile in tiles:
        # Z축 Max Projection (가장 선명한 정보 사용)
        mip = tile[0, 0, ...].max(axis=0) 
        mip_tasks.append(mip)
    
    mips_result = da.compute(mip_tasks)[0]

    # B. MIST 알고리즘 실행
    print(" [MIST] Running PCIAM (Phase Correlation)...")
    mist_tiles = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            mt = InMemoryTile(r, c, mips_result[idx], f"Tile_r{r}_c{c}")
            mist_tiles.append(mt)
            
    mist_grid = InMemoryGrid(mist_tiles, rows, cols)
    mist_args = MockArgs(rows, cols, overlap_pct)

    # PCIAM
    pciam_solver = pciam.PciamSequential(mist_args)
    pciam_solver.execute(mist_grid)

    # Stage Model
    print(" [MIST] Computing Stage Model (Filtering outliers)...")
    sm = stage_model.StageModel(mist_args, mist_grid)
    sm.build()

    # MST (Global Positions)
    print(" [MIST] Computing Global Positions (Minimum Spanning Tree)...")
    gp = translation_refinement.GlobalPositions(mist_grid)
    gp.traverse_minimum_spanning_tree()

    # C. 최종 캔버스 생성 및 배치
    print(" [MIST] Assembling final 3D volume...")

    stitched_rows = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            idx = r * cols + c
            tile = tiles[idx]
            
            # Deconvolution & FFC 적용
            # (필요시 여기서 FFC 등 적용. 현재는 원본 tile 사용)
            
            if c == 0:
                row_tiles.append(tile)
            else:
                prev_mt = mist_grid.get_tile(r, c-1)
                curr_mt = mist_grid.get_tile(r, c)
                
                # MIST가 계산한 X 거리
                dx = curr_mt.abs_x - prev_mt.abs_x
                overlap = W - dx
                cut = int(max(0, overlap))
                
                # 잘라내고 붙이기
                tile_cropped = tile[..., :, :, cut:]
                row_tiles.append(tile_cropped)
        
        row_img = da.concatenate(row_tiles, axis=4) # X축
        stitched_rows.append(row_img)

    # Col Stitching
    final_cols = []
    min_width = min(row.shape[-1] for row in stitched_rows)
    stitched_rows = [row[..., :min_width] for row in stitched_rows]
    
    for r in range(rows):
        row_img = stitched_rows[r]
        if r == 0:
            final_cols.append(row_img)
        else:
            prev_mt_0 = mist_grid.get_tile(r-1, 0)
            curr_mt_0 = mist_grid.get_tile(r, 0)
            
            dy = curr_mt_0.abs_y - prev_mt_0.abs_y
            overlap_y = H - dy
            cut_y = int(max(0, overlap_y))
            
            row_cropped = row_img[..., :, cut_y:, :]
            final_cols.append(row_cropped)

    final_image = da.concatenate(final_cols, axis=3) # Y축
    return final_image
