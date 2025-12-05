import numpy as np
import dask.array as da
import zarr
import logging
import sys
import os

from .mist import pciam, stage_model, translation_refinement, img_tile, mle_estimator
from . import linear_blend

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

def build_graph_center_cut(tiles, cfg, g_min, g_max):
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

    # [수정] 1. 가로(Row) 스티칭: 양쪽 절반씩 자르기 (Center Cut)
    for r in range(rows):
        # 해당 행의 원본 타일들을 리스트로 가져옴
        current_row_tiles = [tiles[r * cols + c] for c in range(cols)]
        
        # 왼쪽(c-1)과 오른쪽(c) 타일 사이의 오버랩을 계산하여 절반씩 자름
        for c in range(1, cols):
            prev_mt = mist_grid.get_tile(r, c - 1)
            curr_mt = mist_grid.get_tile(r, c)
            
            # X축 오버랩 계산 (W - 거리)
            dx = curr_mt.abs_x - prev_mt.abs_x
            overlap = W - dx
            
            # 오버랩이 있을 경우에만 Center Cut 수행
            if overlap > 0:
                cut = int(overlap)
                half_left = cut // 2          # 앞 타일의 뒷부분(오른쪽)에서 자를 양
                half_right = cut - half_left  # 뒤 타일의 앞부분(왼쪽)에서 자를 양
                
                # [Center Cut]
                # 1) 앞 타일: 오른쪽 끝부분 잘라내기 ([..., :-half_left])
                if half_left > 0:
                    current_row_tiles[c - 1] = current_row_tiles[c - 1][..., :, :, :-half_left]
                
                # 2) 뒤 타일: 왼쪽 앞부분 잘라내기 ([..., half_right:])
                if half_right > 0:
                    current_row_tiles[c] = current_row_tiles[c][..., :, :, half_right:]
        
        # 잘라낸 타일들을 가로(X축, axis=4)로 연결
        row_img = da.concatenate(current_row_tiles, axis=4)
        stitched_rows.append(row_img)

    # [수정] 2. 세로(Col) 스티칭: 양쪽 절반씩 자르기 (Center Cut)
    # 먼저 가로 길이를 최소값으로 맞춤 (들쑥날쑥할 수 있으므로)
    min_width = min(row.shape[-1] for row in stitched_rows)
    stitched_rows = [row[..., :min_width] for row in stitched_rows]
    
    # 위(r-1)와 아래(r) 행 사이의 오버랩을 계산하여 절반씩 자름
    for r in range(1, rows):
        # Y축 위치는 첫 번째 컬럼(0)을 기준으로 계산 (일반적으로 행 전체가 같이 움직임)
        prev_mt = mist_grid.get_tile(r - 1, 0)
        curr_mt = mist_grid.get_tile(r, 0)
        
        # Y축 오버랩 계산 (H - 거리)
        dy = curr_mt.abs_y - prev_mt.abs_y
        overlap_y = H - dy
        
        if overlap_y > 0:
            cut = int(overlap_y)
            half_top = cut // 2           # 위쪽 행의 아랫부분 자르기
            half_bottom = cut - half_top  # 아래쪽 행의 윗부분 자르기
            
            # [Center Cut]
            # 1) 위쪽 행: 아랫부분 잘라내기 (Axis 3 = Y축)
            if half_top > 0:
                stitched_rows[r - 1] = stitched_rows[r - 1][..., :, :-half_top, :]
            
            # 2) 아래쪽 행: 윗부분 잘라내기
            if half_bottom > 0:
                stitched_rows[r] = stitched_rows[r][..., :, half_bottom:, :]

    # 최종 세로(Y축, axis=3) 연결
    final_image = da.concatenate(stitched_rows, axis=3) 
    return final_image


# =========================================================
# 3. 메인 로직 (build_graph)
# =========================================================

def build_graph_linear_blend(tiles, cfg, g_min, g_max):
    rows = cfg['preprocess']['rows']
    cols = cfg['preprocess']['cols']
    overlap_pct = cfg['preprocess']['overlap_pct']
    
    if len(tiles) != rows * cols:
        print(f"Error: Tile count ({len(tiles)}) does not match grid ({rows}x{cols})")
        return None

    _, _, Z_dim, H, W = tiles[0].shape
    print(f"Starting MIST Stitching for {rows}x{cols} tiles (Size: {H}x{W}, Depth: {Z_dim})...")

    # A. 2D MIP 생성 (Brightfield/Phase Contrast 최적화)
    print(" [MIST] Generating 2D MIPs (Inverted for Brightfield)...")
    mip_tasks = []
    for tile in tiles:
        # [수정] 명시야 이미지는 배경이 밝고 세포가 어두움 -> 반전시켜서 계산해야 정확함
        # 1. (T, C, Z, Y, X) -> (Z, Y, X) 데이터 추출 (T=0, C=0 기준)
        stack_data = tile[0, 0, ...] 
        
        # 2. 이미지 반전 (Invert): 밝은 배경(255) -> 어두움(0), 어두운 세포(0) -> 밝음(255)
        # 해당 타일의 최대값을 기준으로 반전 (Dask 지연 연산 호환)
        max_val = da.max(stack_data)
        inverted_data = max_val - stack_data
        
        # 3. 반전된 데이터로 Max Projection 수행
        mip = inverted_data.max(axis=0)
        mip_tasks.append(mip)
    
    mips_result = da.compute(mip_tasks)[0]

    # B. MIST 알고리즘 실행 (위치 계산)
    print(" [MIST] Running PCIAM (Phase Correlation)...")
    mist_tiles = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # InMemoryTile은 계산된 MIP 결과(numpy array)를 들고 있음
            mt = InMemoryTile(r, c, mips_result[idx], f"Tile_r{r}_c{c}")
            mist_tiles.append(mt)
            
    mist_grid = InMemoryGrid(mist_tiles, rows, cols)
    mist_args = MockArgs(rows, cols, overlap_pct)

    # 1. PCIAM (대략적 위치)
    pciam_solver = pciam.PciamSequential(mist_args)
    pciam_solver.execute(mist_grid)

    # 2. Stage Model (이상값 제거)
    print(" [MIST] Computing Stage Model (Filtering outliers)...")
    sm = stage_model.StageModel(mist_args, mist_grid)
    sm.build()

    # 3. MST (최종 좌표 확정)
    print(" [MIST] Computing Global Positions (Minimum Spanning Tree)...")
    gp = translation_refinement.GlobalPositions(mist_grid)
    gp.traverse_minimum_spanning_tree()

    # C. 최종 캔버스 생성 및 배치 (Linear Blend 적용)
    print(" [MIST] Assembling final 3D volume using Linear Blend...")

    stitched_rows = []

    # 1. 가로 방향 (Row Stitching)
    for r in range(rows):
        # 행의 첫 번째 타일로 시작
        current_row_img = tiles[r * cols + 0]
        
        for c in range(1, cols):
            next_tile = tiles[r * cols + c]
            
            # 오버랩 계산
            prev_mt = mist_grid.get_tile(r, c - 1)
            curr_mt = mist_grid.get_tile(r, c)
            
            # MIST가 계산한 X 거리 차이
            dx = curr_mt.abs_x - prev_mt.abs_x
            overlap = W - dx
            
            # [수정] Center Cut 대신 Linear Blend 모듈 호출
            # axis=4 : 가로(Width) 방향 연결
            current_row_img = linear_blend.blend_overlap(
                current_row_img, next_tile, int(overlap), axis=4
            )
            
        stitched_rows.append(current_row_img)

    # 2. 세로 방향 (Column Stitching)
    # 가로 스티칭 결과물들의 폭이 미세하게 다를 수 있으므로 최소값으로 맞춤 (Crop)
    min_width = min(row.shape[-1] for row in stitched_rows)
    stitched_rows = [row[..., :min_width] for row in stitched_rows]
    
    final_image = stitched_rows[0]
    for r in range(1, rows):
        next_row = stitched_rows[r]
        
        # 오버랩 계산 (Y축)
        # Y축 위치는 첫 번째 컬럼(0)을 기준으로 계산
        prev_mt = mist_grid.get_tile(r - 1, 0)
        curr_mt = mist_grid.get_tile(r, 0)
        
        dy = curr_mt.abs_y - prev_mt.abs_y
        overlap_y = H - dy
        
        # [수정] Linear Blend 모듈 호출
        # axis=3 : 세로(Height) 방향 연결
        final_image = linear_blend.blend_overlap(
            final_image, next_row, int(overlap_y), axis=3
        )

    return final_image
