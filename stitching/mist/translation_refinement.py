import argparse
import numpy as np
import logging
import time
from abc import ABC
import copy
import enum

# local imports
from . import img_grid
from . import img_tile
from . import stage_model
from . import pciam
from . import utils


class HillClimbDirection(enum.Enum):
    """
    Defines hill climbing direction using cartesian coordinates when observing a two dimensional
    grid where the upper left corner is 0,0. Moving north -1 in the y-direction, south +1 in the
    y-direction, west -1 in the x-direction, and east +1 in the x-direction.
    """
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    NoMove = (0, 0)

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Refine(ABC):
    @staticmethod
    def hill_climb_worker(i1: np.ndarray, i2: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int, start_x: int, start_y: int, cache: np.ndarray) -> img_tile.Peak:
        """
        Computes cross correlation search with hill climbing
        :param i1: image 1 (ego)
        :param i2: image 1 (north or west neigbor)
        :param x_min: min x boundary
        :param x_max: max x boundary
        :param y_min: min y boundary
        :param y_max: max y boundary
        :param start_x: start x position for the hill climb
        :param start_y: start y position for the hill climb
        :param cache: 2d array of np.float32 storing the ncc values for each x,y
        :return:
        """
        """
        역할: 현재 위치에서 시작하여 NCC 점수가 더 높은 쪽으로 조금씩 이동하는 탐욕적 탐색(Greedy Search) 함수.
        동작:
            현재 위치 (start_x, start_y)의 NCC를 계산.
            동, 서, 남, 북 4방향의 이웃 좌표를 검사.
            이웃 중 현재보다 더 높은 점수를 가진 곳이 있다면 그쪽으로 이동.
            더 이상 높은 곳이 없을 때까지(봉우리에 도달할 때까지) 반복. 
            cache를 사용하여 이미 계산한 좌표는 다시 계산하지 않음.
        """

        best_peak = img_tile.Peak(ncc=np.nan, x=start_x, y=start_y)

        # walk hill climb until we reach a top
        while True:
            cur_direction = HillClimbDirection.NoMove

            # translate to 0-based coordinates
            cur_x_idx = best_peak.x - x_min
            cur_y_idx = best_peak.y - y_min

            # check the current location
            best_peak.ncc = cache[cur_y_idx, cur_x_idx]
            if np.isnan(best_peak.ncc):
                best_peak.ncc = pciam.PCIAM.compute_cross_correlation(i1, i2, best_peak.x, best_peak.y)
                cache[cur_y_idx, cur_x_idx] = best_peak.ncc

            search_center = copy.deepcopy(best_peak)
            # Check each direction and move based on highest correlation
            for d in HillClimbDirection._member_names_:
                dir = HillClimbDirection[d]
                if dir == HillClimbDirection.NoMove:
                    continue

                # Check if moving dir is in bounds
                new_x = search_center.x + dir.x
                new_y = search_center.y + dir.y
                if new_y >= y_min and new_y <= y_max and new_x >= x_min and new_x <= x_max:
                    # Check if we have already computed the peak at dir
                    ncc = cache[cur_y_idx + dir.y, cur_x_idx + dir.x]
                    if np.isnan(ncc):
                        ncc = pciam.PCIAM.compute_cross_correlation(i1, i2, new_x, new_y)
                        cache[cur_y_idx + dir.y, cur_x_idx + dir.x] = ncc
                    if ncc > best_peak.ncc:
                        best_peak.ncc = ncc
                        best_peak.x = new_x
                        best_peak.y = new_y
                        cur_direction = dir

            if cur_direction == HillClimbDirection.NoMove:
                # if the direction was a NoMove, then we are done
                break

        if np.isnan(best_peak.ncc):
            # no best peak was found, use center of search area
            best_peak.x = int((x_max + x_min) / 2)
            best_peak.y = int((y_max + y_min) / 2)
            best_peak.ncc = -1.0
        return best_peak

    @staticmethod
    def multipoint_hill_climb(num_hill_climbs: int, t1: img_tile.Tile, t2: img_tile.Tile, x_min: int, x_max: int, y_min: int, y_max: int, start_x: int, start_y: int) -> img_tile.Peak:
        """
        역할: Hill Climbing이 낮은 봉우리(Local Maximum)에 갇히는 것을 막기 위해, 여러 시작점에서 동시에 등반을 시킴.
        동작:
            PCIAM이 알려준 위치에서 한 번 등반함.
            num_hill_climbs 횟수만큼 무작위 위치(Random starting point)를 골라 거기서부터 또 등반을 시킴.
            모든 등반 결과 중 가장 높은 NCC 점수를 가진 위치를 최종 선택함.
        """
        i1 = t1.get_image()
        i2 = t2.get_image()
        img_shape = i1.shape
        height = img_shape[0]
        width = img_shape[1]

        # clamp bounds to valid range
        x_min = np.clip(x_min, -(width - 1), width - 1)
        x_max = np.clip(x_max, -(width - 1), width - 1)
        y_min = np.clip(y_min, -(height - 1), height - 1)
        y_max = np.clip(y_max, -(height - 1), height - 1)

        # create array of peaks +1 for inclusive
        cache = np.nan * np.ones((x_max - x_min + 2, y_max - y_min + 2), dtype=np.float32)  # +2 to be inclusive of both end points

        peak_results = list()
        # evaluate the starting point hill climb
        peak = Refine.hill_climb_worker(i1, i2, x_min, x_max, y_min, y_max, start_x, start_y, cache)
        peak_results.append(peak)

        # perform the random starting point multipoint hill climbing
        for i in range(num_hill_climbs - 1):
            # generate random starting point
            start_x = np.random.randint(x_min, x_max + 1)
            start_y = np.random.randint(y_min, y_max + 1)

            peak = Refine.hill_climb_worker(i1, i2, x_min, x_max, y_min, y_max, start_x, start_y, cache)
            peak_results.append(peak)

        # find the best correlation and translation from the hill climb ending points
        best_index = np.argmax([peak.ncc for peak in peak_results])
        best_peak = peak_results[best_index]

        # determine how many converged
        converged = np.sum([1 for peak in peak_results if peak.x == best_peak.x and peak.y == best_peak.y])
        logging.info("Translation Hill Climb ({}, ({}) had {}/{} hill climbs converge with best ncc = {}".format(t1.name, t2.name, converged, num_hill_climbs, best_peak.ncc))
        return best_peak

    @staticmethod
    def optimize_direction(tile: img_tile.Tile, other: img_tile.Tile, direction: str, repeatability: int, num_hill_climbs: int) -> img_tile.Peak:
        """
        역할: 타일 간의 최종 위치를 확정함.
        동작:
            탐색 범위(repeatability - 기계적 오차 범위)를 설정하고 multipoint_hill_climb를 호출하여 최적의 위치를 찾음.
        """
        assert direction in ['west', 'north']
        relevant_translation = tile.west_translation if direction == 'west' else tile.north_translation
        orig_peak = copy.deepcopy(relevant_translation)
        x_min = orig_peak.x - repeatability
        x_max = orig_peak.x + repeatability
        y_min = orig_peak.y - repeatability
        y_max = orig_peak.y + repeatability

        new_peak = Refine.multipoint_hill_climb(num_hill_climbs, other, tile, x_min, x_max, y_min, y_max, orig_peak.x, orig_peak.y)

        # If the old correlation was a number, then it was a good translation.
        # Increment the new translation by the value of the old correlation to increase beyond 1
        # This will enable these tiles to have higher priority in minimum spanning tree search
        if not np.isnan(orig_peak.ncc):
            new_peak.ncc += 3.0

        return new_peak


class RefineSequential(Refine):
    def __init__(self, args: argparse.Namespace, tile_grid: img_grid.TileGrid, sm: "stage_model.StageModel"):
        self.args = args
        self.tile_grid = tile_grid
        self.sm = sm

    def execute(self):
        logging.info("Starting Translation Refinement")
        start_time = time.time()
        # iterate over the tile grid
        for r in range(self.args.grid_height):
            for c in range(self.args.grid_width):
                tile = self.tile_grid.get_tile(r, c)
                if tile is None:
                    continue

                west = self.tile_grid.get_tile(r, c - 1)
                if west is not None:
                    # optimize with west neighbor
                    tile.west_translation = Refine.optimize_direction(tile, west, 'west', self.sm.repeatability, self.args.num_hill_climbs)

                north = self.tile_grid.get_tile(r - 1, c)
                if north is not None:
                    # optimize with north neighbor
                    tile.north_translation = Refine.optimize_direction(tile, north, 'north', self.sm.repeatability, self.args.num_hill_climbs)

        elapsed_time = time.time() - start_time
        logging.info("Translation Refinement took {} seconds".format(elapsed_time))


class RefineParallel(Refine):
    def __init__(self, args: argparse.Namespace, tile_grid: img_grid.TileGrid, sm: "stage_model.StageModel"):
        self.args = args
        self.tile_grid = tile_grid
        self.sm = sm

    @staticmethod
    def _worker(tile: img_tile.Tile, other: img_tile.Tile, direction: str, repeatability: int, num_hill_climbs: int, r: int, c: int) -> tuple[img_tile.Peak, int, int, str]:
        return Refine.optimize_direction(tile, other, direction, repeatability, num_hill_climbs), r, c, direction


    def execute(self):
        logging.info("Starting Translation Refinement")
        start_time = time.time()

        worker_input_list = list()
        # iterate over the tile grid
        for r in range(self.args.grid_height):
            for c in range(self.args.grid_width):
                tile = self.tile_grid.get_tile(r, c)
                if tile is None:
                    continue

                west = self.tile_grid.get_tile(r, c - 1)
                if west is not None:
                    # optimize with west neighbor
                    worker_input_list.append((tile, west, 'west', self.sm.repeatability, self.args.num_hill_climbs, r, c))

                north = self.tile_grid.get_tile(r - 1, c)
                if north is not None:
                    # optimize with north neighbor
                    worker_input_list.append((tile, north, 'north', self.sm.repeatability, self.args.num_hill_climbs, r, c))

        # results = list()
        # for worker_input in worker_input_list:
        #     results.append(self._worker(*worker_input))
        import multiprocessing
        with multiprocessing.Pool(processes=utils.get_num_workers()) as pool:
            # perform the work in parallel
            results = pool.starmap(self._worker, worker_input_list)

        for result in results:
            peak, r, c, direction = result
            tile = self.tile_grid.get_tile(r, c)
            if tile is not None:
                if direction == 'west':
                    tile.west_translation = peak
                elif direction == 'north':
                    tile.north_translation = peak

        elapsed_time = time.time() - start_time
        logging.info("Translation Refinement took {} seconds".format(elapsed_time))


class GlobalPositions():
    """
    개별 타일 간의 위치(상대 좌표)는 구했지만, 전체 큰 그림(절대 좌표)을 그리기 위해 최소 신장 트리(MST) 알고리즘을 사용함.
    """

    _dx = [0, -1, 1, 0]
    _dy = [-1, 0, 0, 1]

    def __init__(self, tile_grid: img_grid.TileGrid):
        self.tile_grid = tile_grid

    def get_release_count(self, r, c):
        """
        Computes the release count that is based on how many neighbors this tile has assuming that
        # there are tiles on the 4 cardinal directions (north, south, east, west).
        # If a tile is on the edge of the grid, then its release count is 3, if the tile is on a corner
        # then the release count is 2, if the tile is in the center then the release count is 4.
        """
        """
        역할: 특정 타일이 연결되어야 할 이웃의 개수를 셈. MST 탐색 시 타일의 방문 완료 여부를 판단하는 데 쓰임.
        """
        if self.tile_grid.get_tile(r, c) is None:
            return 0

        release_count = (0 if r == 0 else 1) + \
                        (0 if c == 0 else 1) + \
                        (0 if r == self.tile_grid.height - 1 else 1) + \
                        (0 if c == self.tile_grid.width - 1 else 1)

        # handle cases where neighbor tiles are missing
        if r > 1 and self.tile_grid.get_tile(r - 1, c) is None:
            release_count -= 1
        if c > 1 and self.tile_grid.get_tile(r, c - 1) is None:
            release_count -= 1
        return release_count

    def traverse_next_mst_tile(self, frontier_tiles: set[img_tile.Tile], visited_tiles: np.ndarray, mst_release_counts: np.ndarray, mst_size: int) -> int:
        """
        Traverses to the next tile in the minimum spanning tree
        """
        """
        역할: 현재 연결된 타일들의 집합(MST)에서 가장 신뢰할 수 있는(NCC 점수가 높은) 이웃 타일 하나를 골라 연결.
        동작:
        frontier_tiles(이미 조립된 타일들)의 모든 이웃을 검사함.
        아직 조립되지 않은 이웃 중 NCC 점수가 가장 높은 연결(Edge)을 찾음.
        그 타일을 MST에 추가하고, 기준 타일로부터의 상대 좌표를 더해 절대 좌표(abs_x, abs_y)를 확정함.
        """

        origin_tile = None
        next_tile = None
        best_ncc = -np.inf

        # loop over all tiles currently in the MST and find the neighbor with the highest correlation
        for tile in frontier_tiles:
            for i in range(len(self._dx)):
                r = tile.r + self._dy[i]
                c = tile.c + self._dx[i]
                if r >= 0 and r < self.tile_grid.height and c >= 0 and c < self.tile_grid.width:
                    if not visited_tiles[r, c]:
                        neighbor_tile = self.tile_grid.get_tile(r, c)
                        if neighbor_tile is None:
                            continue
                        edge_weight = tile.get_peak(neighbor_tile).ncc
                        if edge_weight > best_ncc:
                            best_ncc = edge_weight
                            origin_tile = tile
                            next_tile = neighbor_tile

        if origin_tile is None:
            return mst_size
        if next_tile is None:
            return mst_size

        next_tile.update_absolute_position(origin_tile)
        frontier_tiles.add(next_tile)
        mst_size += 1

        # increment MST counter for all adjacent tiles so we can skip those tiles that have no non-connected neighbors (update the frontier)
        for i in range(len(self._dx)):
            r = next_tile.r + self._dy[i]
            c = next_tile.c + self._dx[i]
            if r >= 0 and r < self.tile_grid.height and c >= 0 and c < self.tile_grid.width and self.tile_grid.get_tile(r, c) is not None:
                mst_release_counts[r, c] -= 1

        visited_tiles[next_tile.r, next_tile.c] = True
        # purge visited tiles list of entries that are no longer on the frontier
        to_del = set()
        for tile in frontier_tiles:
            if mst_release_counts[tile.r, tile.c] == 0:
                to_del.add(tile)
        for tile in to_del:
            frontier_tiles.remove(tile)

        return mst_size



    def traverse_minimum_spanning_tree(self):
        """
        Traverses the maximum spanning tree of the grid based on correlation coefficient. Each each step it computes the absolute position relative to the edge taken.
        """
        """
        역할: 전체 스티칭 프로세스를 총괄하는 메인 루프.
        동작:
            전체 그리드에서 NCC 점수가 가장 높은 타일을 시작점(Seed)으로 함.
            모든 타일이 연결될 때까지 traverse_next_mst_tile을 반복 호출.
            마지막으로 모든 타일의 좌표에서 최소값(min_x, min_y)을 빼서, 전체 이미지의 시작점이 (0, 0)이 되도록 보정.
        """

        start_tile = None
        logging.info("Starting MST traversal")
        visited_tiles = np.zeros((self.tile_grid.height, self.tile_grid.width), dtype=bool)
        mst_release_counts = np.zeros((self.tile_grid.height, self.tile_grid.width), dtype=int)

        # Find tile that has highest correlation to use as the starting seed point for the MST
        for r in range(self.tile_grid.height):
            for c in range(self.tile_grid.width):
                mst_release_counts[r, c] = self.get_release_count(r, c)
                tile = self.tile_grid.get_tile(r, c)
                if tile is not None:
                    tile.abs_x = 0
                    tile.abs_y = 0

                    ncc = tile.get_max_translation_ncc()
                    if np.isnan(ncc):
                        continue
                    if start_tile is None:
                        start_tile = tile
                    else:
                        st_ncc = start_tile.get_max_translation_ncc()
                        if not np.isnan(st_ncc) and ncc > st_ncc:
                            start_tile = tile


        frontier_tiles = set()
        frontier_tiles.add(start_tile)

        # increment MST counter for all adjacent tiles so we can skip those tiles that have no non-connected neighbors
        for i in range(len(self._dx)):
            r = start_tile.r + self._dy[i]
            c = start_tile.c + self._dx[i]
            if r >= 0 and r < self.tile_grid.height and c >= 0 and c < self.tile_grid.width:
                mst_release_counts[r, c] -= 1

        # set the flag to indicate that the start tile has been added to the MST
        visited_tiles[start_tile.r, start_tile.c] = True
        mst_size = 1  # current size is 1 b/c startTile has been added

        tgt_mst_size = self.tile_grid.get_num_valid_tiles()
        # # [추가] 무한 루프 방지용 카운터
        # max_iterations = tgt_mst_size * 10 
        # iter_count = 0

        while mst_size < tgt_mst_size:
        #     # [추가] 안전장치: 너무 많이 돌면 강제로 멈춤
        #     iter_count += 1
        #     if iter_count > max_iterations:
        #         logging.warning(f"MST Traversal stuck! Force breaking. (Size: {mst_size}/{tgt_mst_size})")
        #         break
                
            mst_size = self.traverse_next_mst_tile(frontier_tiles, visited_tiles, mst_release_counts, mst_size)
        
        logging.info("Completed MST traversal")

        # Translates all vertices in the grid by the minX and minY values of the entire grid.
        min_x = np.inf
        min_y = np.inf
        for r in range(self.tile_grid.height):
            for c in range(self.tile_grid.width):
                tile = self.tile_grid.get_tile(r, c)
                if tile is not None:
                    min_x = min(min_x, tile.abs_x)
                    min_y = min(min_y, tile.abs_y)

        for r in range(self.tile_grid.height):
            for c in range(self.tile_grid.width):
                tile = self.tile_grid.get_tile(r, c)
                if tile is not None:
                    tile.abs_x -= min_x
                    tile.abs_y -= min_y

