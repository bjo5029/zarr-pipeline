import argparse

import numpy as np
import scipy.fft
import time
import logging
from abc import ABC

# local imports
import img_tile
import img_grid
import utils


class PCIAM(ABC):

    @staticmethod
    def extract_subregion(t1: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Extracts the sub-region visible if the image view window is translated the given (x,y) distance).
        :param t1: The image tile a sub-region is being extracted from. The translation (x,y) is relative to the upper left corner of this image.
        :param x: the x component of the translation.
        :param y: the y component of the translation.
        :return: the portion of tile shown if the view is translated (x,y) pixels.
        """
        """
        역할: 이미지 t1을 (x, y)만큼 이동시켰을 때, 화면에 보이는 겹치는 부분만 잘라내는(Crop) 함수
        동작:
            이동 후 유효한 좌표 범위(0 ~ width/height)를 계산하고 np.clip으로 자름. 
            만약 이미지가 너무 멀리 이동해서 겹치는 부분이 아예 없으면 None을 반환. 
            NCC(상관관계) 계산을 위해 필수적인 전처리 함수임.
        """
        w = t1.shape[1]
        h = t1.shape[0]

        x_start = x
        x_end = x + w - 1
        y_start = y
        y_end = y + h - 1

        # constrain to valid coordinates
        x_start = np.clip(x_start, 0, w - 1)
        x_end = np.clip(x_end, 0, w - 1)
        y_start = np.clip(y_start, 0, h - 1)
        y_end = np.clip(y_end, 0, h - 1)

        # if the translations (x,y) would leave no overlap between the images, return None
        if abs(x) >= w or abs(y) >= h:
            return None

        sub_tile = t1[y_start:y_end + 1, x_start:x_end + 1]
        return sub_tile

    @staticmethod
    def cross_correlation(a1: np.ndarray, a2: np.ndarray) -> float:
        """
        Computes the cross correlation between two arrays.
        :param a1: the first array
        :param a2: the second array
        :return: the normalized cross correlation between the two arrays.
        """
        """
        역할: 두 이미지 배열 a1과 a2가 얼마나 비슷한지 정규화 상호 상관(NCC) 점수를 계산.
        동작:
            각 배열의 평균값을 뺌(Zero-mean).
            두 배열의 내적(Dot product)을 분자로, 각 배열의 크기(Norm)의 곱을 분모로 하여 나눔.
            결과값: 1.0이면 완벽히 일치, 0이면 관련 없음, -1.0이면 정반대.
        """
        a1 = a1.ravel().astype(np.float32)
        a2 = a2.ravel().astype(np.float32)

        a1 -= np.mean(a1)
        a2 -= np.mean(a2)

        neumerator = np.matmul(a1.transpose(), a2)
        denominator = np.sqrt(np.matmul(a1.transpose(), a1) * np.matmul(a2.transpose(), a2))
        cr = neumerator / denominator
        if not np.isfinite(cr):
            cr = -1

        return cr

    @staticmethod
    def compute_cross_correlation(t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> float:
        """
        Computes the cross correlation between two ImageTiles given the offset (x,y) from the first to the second.
        :param t1: the first tile
        :param t2: the second tile
        :param x: the x component of the translation from i1 to i2.
        :param y: the y component of the translation from i1 to i2.
        :return: the normalized cross correlation between the overlapping pixels given the translation between t1 and t2 (x,y).
        """
        """
        역할: 두 타일 t1, t2를 (x, y)만큼 이동시켰을 때의 일치도(NCC)를 구함.
        동작:
            extract_subregion을 호출하여 t1과 t2의 겹치는 부위만 잘라냄.
            잘라낸 부위들을 cross_correlation에 넣어 점수를 반환.
        """
        a1 = PCIAM.extract_subregion(t1, x, y)
        if a1 is None:
            return -1.0
        a2 = PCIAM.extract_subregion(t2, -x, -y)
        if a2 is None:
            return -1.0
        return PCIAM.cross_correlation(a1, a2)

    @staticmethod
    def peak_cross_correlation_worker(t1: np.ndarray, t2: np.ndarray, dims: list[tuple[int, int]]) -> img_tile.Peak:
        """
        역할: 여러 개의 후보 이동 좌표들(dims) 중에서 가장 점수가 높은 좌표를 선택.
        동작: 입력받은 dims 리스트(x, y 좌표들)를 순회하며 compute_cross_correlation을 실행하고, 가장 높은 NCC 값을 가진 Peak 객체(점수, x, y)를 반환.
        """
        # remove duplicate dim values to prevent redundant computation
        dims = list(set(dims))

        ncc_list = list()
        x_list = list()
        y_list = list()
        for i in range(len(dims)):
            nr = dims[i][0]
            nc = dims[i][1]

            peak = PCIAM.compute_cross_correlation(t1, t2, nc, nr)
            if np.isnan(peak):
                peak = -1
            ncc_list.append(peak)
            x_list.append(nc)
            y_list.append(nr)

        idx = np.argmax(ncc_list)
        peak = img_tile.Peak(ncc_list[idx], x_list[idx], y_list[idx])

        return peak

    @staticmethod
    def peak_cross_correlation_lr(t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images t1 and t2, where t1 is the left image
        """
        """
        역할: FFT 결과는 이동 거리의 크기는 알려주지만 방향(+, -)이나 사분면이 모호할 때가 있음. 이를 해결하기 위해 가능한 8가지 방향의 조합을 모두 검사.
        동작:
            lr: 좌우(Left-Right) 타일 관계일 때 가능한 좌표 조합 8개를 생성.
            생성된 조합을 peak_cross_correlation_worker에 넘겨 진짜 정답을 찾음.
        """

        w = t1.shape[1]
        h = t1.shape[0]

        # a given correlation triple between two images can have multiple interpretations
        # In the general case the translation from t1 to t2 can be any (x,y) so long as the two
        # images overlap. Therefore, given an input (x,y) where x and y are positive by definition
        # of the translation, we need to check 16 possible translations to find the correct
        # interpretation of the translation offset magnitude (x,y). The general case of 16
        # translations arise from the four Fourier transform possibilities, [(x, y); (x, H-y); (W-x,
        # y); (W-x,H-y)] and the four direction possibilities (+-x, +-y) = [(x,y); (x,-y); (-x,y);
        # (-x,-y)].
        # Because we know t1 and t2 form a left right pair, we can limit this search to the 8
        # possible combinations by only considering (x,+-y).
        dims = [(y, x), (y, w - x), (h - y, x), (h - y, w - x),
                ((-y), x), ((-y), w - x), (-(h - y), x), (-(h - y), w - x)]

        return PCIAM.peak_cross_correlation_worker(t1, t2, dims)

    @staticmethod
    def peak_cross_correlation_ud(t1: np.ndarray, t2: np.ndarray, x: int, y: int) -> img_tile.Peak:
        """
        Computes the peak cross correlation between two images t1 and t2, where t1 is the top image
        """
        """
        역할: FFT 결과는 이동 거리의 크기는 알려주지만 방향(+, -)이나 사분면이 모호할 때가 있음. 이를 해결하기 위해 가능한 8가지 방향의 조합을 모두 검사.
        동작:
            ud: 상하(Up-Down) 타일 관계일 때 가능한 좌표 조합 8개를 생성.
            생성된 조합을 peak_cross_correlation_worker에 넘겨 진짜 정답을 찾음.
        """
        w = t1.shape[1]
        h = t1.shape[0]

        # a given correlation triple between two images can have multiple interpretations
        # In the general case the translation from t1 to t2 can be any (x,y) so long as the two
        # images overlap. Therefore, given an input (x,y) where x and y are positive by definition
        # of the translation, we need to check 16 possible translations to find the correct
        # interpretation of the translation offset magnitude (x,y). The general case of 16
        # translations arise from the four Fourier transform possibilities, [(x, y); (x, H-y); (W-x,
        # y); (W-x,H-y)] and the four direction possibilities (+-x, +-y) = [(x,y); (x,-y); (-x,y);
        # (-x,-y)].
        # Because we know t1 and t2 form an up down pair, we can limit this search to the 8
        # possible combinations by only considering (+-x,y).
        dims = [(y, x), (y, w - x), (h - y, x), (h - y, w - x),
                (y, (-x)), (y, -(w - x)), (h - y, (-x)), (h - y, -(w - x))]

        return PCIAM.peak_cross_correlation_worker(t1, t2, dims)

    @staticmethod
    def compute_pciam(t1: img_tile.Tile, t2: img_tile.Tile, n_peaks: int) -> img_tile.Peak:
        """
        역할: 두 타일 t1, t2 사이의 이동 거리를 FFT를 통해 계산.
        동작:
            scipy.fft.fft2: 두 이미지를 주파수 도메인으로 변환.
            위상 상관(Phase Correlation) 계산: 두 주파수를 곱하고 정규화한 뒤, 다시 ifft2로 역변환하여 PCM(위상 상관 행렬)을 얻음. 
            이게 이 행렬에서 값이 가장 높은 곳이 겹치는 위치.
            argpartition: PCM에서 값이 가장 높은 상위 n_peaks개의 지점을 후보로 뽑음.
            후보 지점들에 대해 lr 또는 ud 함수를 호출하여 실제 NCC 점수를 검증하고, 최적의 Peak를 반환함.
        """
        t1_img = t1.get_image()
        t2_img = t2.get_image()

        # use scipy over np to do fft in 32bit (for complex64 result)
        fc = scipy.fft.fft2(t1_img.astype(np.float32)) * np.conj(scipy.fft.fft2(t2_img.astype(np.float32)))
        np.clip(fc.real, a_min=1e-16, a_max=None, out=fc.real)  # specify out for in place clip
        np.clip(fc.imag, a_min=1e-16, a_max=None, out=fc.imag)  # specify out for in place clip
        fc = np.nan_to_num(fc, nan=1e-16, copy=False)  # replace nans with min value, copy=False for in place
        fcn = fc / np.abs(fc)
        pcm = np.real(scipy.fft.ifft2(fcn))

        # get the n_peaks largest values using argpartition to avoid sort
        indices = pcm.argpartition(pcm.size - n_peaks, axis=None)[-n_peaks:]
        # y, x = np.unravel_index(indices, pcm.shape)
        # peak_vals = pcm[y, x]  # if you want the actual peak values

        peak_list = list()

        for ind in indices:
            y, x = np.unravel_index(ind, pcm.shape)
            if t1.r == t2.r:
                # same row, so compute NCC along Left-Right
                peak = PCIAM.peak_cross_correlation_lr(t1_img, t2_img, x, y)
            else:
                # different row, so compute NCC along Up-Down
                peak = PCIAM.peak_cross_correlation_ud(t1_img, t2_img, x, y)
            peak_list.append(peak)

        peak = max(peak_list, key=lambda p: p.ncc)
        return peak


class PciamSequential(PCIAM):
    """
    반복문을 돌며 West 이웃, North 이웃 타일과의 위치를 하나씩 계산.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def execute(self, tile_grid: img_grid.TileGrid):
        start_time = time.time()
        # iterate over the rows and columns of the grid
        for r in range(self.args.grid_height):
            for c in range(self.args.grid_width):
                tile = tile_grid.get_tile(r, c)
                if tile is None:
                    continue

                west = tile_grid.get_tile(r, c - 1)
                if west is not None:
                    peak = self.compute_pciam(west, tile, self.args.num_fft_peaks)
                    tile.west_translation = peak

                north = tile_grid.get_tile(r - 1, c)
                if north is not None:
                    peak = self.compute_pciam(north, tile, self.args.num_fft_peaks)
                    tile.north_translation = peak

        elapsed_time = time.time() - start_time
        logging.info("Finished computing all pairwise translations in {} seconds".format(elapsed_time))


class PciamParallel(PCIAM):
    """
    multiprocessing.Pool을 사용하여 모든 타일 쌍의 계산 작업을 병렬로 처리하여 속도를 높임.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @staticmethod
    def _worker(tile: img_tile.Tile, other: img_tile.Tile, r: int, c: int, direction: str, num_fft_peaks) -> tuple[img_tile.Peak, int, int, str]:
        return PciamParallel.compute_pciam(other, tile, num_fft_peaks), r, c, direction


    def execute(self, tile_grid: img_grid.TileGrid):
        start_time = time.time()
        logging.info("Computing all pairwise translations in parallel")
        logging.info("Preloading all images into memory")
        tile_grid.load_images_into_memory()
        logging.info("Finished preloading all images into memory. Took {}s".format(time.time() - start_time))

        worker_input_list = list()
        # iterate over the rows and columns of the grid
        for r in range(self.args.grid_height):
            for c in range(self.args.grid_width):
                tile = tile_grid.get_tile(r, c)
                if tile is None:
                    continue

                west = tile_grid.get_tile(r, c - 1)
                if west is not None:
                    worker_input_list.append((tile, west, r, c, 'west', self.args.num_fft_peaks))

                north = tile_grid.get_tile(r - 1, c)
                if north is not None:
                    worker_input_list.append((tile, north, r, c, 'north', self.args.num_fft_peaks))

        # results = list()
        # for worker_input in worker_input_list:
        #     results.append(self._worker(*worker_input))
        import multiprocessing
        with multiprocessing.Pool(processes=utils.get_num_workers()) as pool:
            # perform the work in parallel
            results = pool.starmap(self._worker, worker_input_list)

        for result in results:
            peak, r, c, direction = result
            tile = tile_grid.get_tile(r, c)
            if tile is not None:
                if direction == 'west':
                    tile.west_translation = peak
                elif direction == 'north':
                    tile.north_translation = peak

        elapsed_time = time.time() - start_time
        logging.info("Finished computing all pairwise translations in {} seconds".format(elapsed_time))