import dask.array as da
import numpy as np

def generate_pyramid_weight(shape, alpha=1.5):
    """
    Java MIST의 LinearBlend.java 로직(Distance-Weighted)을 Python(Dask)으로 구현.
    이미지의 기하학적 중심일수록 가중치가 높고, 가장자리로 갈수록 0이 되는 '피라미드 마스크'를 생성함.
    
    Args:
        shape (tuple): 타일의 (Height, Width)
        alpha (float): 가중치 감소율 (Java 기본값 1.5). 클수록 중앙 집중도가 강해짐.
        
    Returns:
        dask.array: (H, W) 크기의 가중치 마스크
    """
    H, W = shape
    
    # 1. 좌표 그리드 생성 (Dask의 arange 사용)
    # y: (H, 1), x: (1, W) 형태로 만들어서 브로드캐스팅 준비
    y = da.arange(H, chunks=H).reshape(H, 1)
    x = da.arange(W, chunks=W).reshape(1, W)
    
    # 2. 4면(상하좌우) 끝에서의 거리 계산 (Java 코드와 동일)
    # 좌표는 0부터 시작하므로 +1을 해줍니다.
    dist_north = y + 1.0
    dist_south = H - y
    dist_west = x + 1.0
    dist_east = W - x
    
    # 3. 최소 거리 선택 (상하 중 작은 것, 좌우 중 작은 것)
    # 가장자리에 가까운 거리를 선택합니다.
    min_ns = da.minimum(dist_north, dist_south)
    min_ew = da.minimum(dist_west, dist_east)
    
    # 4. 가중치 계산 (두 거리를 곱하고 alpha승)
    # 상하 거리와 좌우 거리를 곱하면 피라미드 형태가 됩니다.
    weight = min_ns * min_ew
    weight = weight ** alpha
    
    # 5. 분모가 0이 되는 것을 방지하기 위해 아주 작은 값 더함
    return weight + 1e-6

def blend_overlap(arr_left, arr_right, overlap_size, axis=4):
    """
    두 배열(arr_left, arr_right)의 겹치는 구간을 Pyramid Weight로 블렌딩하여 연결함.
    
    Args:
        arr_left: 앞쪽(왼쪽 또는 위쪽) 타일 Dask Array
        arr_right: 뒤쪽(오른쪽 또는 아래쪽) 타일 Dask Array
        overlap_size: 겹치는 픽셀 수 (int)
        axis: 연결할 축 (4=가로/X축, 3=세로/Y축) - 5D (T, C, Z, Y, X) 기준
    """
    # 겹침이 없거나 비정상적인 경우 단순히 연결
    if overlap_size <= 0:
        return da.concatenate([arr_left, arr_right], axis=axis)
        
    # 타일의 크기 (H, W) 가져오기
    # arr_left.shape는 (T, C, Z, Y, X)이므로 뒤에서 2개
    shape = arr_left.shape[-2:] 
    
    # 전체 타일 크기에 맞는 가중치 마스크 생성 (Lazy Evaluation)
    full_mask = generate_pyramid_weight(shape)
    
    # === 가로 연결 (Horizontal Stitching, Axis=4) ===
    if axis == 4: 
        # 1. 데이터 슬라이싱
        left_pure = arr_left[..., :-overlap_size]       # 겹치지 않는 왼쪽 부분
        left_ov = arr_left[..., -overlap_size:]         # 왼쪽 타일의 겹치는 부분 (오른쪽 끝)
        
        right_pure = arr_right[..., overlap_size:]      # 겹치지 않는 오른쪽 부분
        right_ov = arr_right[..., :overlap_size]        # 오른쪽 타일의 겹치는 부분 (왼쪽 끝)
        
        # 2. 마스크 슬라이싱
        # 왼쪽 타일의 마스크 중 오른쪽 끝부분
        w_left = full_mask[:, -overlap_size:] 
        # 오른쪽 타일의 마스크 중 왼쪽 끝부분
        w_right = full_mask[:, :overlap_size] 
        
    # === 세로 연결 (Vertical Stitching, Axis=3) ===
    else: 
        # 1. 데이터 슬라이싱
        left_pure = arr_left[..., :-overlap_size, :]    # 겹치지 않는 위쪽 부분
        left_ov = arr_left[..., -overlap_size:, :]      # 위쪽 타일의 겹치는 부분 (아래쪽 끝)
        
        right_pure = arr_right[..., overlap_size:, :]   # 겹치지 않는 아래쪽 부분
        right_ov = arr_right[..., :overlap_size, :]     # 아래쪽 타일의 겹치는 부분 (위쪽 끝)
        
        # 2. 마스크 슬라이싱
        # 위쪽 타일 마스크의 아래쪽 끝
        w_left = full_mask[-overlap_size:, :] 
        # 아래쪽 타일 마스크의 위쪽 끝
        w_right = full_mask[:overlap_size, :]

    # === 블렌딩 연산 (Weighted Average) ===
    
    # 차원 맞추기 (Broadcasting)
    # 이미지는 5D(T,C,Z,Y,X)인데 마스크는 2D(Y,X)이므로 앞의 차원을 확장
    while w_left.ndim < arr_left.ndim:
        w_left = w_left[None, ...]
        w_right = w_right[None, ...]

    # 공식: (Pixel1 * Weight1 + Pixel2 * Weight2) / (Weight1 + Weight2)
    numerator = (left_ov * w_left) + (right_ov * w_right)
    denominator = w_left + w_right
    blended_zone = numerator / denominator
    
    # 원본 데이터 타입 유지 (float 계산 후 변환)
    blended_zone = blended_zone.astype(arr_left.dtype)
    
    # 최종 연결: [순수 영역] + [블렌딩된 영역] + [순수 영역]
    return da.concatenate([left_pure, blended_zone, right_pure], axis=axis)