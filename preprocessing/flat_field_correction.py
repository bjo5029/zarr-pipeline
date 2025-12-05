import numpy as np
import dask.array as da

def generate_dummy_references(shape_yx, dtype=np.uint16):
    """
    속도 측정을 위한 더미 Flat/Dark 이미지 생성 (Dask Array)
    """
    H, W = shape_yx
    
    # [수정] 랜덤 상태(State) 고정 -> 언제 실행해도 똑같은 노이즈 생성
    rs = da.random.RandomState(42)
    
    # 1. Dark Image: 낮은 값의 노이즈
    # chunks는 통짜로 하나로 잡거나, 타일 크기에 맞추는게 좋음
    dark = rs.randint(50, 60, size=(H, W), chunks=(H, W)).astype(np.float32)
    
    # 2. Flat Image: 비네팅 효과 + 노이즈
    y = da.linspace(-1, 1, H)
    x = da.linspace(-1, 1, W)
    yy, xx = da.meshgrid(y, x, indexing='ij')
    dist = da.sqrt(yy**2 + xx**2)
    
    # 거리에 따라 어두워짐
    vignette = 40000 * (1 - 0.2 * dist)
    
    # [수정] 고정된 시드(rs) 사용
    flat_noise = rs.normal(0, 100, size=(H, W), chunks=(H, W))
    
    flat = vignette + flat_noise
    flat = flat.astype(np.float32)
    
    return flat, dark

def apply_flat_field(img_stack, flat, dark):
    """
    Flat Field Correction 수식 적용
    Formula: (Image - Dark) / (Flat - Dark) * Mean(Flat)
    """
    epsilon = 1e-6
    
    # Flat 평균 계산
    flat_mean = da.mean(flat)
    
    # 분모: Flat - Dark
    denominator = flat - dark
    denominator = da.where(denominator < epsilon, epsilon, denominator)
    
    # 분자: Image - Dark (img_stack은 5D, dark는 2D -> 자동 브로드캐스팅)
    numerator = img_stack.astype(np.float32) - dark
    
    # 보정 계산
    corrected = (numerator / denominator) * flat_mean
    
    # 음수 방지
    corrected = da.maximum(corrected, 0)
    
    return corrected.astype(np.float32)
