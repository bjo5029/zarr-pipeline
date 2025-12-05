import numpy as np
import torch
import torch.fft
import torch.nn.functional as F

def get_psf_numpy(shape=(31, 31), sigma=2.0):
    """PSF 생성 (기존 동일)"""
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    if h.sum() != 0: h /= h.sum()
    return h

def richardson_lucy_torch(image, psf, num_iter=20):
    """
    image shape: (Batch, H, W) or (Z, H, W) or (H, W)
    Torch FFT는 마지막 두 차원(-2, -1)을 기준으로 연산하므로
    앞쪽 차원(Batch/Z)은 자동으로 병렬 처리됨.
    """
    with torch.no_grad():
        if psf.sum() != 0: psf /= psf.sum()
        psf_mirror = torch.flip(psf, dims=[-2, -1])
        im_deconv = image.clone()
        epsilon = 1e-8
        
        # FFT shape 계산
        s1 = torch.tensor(image.shape[-2:])
        s2 = torch.tensor(psf.shape[-2:])
        shape = tuple((s1 + s2 - 1).tolist())
        
        # PSF FFT (Broadcasting을 위해 차원 맞춤 필요 없음, 자동 적용)
        psf_f = torch.fft.rfftn(psf, s=shape, dim=(-2, -1))
        psf_mirror_f = torch.fft.rfftn(psf_mirror, s=shape, dim=(-2, -1))
        
        start_idx = (s2 - 1) // 2
        end_idx = start_idx + s1

        for _ in range(num_iter):
            im_f = torch.fft.rfftn(im_deconv, s=shape, dim=(-2, -1))
            conv = torch.fft.irfftn(im_f * psf_f, s=shape, dim=(-2, -1))
            
            # Crop
            conv = conv[..., start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]]
            
            relative_blur = image / (conv + epsilon)
            
            blur_f = torch.fft.rfftn(relative_blur, s=shape, dim=(-2, -1))
            error_est = torch.fft.irfftn(blur_f * psf_mirror_f, s=shape, dim=(-2, -1))
            
            # Crop
            error_est = error_est[..., start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]]
            
            im_deconv *= error_est
            
        return torch.clamp(im_deconv, 0, 1)

def deconv_wrapper_torch(block, psf_cpu, iterations, global_min, global_max, pad_width):
    try:
        # 1. 빈 블록 체크
        if block.size == 0:
            return block

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. 데이터 타입 및 차원 정리
        # Dask에서 넘어오는 block은 (Z, Y, X) 또는 (1, 1, Z, Y, X)일 수 있음
        # PyTorch 처리를 위해 마지막 3차원 (Z, Y, X) 또는 2차원 (Y, X)로 맞춤
        arr = block.astype(np.float32)
        
        # 만약 (1, 1, Z, Y, X) 등으로 넘어오면 squeeze
        while arr.ndim > 3:
            arr = arr.squeeze(0)
        # 이제 arr는 (Z, Y, X) 또는 (Y, X)라고 가정

        # 3. GPU 로드
        gpu_img = torch.tensor(arr, device=device)
        gpu_psf = torch.tensor(psf_cpu, device=device)

        # 4. Normalize
        if global_max > global_min:
            gpu_img = (gpu_img - global_min) / (global_max - global_min)
        gpu_img = torch.clamp(gpu_img, 0, 1)

        # 5. Pad
        # pad 인자는 (Left, Right, Top, Bottom) 순서 (마지막 차원부터)
        # 3D 입력 (Z, Y, X)인 경우 Y, X만 패딩하면 됨
        gpu_img_padded = F.pad(gpu_img, (pad_width, pad_width, pad_width, pad_width), mode='reflect')

        # 6. Deconv Execution
        # (Z, H, W) 형태여도 함수 내부에서 H, W 기준 FFT 수행하므로 Z는 Batch처럼 동작
        deconv_padded = richardson_lucy_torch(gpu_img_padded, gpu_psf, num_iter=iterations)
        
        # 7. Unpad
        # 마지막 두 차원(Y, X)에 대해서만 슬라이싱
        deconv_result = deconv_padded[..., pad_width:-pad_width, pad_width:-pad_width]

        # 8. 원래 shape으로 복구하여 반환
        return deconv_result.cpu().numpy().reshape(block.shape).astype(np.float32)

    except Exception as e:
        print(f"Worker Error with block shape {block.shape}: {e}")
        # 에러 시 원본 반환 (파이프라인 중단 방지)
        return block
    