import numpy as np
import torch
import torch.fft
import torch.nn.functional as F


def get_psf_numpy(shape=(31, 31), sigma=2.0):
    """PSF 생성"""
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    if h.sum() != 0: h /= h.sum()
    return h

def richardson_lucy_torch(image, psf, num_iter=20):
    with torch.no_grad():
        if psf.sum() != 0: psf /= psf.sum()
        psf_mirror = torch.flip(psf, dims=[-2, -1])
        im_deconv = image.clone()
        epsilon = 1e-8
        
        s1 = torch.tensor(image.shape[-2:])
        s2 = torch.tensor(psf.shape[-2:])
        shape = tuple((s1 + s2 - 1).tolist())
        
        psf_f = torch.fft.rfftn(psf, s=shape, dim=(-2, -1))
        psf_mirror_f = torch.fft.rfftn(psf_mirror, s=shape, dim=(-2, -1))
        start_idx = (s2 - 1) // 2
        end_idx = start_idx + s1

        for _ in range(num_iter):
            im_f = torch.fft.rfftn(im_deconv, s=shape, dim=(-2, -1))
            conv = torch.fft.irfftn(im_f * psf_f, s=shape, dim=(-2, -1))
            conv = conv[..., start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]]
            
            relative_blur = image / (conv + epsilon)
            
            blur_f = torch.fft.rfftn(relative_blur, s=shape, dim=(-2, -1))
            error_est = torch.fft.irfftn(blur_f * psf_mirror_f, s=shape, dim=(-2, -1))
            error_est = error_est[..., start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]]
            
            im_deconv *= error_est
        return torch.clamp(im_deconv, 0, 1)

def deconv_wrapper_torch(block, psf_cpu, iterations, global_min, global_max, pad_width):
    try:
        # [수정 1] 빈 블록(크기가 0)이 들어오면 즉시 반환 (Dask 내부 로직 방어)
        if block.size == 0:
            return block

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 차원 안전하게 확보하기
        img_2d = block.reshape(-1, block.shape[-2], block.shape[-1])[0]

        # Byte Order 안전하게 변환
        if img_2d.dtype.byteorder != '=':
            img_2d = img_2d.astype(np.float32).copy()
        else:
            img_2d = img_2d.astype(np.float32)

        # Numpy -> Tensor
        gpu_img = torch.tensor(img_2d, device=device, dtype=torch.float32)
        gpu_psf = torch.tensor(psf_cpu, device=device, dtype=torch.float32)

        # Normalize
        if global_max > global_min:
            gpu_img = (gpu_img - global_min) / (global_max - global_min)
        gpu_img = torch.clamp(gpu_img, 0, 1)

        # Pad
        gpu_img_4d = gpu_img.unsqueeze(0).unsqueeze(0)
        gpu_img_padded_4d = F.pad(gpu_img_4d, (pad_width,)*4, mode='reflect')
        gpu_img_padded = gpu_img_padded_4d.squeeze(0).squeeze(0)

        # Deconv
        deconv_padded = richardson_lucy_torch(gpu_img_padded, gpu_psf, num_iter=iterations)
        
        # Unpad
        deconv_result = deconv_padded[pad_width:-pad_width, pad_width:-pad_width]

        return deconv_result.cpu().numpy().reshape(block.shape).astype(np.float32)

    except Exception as e:
        # 에러 발생 시 블록 모양 출력
        print(f"Worker Error with block shape {block.shape}: {e}")
        raise e