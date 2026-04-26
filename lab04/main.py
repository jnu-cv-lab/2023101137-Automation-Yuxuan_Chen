import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

def normalize_0_255(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def get_fft_spectrum(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    mag = 20 * np.log(np.abs(fft_shift) + 1)
    return normalize_0_255(mag)

def mse_psnr(im1, im2):
    mse = np.mean((im1 - im2)**2)
    psnr = cv2.PSNR(im1, im2)
    return mse, psnr

def gradient_and_mask(img, thresh=30):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad = normalize_0_255(grad)
    mask = (grad > thresh).astype(np.uint8)
    return grad, mask

def adaptive_down(img, M_detail=2, M_smooth=8, sigma_detail=1.0, sigma_smooth=3.0):
    h, w = img.shape
    grad, mask = gradient_and_mask(img)
    mask_smooth = 1 - mask

    blur_detail = cv2.GaussianBlur(img, (5,5), sigma_detail)
    blur_smooth = cv2.GaussianBlur(img, (5,5), sigma_smooth)
    blur_adp = blur_detail * mask + blur_smooth * mask_smooth

    def down_up(im, M):
        small = cv2.resize(im, (w//M, h//M), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    res_d = down_up(img, M_detail)
    res_s = down_up(img, M_smooth)
    res_adp = res_d * mask + res_s * mask_smooth
    return res_adp, blur_adp, grad, mask

# 实验1：棋盘格 & Chirp 下采样抗混叠 
size = 512

# 棋盘格
checkerboard = np.zeros((size, size), dtype=np.uint8)
block_size = size // 8
for i in range(8):
    for j in range(8):
        if (i + j) % 2 == 0:
            checkerboard[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 255

# Chirp
x, y = np.linspace(-1,1,size), np.linspace(-1,1,size)
X, Y = np.meshgrid(x,y)
r = np.sqrt(X**2 + Y**2)
chirp = np.sin(2 * np.pi * (50*r + 200*r**2))
chirp = normalize_0_255(chirp)

h, w = checkerboard.shape
M = 4
sigma_theory = 0.45 * M

# 直接下采样
checker_direct = cv2.resize(checkerboard, (w//M, h//M), interpolation=cv2.INTER_AREA)
chirp_direct = cv2.resize(chirp, (w//M, h//M), interpolation=cv2.INTER_AREA)

# 高斯滤波 + 下采样
checker_blur = cv2.GaussianBlur(checkerboard, (5,5), sigma_theory)
chirp_blur = cv2.GaussianBlur(chirp, (5,5), sigma_theory)
checker_blur_down = cv2.resize(checker_blur, (w//M, h//M), interpolation=cv2.INTER_AREA)
chirp_blur_down = cv2.resize(chirp_blur, (w//M, h//M), interpolation=cv2.INTER_AREA)

# 频谱
checker_fft = get_fft_spectrum(checkerboard)
checker_direct_fft = get_fft_spectrum(checker_direct)
checker_blur_fft = get_fft_spectrum(checker_blur_down)

chirp_fft = get_fft_spectrum(chirp)
chirp_direct_fft = get_fft_spectrum(chirp_direct)
chirp_blur_fft = get_fft_spectrum(chirp_blur_down)

# 实验1 对比图
plt.figure(figsize=(18,12))
plt.subplot(231), plt.imshow(checkerboard,'gray'), plt.title('checker orig'), plt.axis('off')
plt.subplot(232), plt.imshow(checker_direct,'gray'), plt.title('direct down'), plt.axis('off')
plt.subplot(233), plt.imshow(checker_blur_down,'gray'), plt.title('blur down'), plt.axis('off')
plt.subplot(234), plt.imshow(checker_fft,'gray'), plt.title('orig fft'), plt.axis('off')
plt.subplot(235), plt.imshow(checker_direct_fft,'gray'), plt.title('direct fft'), plt.axis('off')
plt.subplot(236), plt.imshow(checker_blur_fft,'gray'), plt.title('blur fft'), plt.axis('off')
plt.tight_layout()
plt.savefig('exp1_checker_result.png',dpi=300)
plt.close()

plt.figure(figsize=(18,12))
plt.subplot(231), plt.imshow(chirp,'gray'), plt.title('chirp orig'), plt.axis('off')
plt.subplot(232), plt.imshow(chirp_direct,'gray'), plt.title('direct down'), plt.axis('off')
plt.subplot(233), plt.imshow(chirp_blur_down,'gray'), plt.title('blur down'), plt.axis('off')
plt.subplot(234), plt.imshow(chirp_fft,'gray'), plt.title('orig fft'), plt.axis('off')
plt.subplot(235), plt.imshow(chirp_direct_fft,'gray'), plt.title('direct fft'), plt.axis('off')
plt.subplot(236), plt.imshow(chirp_blur_fft,'gray'), plt.title('blur fft'), plt.axis('off')
plt.tight_layout()
plt.savefig('exp1_chirp_result.png',dpi=300)
plt.close()

#不同 σ 对比循环遍历 
sigma_list = [0.5, 1.0, 1.8, 2.0, 4.0]
checker_res = []
chirp_res = []

for sigma in sigma_list:
    cb_blur = cv2.GaussianBlur(checkerboard, (5,5), sigma)
    ch_blur = cv2.GaussianBlur(chirp, (5,5), sigma)
    cb_down = cv2.resize(cb_blur, (w//M, h//M), interpolation=cv2.INTER_AREA)
    ch_down = cv2.resize(ch_blur, (w//M, h//M), interpolation=cv2.INTER_AREA)
    checker_res.append((cb_down, get_fft_spectrum(cb_down)))
    chirp_res.append((ch_down, get_fft_spectrum(ch_down)))

# 棋盘格 σ 对比图
plt.figure(figsize=(22,10))
for i,(sigma,(im,ff)) in enumerate(zip(sigma_list, checker_res)):
    plt.subplot(2,5,i+1), plt.imshow(im,'gray'), plt.title(f'σ={sigma}'), plt.axis('off')
    plt.subplot(2,5,i+6), plt.imshow(ff,'gray'), plt.title(f'σ={sigma} FFT'), plt.axis('off')
plt.tight_layout()
plt.savefig('exp2_checker_sigma_compare.png',dpi=300)
plt.close()

# Chirp σ 对比图
plt.figure(figsize=(22,10))
for i,(sigma,(im,ff)) in enumerate(zip(sigma_list, chirp_res)):
    plt.subplot(2,5,i+1), plt.imshow(im,'gray'), plt.title(f'σ={sigma}'), plt.axis('off')
    plt.subplot(2,5,i+6), plt.imshow(ff,'gray'), plt.title(f'σ={sigma} FFT'), plt.axis('off')
plt.tight_layout()
plt.savefig('exp2_chirp_sigma_compare.png',dpi=300)
plt.close()

# 实验3：自适应下采样
#棋盘格自适应
checker_adp, checker_blur_adp, checker_grad, checker_mask = adaptive_down(checkerboard)
checker_global = cv2.resize(cv2.GaussianBlur(checkerboard,(5,5),sigma_theory), (w//M, h//M), interpolation=cv2.INTER_AREA)
checker_global = cv2.resize(checker_global, (w,h), interpolation=cv2.INTER_LINEAR)

mse_c_adp, psnr_c_adp = mse_psnr(checkerboard, checker_adp)
mse_c_glo, psnr_c_glo = mse_psnr(checkerboard, checker_global)

#Chirp 自适应
chirp_adp, chirp_blur_adp, chirp_grad, chirp_mask = adaptive_down(chirp)
chirp_global = cv2.resize(cv2.GaussianBlur(chirp,(5,5),sigma_theory), (w//M, h//M), interpolation=cv2.INTER_AREA)
chirp_global = cv2.resize(chirp_global, (w,h), interpolation=cv2.INTER_LINEAR)

mse_ch_adp, psnr_ch_adp = mse_psnr(chirp, chirp_adp)
mse_ch_glo, psnr_ch_glo = mse_psnr(chirp, chirp_global)

def plot_exp3(img, adp, glo, grad, mask, name):
    plt.figure(figsize=(20,10))
    plt.subplot(2,4,1), plt.imshow(img,'gray'), plt.title(f'{name} orig'), plt.axis('off')
    plt.subplot(2,4,2), plt.imshow(grad,'gray'), plt.title('gradient'), plt.axis('off')
    plt.subplot(2,4,3), plt.imshow(mask,'gray'), plt.title('mask'), plt.axis('off')
    plt.subplot(2,4,4), plt.imshow(adp,'gray'), plt.title('adaptive down'), plt.axis('off')
    plt.subplot(2,4,5), plt.imshow(glo,'gray'), plt.title('global down'), plt.axis('off')
    plt.subplot(2,4,6), plt.imshow(np.abs(img-adp),'gray'), plt.title('adaptive error'), plt.axis('off')
    plt.subplot(2,4,7), plt.imshow(np.abs(img-glo),'gray'), plt.title('global error'), plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'exp3_{name}_adaptive.png',dpi=300)
    plt.close()

plot_exp3(checkerboard, checker_adp, checker_global, checker_grad, checker_mask, 'checker')
plot_exp3(chirp, chirp_adp, chirp_global, chirp_grad, chirp_mask, 'chirp')

# -------------------- 输出结果 --------------------
print("全部运行完成 结果已保存")
print("实验3 MSE量化对比：")
print(f"棋盘格| 自适应 MSE={mse_c_adp:.1f}   全局 MSE={mse_c_glo:.1f}")
print(f"Chirp| 自适应 MSE={mse_ch_adp:.1f}   全局 MSE={mse_ch_glo:.1f}")