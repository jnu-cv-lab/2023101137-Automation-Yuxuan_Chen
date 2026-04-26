import cv2
import numpy as np

image_path = "ziyue.jpg"
original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
height, width = original.shape

# 先做高斯模糊，抑制混叠
blurred = cv2.GaussianBlur(original, (5, 5), 0)

# 直接下采样 1/2
direct_down = cv2.resize(original, (width//2, height//2), interpolation=cv2.INTER_AREA)
# 先模糊再下采样
blur_down = cv2.resize(blurred, (width//2, height//2), interpolation=cv2.INTER_AREA)

nearest = cv2.resize(direct_down, (width, height), interpolation=cv2.INTER_NEAREST)
bilinear = cv2.resize(direct_down, (width, height), interpolation=cv2.INTER_LINEAR)
bicubic = cv2.resize(direct_down, (width, height), interpolation=cv2.INTER_CUBIC)

def mse_psnr(original_img, restored_img):
    mse_val = np.mean((original_img - restored_img) ** 2)
    psnr_val = cv2.PSNR(original_img, restored_img)
    return mse_val, psnr_val

mse_n, psnr_n = mse_psnr(original, nearest)
mse_b, psnr_b = mse_psnr(original, bilinear)
mse_c, psnr_c = mse_psnr(original, bicubic)

print("===== 图像质量评价 =====")
print(f"最近邻插值    MSE={mse_n:.4f}    PSNR={psnr_n:.4f} dB")
print(f"双线性插值    MSE={mse_b:.4f}    PSNR={psnr_b:.4f} dB")
print(f"双三次插值    MSE={mse_c:.4f}    PSNR={psnr_c:.4f} dB")

def dct_process(img):
    dct_mat = cv2.dct(np.float32(img))
    log_dct = 20 * np.log(np.abs(dct_mat) + 1e-6)
    return dct_mat, log_dct

dct_orig, log_orig = dct_process(original)
dct_bili, log_bili = dct_process(bilinear)
dct_near, log_near = dct_process(nearest)
dct_bicu, log_bicu = dct_process(bicubic)

def low_freq_ratio(dct_mat, h, w):
    total = np.sum(dct_mat ** 2)
    low = np.sum(dct_mat[:h//4, :w//4] ** 2)
    return low / total

r1 = low_freq_ratio(dct_orig, height, width)
r2 = low_freq_ratio(dct_near, height, width)
r3 = low_freq_ratio(dct_bili, height, width)
r4 = low_freq_ratio(dct_bicu, height, width)

print("\n===== DCT 低频能量占比 =====")
print(f"原图：\t{r1:.4f}")
print(f"最近邻：\t{r2:.4f}")
print(f"双线性：\t{r3:.4f}")
print(f"双三次：\t{r4:.4f}")

# ===================== 傅里叶频谱 =====================
def spectrum(img):
    fft = np.fft.fft2(img)
    shift = np.fft.fftshift(fft)
    mag = 20 * np.log(np.abs(shift) + 1e-6)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

fft_original = spectrum(original)
fft_down = spectrum(direct_down)
fft_restore = spectrum(bilinear)

# ===================== 显示与保存 =====================
def show(title, img):
    cv2.imshow(title, img)
    cv2.imwrite(f"{title}.png", img)

show("original", original)
show("direct_down", direct_down)
show("blur_down", blur_down)
show("nearest", nearest)
show("bilinear", bilinear)
show("bicubic", bicubic)

show("fft_original", fft_original)
show("fft_down", fft_down)
show("fft_restore", fft_restore)

show("dct_original", log_orig)
show("dct_bilinear", log_bili)

cv2.waitKey(0)
cv2.destroyAllWindows()