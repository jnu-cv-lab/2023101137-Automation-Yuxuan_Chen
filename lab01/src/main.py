import cv2 as cv        # OpenCV核心库，用于图像处理
import numpy as np      # 数值计算库，OpenCV图像本质是NumPy数组
import sys              # 系统模块，用于程序退出和错误提示
import os

# 屏蔽Qt字体相关警告
os.environ["QT_LOGGING_RULES"] = "*.warning=false;qt.fontdatabase.warning=false"

#读取图片（相对路径：图片放在src/目录下）
# 相对路径：./ 代表当前代码所在的src/目录
img_path = "./ziyue.jpg"
img = cv.imread(img_path)  # 移除cv.samples.findFile（自定义图片无需这个函数）

if img is None:
    sys.exit(f"无法读取图片，请检查文件路径和文件名是否正确。\n当前尝试读取的路径：{os.path.abspath(img_path)}")  # 打印绝对路径方便调试

#输出图像基本信息
height, width = img.shape[:2]  # 取前2维，兼容彩色/灰度图（注意OPENCV中坐标默认是先高度y，再宽度x）
# 判断通道数：如果shape长度是3，说明是彩色图（3通道）；否则是灰度图（1通道）
channels = img.shape[2] if len(img.shape) == 3 else 1
dtype = img.dtype       # img.dtype：图像像素的数据类型

print("图像基本信息:\n")
print(f"图像尺寸：高度(长度) = {height} 像素，宽度 = {width} 像素")
# 修复原代码的小bug：通道数打印重复（原代码会输出"33通道"）
print(f"图像通道数：{channels}通道：{'彩色(BGR)' if channels==3 else '灰度'}")
print(f"图像数据类型：{dtype}")

#显示原图
cv.imshow("Display window", img)    # OPENCV显示图像
#k = cv.waitKey(0)                   # 等待用户输入

#转换为灰度图并显示
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR转灰度图
cv.namedWindow("Gray (OpenCV)", cv.WINDOW_NORMAL)  # 可缩放窗口
cv.imshow("Gray (OpenCV)", img_gray)

#保存灰度图
gray_save_path = "./gray_test.jpg"  # 保存到src/目录
cv.imwrite(gray_save_path, img_gray)
print(f"\n灰度图已保存至：{os.path.abspath(gray_save_path)}")  # 打印绝对路径，方便查看

#NumPy操作：读取指定像素值
print("\nNumPy 操作结果:")
pixel_y, pixel_x = 800, 450  # 获取指定像素值(y, x)
if channels == 3:
    pixel_value = img[pixel_y, pixel_x]  # 彩色图：[B, G, R]
    print(f"坐标({pixel_x}, {pixel_y})的像素值 (B,G,R)：{pixel_value}")
else:
    pixel_value = img[pixel_y, pixel_x]  # 灰度图：单个数字
    print(f"坐标({pixel_x}, {pixel_y})的像素值 (灰度)：{pixel_value}")

#裁剪图像并保存
crop_size = 400  # 裁剪400x400像素的区域
start_y, start_x = 300, 300  # 起始坐标（y=行，x=列）
img_crop = img[start_y:start_y+crop_size, start_x:start_x+crop_size]

# 检查裁剪区域是否有效
if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:  # 完全超出尺寸
    sys.exit("错误：裁剪区域完全超出图像尺寸！")
elif img_crop.shape[0] < crop_size or img_crop.shape[1] < crop_size:  # 部分超出尺寸
    print(f"警告：图像尺寸不足，实际裁剪区域为 {img_crop.shape[0]}x{img_crop.shape[1]}")

# 显示裁剪图
cv.namedWindow("After Crop", cv.WINDOW_NORMAL)
cv.imshow("After Crop", img_crop)

# 保存裁剪图（相对路径）
crop_save_path = "./crop.jpg"  # 保存到src/目录
cv.imwrite(crop_save_path, img_crop)
print(f"从坐标({start_x}, {start_y})开始的{crop_size}x{crop_size}区域已裁剪并保存至：{os.path.abspath(crop_save_path)}")

k = cv.waitKey(0)                   # 等待用户输入
if k == ord("s"):  # 如果用户按下's'键，保存原图
    cv.imwrite("./ziyue_saved.jpg", img)  # 避免覆盖原图片，重命名为ziyue_saved.jpg
#关闭所有窗口
cv.destroyAllWindows()