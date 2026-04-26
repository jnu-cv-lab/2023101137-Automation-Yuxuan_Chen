import matplotlib.pyplot as plt
import cv2 as cv        # OpenCV核心库，用于图像处理
import numpy as np      # 数值计算库，OpenCV图像本质是NumPy数组
import sys              # 系统模块，用于程序退出和错误提示
import math             # 数学模块，用于角度转换和三角函数计算
import os               # 操作系统模块，用于环境变量设置和路径处理

# 屏蔽Qt字体相关警告
os.environ["QT_LOGGING_RULES"] = "*.warning=false;qt.fontdatabase.warning=false"
# mouse callback函数，记录鼠标点击的坐标
def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv.circle(img_show, (x, y), 5, (0, 255, 0), -1)
        cv.imshow("click 4 corners", img_show)
#读取图片（相对路径：图片放在src/目录下）
# 相对路径：./ 代表当前代码所在的src/目录
img_path = "./test.png"
img_path1 = "./jiaozheng.jpg"
img = cv.imread(img_path)  # 移除cv.samples.findFile（自定义图片无需这个函数）
img1 = cv.imread(img_path1)  # 读取校正图片

if img is None:
    sys.exit(f"无法读取test图片，请检查文件路径和文件名是否正确。\n当前尝试读取的路径：{os.path.abspath(img_path)}")  # 打印绝对路径方便调试
h, w = img.shape[:2]
if img1 is None:
    sys.exit(f"无法读取矫正图片，请检查文件路径和文件名是否正确。\n当前尝试读取的路径：{os.path.abspath(img_path1)}")  # 打印绝对路径方便调试
h1, w1 = img1.shape[:2]

# 相似变换参数设置
theta = 60  # 旋转 60°
s = 0.7     # 缩小到 70%
tx = 300     # 向右平移 100 像素
ty = 30     # 向下平移 30 像素
theta_rad = math.radians(theta) # 角度转换为弧度
cos_theta = math.cos(theta_rad)
sin_theta = math.sin(theta_rad)
# 构造相似变换矩阵
M = np.array([
    [s * cos_theta, -s * sin_theta, tx],
    [s * sin_theta,  s * cos_theta, ty]
])
dst_similarity = cv.warpAffine(img, M, (w, h))
# 仿射变换参数设置
M_affine = np.array([
    [0.8,  0.2,  30],
    [0.1,  0.9,  40]
])

dst_affine = cv.warpAffine(img, M_affine, (w, h))
# 透视变换参数设置
M_perspective = np.array([
    [0.8,  0.1,  20],
    [0.2,  0.9,  30],
    [0.001, 0.002, 1]
])
dst_perspective = cv.warpPerspective(img, M_perspective, (w, h))
cv.imwrite("test_origin.jpg", img)
cv.imwrite("similarity.jpg", dst_similarity)
cv.imwrite("affine.jpg", dst_affine)
cv.imwrite("perspective.jpg", dst_perspective)

cv.imshow("origin", img)
cv.imshow("similarity", dst_similarity)
cv.imshow("Affine", dst_affine)
cv.imshow("Perspective", dst_perspective)
cv.waitKey(0)

##矫正透视畸变
h, w = img1.shape[:2]
scale = 800 / max(h, w)
img_show = cv.resize(img1, None, fx=scale, fy=scale)
points = []

cv.imshow("click 4 corners", img_show)
cv.setMouseCallback("click 4 corners", mouse_click)
cv.waitKey(0)
cv.destroyAllWindows()

if len(points) != 4:
    print("必须点 4 个点！")
    exit()

# 还原到原图坐标
pts1 = np.float32([(x/scale, y/scale) for x, y in points])

w_a = 600
h_a = int(w_a * 1.414)
pts2 = np.float32([[0,0],[w_a,0],[w_a,h_a],[0,h_a]])

# 构造透视矩阵 + 矫正
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img1, M, (w_a, h_a))

# 保存矫正结果
cv.imwrite("jiaozheng_origin.jpg", img1)
cv.imwrite("corrected.jpg", dst)
# 显示结果
img1_small = cv.resize(img1, None, fx=0.25, fy=0.25)
cv.imshow("Original", img1_small)
cv.imshow("Corrected", dst)
cv.waitKey(0)

cv.destroyAllWindows()