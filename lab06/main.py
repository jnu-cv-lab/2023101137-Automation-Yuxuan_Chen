import cv2
import numpy as np
import os

IMG_BOX = "./box.png"
IMG_SCENE = "./box_in_scene.png"
OUTPUT = "./output"
os.makedirs(OUTPUT, exist_ok=True)

def main():
    # 读取图像
    img1 = cv2.imread(IMG_BOX)
    img2 = cv2.imread(IMG_SCENE)

    if img1 is None or img2 is None:
        print("图片缺失")
        return

    # 任务1：ORB特征检测
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    img1_kp = cv2.drawKeypoints(img1, kp1, None, (0,255,0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, (0,255,0))

    cv2.imwrite(f"{OUTPUT}/task1_box.png", img1_kp)
    cv2.imwrite(f"{OUTPUT}/task1_scene.png", img2_kp)

    print(f"box关键点：{len(kp1)}")
    print(f"scene关键点：{len(kp2)}")
    print(f"描述子维度：{des1.shape[1]}")

    # 任务2：暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    cv2.imwrite(f"{OUTPUT}/task2_matches.png", img_match)
    print(f"总匹配数：{len(matches)}")

    # 任务3：RANSAC提纯
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = sum(mask.ravel())
    ratio = inliers / len(matches)

    img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchesMask=mask.ravel().tolist(), flags=2)
    cv2.imwrite(f"{OUTPUT}/task3_ransac.png", img_ransac)

    print("Homography矩阵：")
    print(H)
    print(f"总匹配：{len(matches)}, 内点：{inliers}, 内点率：{ratio:.2f}")

    # 任务4：目标定位
    h, w = img1.shape[:2]
    pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    pts_proj = cv2.perspectiveTransform(pts, H)

    res = img2.copy()
    res = cv2.polylines(res, [np.int32(pts_proj)], True, (0,0,255), 3)
    cv2.imwrite(f"{OUTPUT}/task4_result.png", res)
    print("目标已定位，结果保存在 output/task4_result.png")

    # 任务6：参数对比实验
    nfeatures_list = [500, 1000, 2000]
    table = []

    for n in nfeatures_list:
        orb = cv2.ORB_create(nfeatures=n)
        kp1_, des1_ = orb.detectAndCompute(img1, None)
        kp2_, des2_ = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        ms = bf.match(des1_, des2_)

        suc = "否"
        in_cnt = 0
        r = 0.0

        if len(ms) >= 4:
            s = np.float32([kp1_[m.queryIdx].pt for m in ms]).reshape(-1,1,2)
            d = np.float32([kp2_[m.trainIdx].pt for m in ms]).reshape(-1,1,2)
            _, msk = cv2.findHomography(s, d, cv2.RANSAC, 5.0)
            in_cnt = sum(msk.ravel())
            r = in_cnt / len(ms)
            if in_cnt > 10:
                suc = "是"

        table.append([n, len(kp1_), len(kp2_), len(ms), in_cnt, r, suc])

    # 输出对比表
    print("\nnfeatures | box_kp | scene_kp | matches | inliers | ratio | success")
    for row in table:
        print(f"{row[0]:<9} {row[1]:<7} {row[2]:<9} {row[3]:<9} {row[4]:<8} {row[5]:<.4f} {row[6]:<5}")

if __name__ == "__main__":
    main()