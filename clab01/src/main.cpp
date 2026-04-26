#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // 1. 读取图片（和Python路径一致）
    string img_path = "./ziyue.jpg";
    Mat img = imread(img_path);

    
    if (img.empty()) {
        cout << "无法读取图片，请检查路径！" << endl;
        cout << "当前路径：" << img_path << endl;
        return -1;
    }

    // 2. 输出图像基本信息
    int height = img.rows;
    int width = img.cols;
    int channels = img.channels();
    string dtype = "8U";  // OpenCV默认类型

    cout << "图像基本信息:" << endl;
    cout << "图像尺寸：高度 = " << height << " 像素，宽度 = " << width << " 像素" << endl;
    cout << "图像通道数：" << channels << "通道：" 
         << (channels == 3 ? "彩色(BGR)" : "灰度") << endl;
    cout << "图像数据类型：" << dtype << endl;

    // 3. 显示原图
    imshow("Display window", img);

    // 4. 转换为灰度图并显示 + 保存
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    namedWindow("Gray (OpenCV)", WINDOW_NORMAL);
    imshow("Gray (OpenCV)", img_gray);

    string gray_save_path = "./gray_test.jpg";
    imwrite(gray_save_path, img_gray);
    cout << "\n灰度图已保存至：" << gray_save_path << endl;

    // 5. 读取指定像素 (y, x)
    cout << "\n像素读取结果:" << endl;
    int pixel_y = 800, pixel_x = 450;
    if (channels == 3) {
        Vec3b pixel = img.at<Vec3b>(pixel_y, pixel_x);
        cout << "坐标(" << pixel_x << ", " << pixel_y << ") 的像素值 (B,G,R)："
             << (int)pixel[0] << " " << (int)pixel[1] << " " << (int)pixel[2] << endl;
    } else {
        uchar pixel = img_gray.at<uchar>(pixel_y, pixel_x);
        cout << "坐标(" << pixel_x << ", " << pixel_y << ") 的像素值(灰度)：" << (int)pixel << endl;
    }

    // 6. 裁剪图像
    int crop_size = 400;
    int start_y = 300, start_x = 300;

    // 检查边界
    if (start_x + crop_size > width || start_y + crop_size > height) {
        cout << "警告：裁剪超出范围，自动调整" << endl;
        crop_size = min(width - start_x, height - start_y);
    }

    Mat img_crop = img(Rect(start_x, start_y, crop_size, crop_size));
    namedWindow("After Crop", WINDOW_NORMAL);
    imshow("After Crop", img_crop);

    string crop_save_path = "./crop.jpg";
    imwrite(crop_save_path, img_crop);
    cout << "\n裁剪图已保存至：" << crop_save_path << endl;

    // 7. 等待按键
    int k = waitKey(0);
    if (k == 's') {
        imwrite("./ziyue_saved.jpg", img);
    }

    // 关闭窗口
    destroyAllWindows();
    return 0;
}