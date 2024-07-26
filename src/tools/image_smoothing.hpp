#ifndef IMAGE_SMOOTHING_HPP
#define IMAGE_SMOOTHING_HPP
#include <iostream>
#include <opencv4/opencv2/imgproc.hpp>
#include "motion_blur.hpp"
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

inline int DELAY_CAPTION =  1500;
inline int DELAY_BLUR = 100;
inline int MAX_KERNEL_LENGTH = 31;

class ImageSmoothing {
    public:
    static void smooth(const cv::Mat &src) {
        const Mat clone = src.clone();
        MotionBlur motionBlur;

        Mat dst = motionBlur.motionBlur(clone);
        dst = Mat::zeros(src.size(), src.type());

        if (!src.data) {
            cout << "Error loading src" << endl;
        }

        // Guassian blur: Save to dst
        for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 ) {
            GaussianBlur( src, dst, cv::Size( i, i ), 0, 0 );
        }
    }
};
#endif // IMAGE_SMOOTHING_HPP
