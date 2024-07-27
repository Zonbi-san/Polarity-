#ifndef opencv_hpp
#define opencv_hpp 

#include <cstdio>
#include <iostream>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace cv {
    const Scalar BLACK    (  0,   0,   0);
    const Scalar BLUE     (255,   0,   0);
    const Scalar GREEN    (  0, 255,   0);
    const Scalar RED      (  0,   0, 255);
    const Scalar WHITE    (255, 255, 255);
    const Scalar ZERO     (0);
    const Scalar ONE      (1);


    // Optimised Functions
    inline void sin(Mat &src, Mat &dst) {
        CV_Assert(src.type() == CV_64F && dst.type() == CV_64F);

        for (int i = 0; i < src.rows; i++) {
            dst.at<double>(i, 0) = sinf(src.at<double>(i, 0));
        }
    };
    
    inline void cos(Mat &src, Mat &dst) {
        CV_Assert(src.type() == CV_64F && dst.type() == CV_64F);

        for (int i = 0; i < src.rows; i++) {
            dst.at<double>(i, 0) = std::cos(src.at<double>(i, 0));
        }
    };

    inline void dft(const Mat &src, const Mat &dst, int flags = 0) {
        CV_Assert(src.type() == CV_64F && dst.type() == CV_64F);

        auto tmp = Mat(src.rows, 1, CV_64F);
        auto tmp2 = Mat(src.rows, 1, CV_64F);
        auto tmp3 = Mat(src.rows, 1, CV_64F);
        auto tmp4 = Mat(src.rows, 1, CV_64F);

        for (int i = 0; i < src.cols; i++) {
            src.col(i).copyTo(tmp);
            sin(tmp, tmp2);
            cos(tmp, tmp3);
            multiply(tmp2, tmp3, tmp4);
            tmp4.copyTo(dst.col(i));
        }
    };

    // Common Functions
    inline double getFps(cv::Mat &t, const double timeBase) {
        double result;

        if (t.empty()) {
            result = 1.0;
        } else if (t.rows == 1) {
            result = std::numeric_limits<double>::max();
        } else {
            double diff = (t.at<int>(t.rows-1, 0) - t.at<int>(0, 0)) * timeBase;
            result = diff == 0 ? std::numeric_limits<double>::max() : t.rows/diff;
        }

        return result;
    };

    inline void push(cv::Mat &m) {
        const int length = m.rows;
        m.rowRange(1, length).copyTo(m.rowRange(0, length - 1));
        m.pop_back();
    };

    inline void plot(const cv::Mat &mat) {
        while (true) {
            cv::imshow("plot", mat);
            if (waitKey(30) >= 0) break;
        }
    };

    // Filters
    /**
     * @defgroup Filters normalization
     * @brief Subtract mean and divide by standard deviation
     * @param _a cv::InputArray
     * @param _b cv::OutputArray
     * @note This function is used to normalize the data
    */
    inline void normalization(cv::InputArray _a, cv::OutputArray _b) {
        _a.getMat().copyTo(_b);
        const Mat b = _b.getMat();
        Scalar mean, stdDev;
        for (int i = 0; i < b.cols; i++) {
            meanStdDev(b.col(i), mean, stdDev);
            b.col(i) = (b.col(i) - mean[0]) / stdDev[0];
        }
    };

    /**
     * @defgroup Filters denoise
     * @brief Eliminate jumps
     * @param _a cv::InputArray
     * @param _jumps cv::InputArray
     * @param _b cv::OutputArray
     * @note This function is used to eliminate jumps
    */
    inline void denoise(cv::InputArray _a, cv::InputArray _jumps, cv::OutputArray _b) {
        const Mat a = _a.getMat().clone();
        Mat jumps = _jumps.getMat().clone();

        CV_Assert(a.type() == CV_64F && jumps.type() == CV_8U);

        if (jumps.rows != a.rows) {
            jumps.rowRange(jumps.rows-a.rows, jumps.rows).copyTo(jumps);
        }

        Mat diff;
        subtract(a.rowRange(1, a.rows), a.rowRange(0, a.rows-1), diff);

        for (int i = 1; i < jumps.rows; i++) {
            if (jumps.at<bool>(i, 0)) {
                Mat mask = Mat::zeros(a.size(), CV_8U);
                mask.rowRange(i, mask.rows).setTo(ONE);
                for (int j = 0; j < a.cols; j++) {
                    add(a.col(j), -diff.at<double>(i-1, j), a.col(j), mask.col(j));
                }
            }
        }

        a.copyTo(_b);
    };

    /**
     * @defgroup Filters detrend
     * @brief Remove linear trend
     * @param _a cv::InputArray
     * @param _b cv::OutputArray
     * @param lambda int
     * @note Advanced detrending filter based on smoothness priors approach (High pass equivalent)
    */
    inline void detrend(cv::InputArray _a, cv::OutputArray _b, int lambda) {
        Mat a = _a.getMat();
        CV_Assert(a.type() == CV_64F);

        // Number of rows
        int rows = a.rows;

        if (rows < 3) {
            a.copyTo(_b);
        } else {
            // Construct I
            Mat i = Mat::eye(rows, rows, a.type());
            // Construct D2
            auto d = Mat(Matx<double,1,3>(1, -2, 1));
            Mat d2Aux = Mat::ones(rows-2, 1, a.type()) * d;
            Mat d2 = Mat::zeros(rows-2, rows, a.type());
            for (int k = 0; k < 3; k++) {
                d2Aux.col(k).copyTo(d2.diag(k));
            }
            // Calculate b = (I - (I + λ^2 * D2^t*D2)^-1) * a
            Mat b = (i - (i + lambda * lambda * d2.t() * d2).inv()) * a;
            b.copyTo(_b);
        }
    };

    /**
     * @defgroup Filters movingAverage
     * @brief Moving average filter
     * @param _a cv::InputArray
     * @param _b cv::OutputArray
     * @param n int
     * @param s int
     * @note This function is used to smooth the data (low pass equivalent)
    */
    inline void movingAverage(cv::InputArray _a, cv::OutputArray _b, int n, int s) {
        CV_Assert(s > 0);

        _a.getMat().copyTo(_b);
        Mat b = _b.getMat();
        for (size_t i = 0; i < n; i++) {
            cv::blur(b, b, Size(s, s));
        }
    };

    void frequencyToTime(const Mat & mat, _OutputArray b);

    inline void timeToFrequency(InputArray _a, OutputArray _b, const bool magnitude) {
        // Prepare planes
        const Mat a = _a.getMat();
        Mat planes[] = {cv::Mat_<float>(a), cv::Mat::zeros(a.size(), CV_32F)};
        Mat powerSpectrum;
        merge(planes, 2, powerSpectrum);

        // Fourier transform
        dft(powerSpectrum, powerSpectrum, DFT_COMPLEX_OUTPUT);

        if (magnitude) {
            split(powerSpectrum, planes);
            cv::magnitude(planes[0], planes[1], planes[0]);
            planes[0].copyTo(_b);
        } else {
            powerSpectrum.copyTo(_b);
        }
    };

    inline void butterworth_lowpass_filter(const Mat & filter, const double cutoff, const int n) {
        CV_DbgAssert(cutoff > 0 && n > 0 && filter.rows % 2 == 0 && filter.cols % 2 == 0);

        auto tmp = Mat(filter.rows, filter.cols, CV_32F);
        //Point centre = Point(filter.rows / 2, filter.cols / 2);

        for (int i = 0; i < filter.rows; i++) {
            for (int j = 0; j < filter.cols; j++) {
                const double radius = i;
                //radius = (double)sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
                tmp.at<float>(i, j) = static_cast<float>(1 / (1 + pow(radius / cutoff, 2 * n)));
            }
        }

        const Mat toMerge[] = {tmp, tmp};
        merge(toMerge, 2, filter);
    };

    inline void butterworth_bandpass_filter(Mat & filter, const double cutin, const double cutoff, int n) {
        CV_DbgAssert(cutoff > 0 && cutin < cutoff && n > 0 &&
                     filter.rows % 2 == 0 && filter.cols % 2 == 0);
        const Mat off = filter.clone();
        butterworth_lowpass_filter(off, cutoff, n);
        const Mat in = filter.clone();
        butterworth_lowpass_filter(in, cutin, n);
        filter = off - in;
    };

    /**
     * @defgroup Filters bandpass
     * @brief Bandpass filter
     * @param _a cv::InputArray
     * @param _b cv::OutputArray
     * @param low double
     * @param high double
     * @note This function is used to filter the data
    */
    inline void bandpass(cv::InputArray _a, cv::OutputArray _b, double low, double high) {
        if (const Mat a = _a.getMat(); a.total() < 3) {
            a.copyTo(_b);
        } else {

            // Convert to frequency domain
            auto frequencySpectrum = Mat(a.rows, a.cols, CV_32F);
            timeToFrequency(a, frequencySpectrum, false);

            // Make the filter
            Mat filter = frequencySpectrum.clone();
            butterworth_bandpass_filter(filter, low, high, 8);

            // Apply the filter
            multiply(frequencySpectrum, filter, frequencySpectrum);

            // Convert to time domain
            frequencyToTime(frequencySpectrum, _b);
        }
    };

    /**
     * @defgroup Filters timeToFrequency
     * @brief Convert time domain to frequency domain
     * @param _a cv::InputArray
     * @param _b cv::OutputArray
     * @note This function is used to convert time domain to frequency domain
    */
    inline void frequencyToTime(InputArray _a, OutputArray _b) {
        Mat a = _a.getMat();

        // Inverse fourier transform
        idft(a, a);

        // Split into planes; plane 0 is output
        Mat outputPlanes[2];
        split(a, outputPlanes);
        auto output = Mat(a.rows, 1, a.type());
        normalize(outputPlanes[0], output, 0, 1, NORM_MINMAX);
        output.copyTo(_b);
    }

    /**
     * @defgroup Filters pcaComponent
     * @brief Principal component analysis
     * @param _a cv::InputArray
     * @param _b cv::OutputArray
     * @param _pc cv::OutputArray
     * @param low int
     * @param high int
     * @note This function is used to perform principal component analysis
    */
    inline void pcaComponent(cv::InputArray _a, cv::OutputArray _b, cv::OutputArray _pc, int low, int high) {
        Mat a = _a.getMat();
        CV_Assert(a.type() == CV_64F);

        // Perform PCA
        cv::PCA pca(a, cv::Mat(), PCA::DATA_AS_ROW);

        // Calculate PCA components
        cv::Mat pc = a * pca.eigenvectors.t();

        // Band mask
        const int total = a.rows;
        Mat bandMask = Mat::zeros(a.rows, 1, CV_8U);
        bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);

        // Identify most distinct
        std::vector<double> vals;
        for (int i = 0; i < pc.cols; i++) {
            cv::Mat magnitude = Mat(pc.rows, 1, CV_32F);
            // Calculate spectral magnitudes
            cv::timeToFrequency(pc.col(i), magnitude, true);
            // Normalize
            //printMat<float>("magnitude1", magnitude);
            cv::normalize(magnitude, magnitude, 1, 0, NORM_L1, -1, bandMask);
            //printMat<float>("magnitude2", magnitude);
            // Grab index of max
            double min, max;
            Point pmin, pmax;
            cv::minMaxLoc(magnitude, &min, &max, &pmin, &pmax, bandMask);
            vals.push_back(max);

        }

        // Select most distinct
        int idx[2];
        cv::minMaxIdx(vals, nullptr, nullptr, nullptr, &idx[0]);
        if (idx[0] == -1) {
            pc.col(1).copyTo(_b);
        } else {
            //pc.col(1).copyTo(_b);
            pc.col(idx[1]).copyTo(_b);
        }

        pc.copyTo(_pc);
    }
};

#endif /* opencv_hpp */