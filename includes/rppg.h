#ifndef RPPG_H
#define RPPG_H

#include <vector>
#include <unordered_map>
#include <cmath>
#include <iterator>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.4
#define SEC_PER_MIN 60
#define MAX_CORNERS 10
#define MIN_CORNERS 5
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 25


typedef long double ld;
typedef unsigned int uint;
typedef std::vector<ld>::iterator vec_iter_ld;

class VectorStats {
public:
    VectorStats(vec_iter_ld start, vec_iter_ld end);

    void compute();
    ld mean() const;
    ld standardDeviation() const;

private:
    vec_iter_ld start;
    vec_iter_ld end;
    ld m1{}; // Standard Mean
    ld m2{}; // Deviation Mean
};

enum rPPGAlgorithm { g, pca, xminay };
enum faceDetAlgorithm { haar, deep };

class RPPG {
public:
    RPPG();

    static std::unordered_map<std::string, std::vector<ld>> z_score_thresholding(std::vector<ld>, int, ld);

    auto load(
        rPPGAlgorithm rppga, faceDetAlgorithm fda,
        int width, int height, double samplingFrequency,
        double rescanFrequency, int minSignalSize,
        int maxSignalSize,
        const std::string &dnnProtoPath, const std::string &dnnModelPath
    ) -> bool;

    int runDetection();

    float processFrame(cv::Mat& frameRGB, cv::Mat& frameGray, int time);

    static void exit();

    typedef std::vector<cv::Point2f> Contour2f;
    static std::unordered_map<std::string, std::vector<ld>> z_score_thresholding(
        std::vector<ld> input, int lag, 
        ld threshold, ld influence
    );

private:
    void detectFace(const cv::Mat &frameRGB, cv::Mat &frameGray);
    void setNearestBox(std::vector<cv::Rect> boxes);
    void detectCorners(cv::Mat& frameGray);
    void trackFace(cv::Mat& frameGray);
    void updateMask(cv::Mat& frameGray);
    void updateROI();
    void extractSignal_g();
    void extractSignal_pca();
    void extractSignal_xminay();
    float estimateHeartrate();
    void invalidateFace();

    // Algorithms
    rPPGAlgorithm rppga;
    faceDetAlgorithm fda;
    cv::dnn::Net dnnClassifier;

    // Settings
    cv::Size minFaceSize;
    int maxSignalSize;
    int minSignalSize;
    double rescanFrequency;
    double samplingFrequency;
    double timeBase;

    // State variables
    int64_t time;
    double fps;
    int high;
    int64_t lastSamplingTime;
    int64_t lastScanTime;
    int low;
    
    // int64_t now;
    bool faceValid;
    bool rescanFlag;

    // Tracking
    cv::Mat lastFrameGray;
    Contour2f corners;

    // Mask
    cv::Rect box;
    cv::Mat1b mask;
    cv::Rect roi;

    // Raw signal
    cv::Mat1d s;
    cv::Mat1d t;
    cv::Mat1b re;

    // Estimation
    cv::Mat1d s_f;
    cv::Mat1d bpms;
    cv::Mat1d powerSpectrum;
    double bpm = 0.0;
    double meanBpm;
    double minBpm;
    double maxBpm;
};

#endif