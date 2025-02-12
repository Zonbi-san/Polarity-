#ifndef NED_HPP
#define NED_HPP
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "../../../includes/emotions.h"
#include "../../tools/image_smoothing.hpp"

using namespace cv;
using namespace cv::face;

struct Condition {
	std::function<bool()> check;
	std::string result;
};

inline std::string NED::detectEmotion(cv::VideoCapture camera) {
    std::string label;
    bool firstFrame = true;
    CascadeClassifier faceDetector;
    Mat frame, gray, lastframe;
    std::vector<Rect> faces;
    std::vector<std::vector<Point2f>> landmarks;

    std::string cascadeName = "data/haarcascade_frontalface_alt2.xml";
    std::string modelFileName = "data/lbfmodel.yml";


    try {
        faceDetector.load(cascadeName);
    } catch (std::exception&) {
        return "Error Loading Cascade Classifier";
    }

    cv::Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel(modelFileName);

    if (!camera.isOpened()) {
        try {
            camera.open(0);
        } catch (std::exception&) {
            return "Camera not found";
        }
    }

    while (camera.read(frame)) {
		// check the buffer for the next frame
		if (frame.empty()) {
			break;
		}

		// Buffer Boundary checks!
		if (frame.cols > 1920 || frame.rows > 1080) {
			break;
		}


		ImageSmoothing::smooth(frame);

        faces.clear();
        landmarks.clear();
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if (firstFrame) {
            lastframe = gray.clone();
            firstFrame = false;
        }

        faceDetector.detectMultiScale(gray, faces);

        if (facemark->fit(gray, faces, landmarks)) {
            label = getEmotion(
                gray, lastframe, faceDetector,
                landmarks, faces, facemark
            );
        } else {
            label = "No Known Emotion Detected...";
        }

        lastframe = gray.clone();
    }

    return label;
};

inline float NED::getDistance(
    const Point2f& point1, const Point2f& point2
) {
    return sqrt(
        powf((
            point1.x - point2.x
        ), 2) + powf((
            point1.y - point2.y
        ), 2)
    );
};

inline std::string NED::getEmotion(
    InputArray frame, InputArray lastFrame,
    CascadeClassifier classifier,
    const std::vector<std::vector<Point2f>> &landmarks,
    const std::vector<Rect>& faces, const Ptr<Facemark>& facemark
) {
    std::string result;
    std::vector<std::vector<Point2f>> lastLandmarks;
    std::vector<Rect> lastFaces;

    classifier.detectMultiScale(lastFrame, lastFaces);
    facemark->fit(lastFrame, lastFaces, lastLandmarks);

    std::vector<float> distTo33(landmarks[0].size());
    std::vector<float> lastDistTo33(lastLandmarks[0].size());

    #pragma omp parallel for
    for (int i = 0; i < landmarks[0].size(); i++) {
        distTo33[i] = getDistance(landmarks[0][i], landmarks[0][33]);
        lastDistTo33[i] = getDistance(lastLandmarks[0][i], lastLandmarks[0][33]);
    }

	std::vector<Condition> conditions = {
		{
			#pragma region Surprise
			[&]() {
				return dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
					dist17to36 > 0 && dist26to45 > 0 && // AU 2
					dist33to38 > 0 && dist33to37 > 0 && dist33to43 > 0 && dist33to44 > 0  && // AU 5
					(
						dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0 && // AU 26
						dist8to33 > 0 && dist7to33 > 0 && dist9to33 > 0
					); // AU 27
			},
			"Surprise"
			#pragma endregion
		},
		{
			#pragma region Sadness
			[&]() {
				return dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
					dist21to22 < 0 && dist20to23 < 0 && // AU 4
					dist5to48 < 0 && dist11to54 < 0; // AU 15
			},
			"Sadness"
			#pragma endregion
		},
		{
			#pragma region Fear
			[&]() {
				return dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
					dist17to36 > 0 && dist26to45 > 0 && // AU 2
					dist21to22 < 0 && dist20to23 < 0 && // AU 4
					dist33to37 > 0 && dist33to38 > 0 && dist33to43 > 0 && dist33to44 > 0 && // AU 5
					(
						dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0 && // AU 26
						dist8to33 > 0 && dist7to33 > 0 && dist9to33 > 0
					); // AU 27
			},
			"Fear"
			#pragma endregion
		},
		{
			#pragma region Disgust
			[&]() {
				return dist21to22 < 0 && dist20to23 < 0 && dist21to33 < 0 && dist22to33 < 0 && // AU 9
					dist33to56 > 0 && dist33to57 > 0 && dist33to58 > 0 && // AU 16
					(
						(dist5to48 < 0 && dist11to54 < 0) ||
						(dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0)
					); // AU 15 or AU 26
			},
			"Disgust"
			#pragma endregion
		},
		{
			#pragma region Angry
			[&]() {
				return dist21to22 < 0 && dist20to23 < 0 && // AU 4
					dist33to38 > 0 && dist33to37 > 0 && dist33to43 > 0 && dist33to44 > 0 &&// AU 5
					dist27to40 < 0 && dist27to47 < 0 && // AU 7
					((dist61to67 < 0 && dist62to66 < 0 && dist63to65 < 0 ) || // AU 23 or AU 24
					(dist50to61 < 0 && dist51to62 < 0 && dist52to63 < 0));
			},
			"Angry"
			#pragma endregion
		},
		{
			#pragma region Happiness
			[&]() {
				return dist36to48 < 0 && dist45to54 < 0;
			},
			"Happiness"
			#pragma endregion
		},
		{
			#pragma region Neutral
			[&]() {
				return true;
			},
			"Neutral"
			#pragma endregion
		},
		{
			#pragma region None
			[&]() {
				return false;
			},
			"No emotion detected"
			#pragma endregion
		}
	};

	for (const auto& c	: conditions) {
		if (c.check()) {
			result = c.result;
			break;
		}
	}

    return result;
};
#endif // NED_HPP
