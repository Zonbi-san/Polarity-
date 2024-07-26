#ifndef Moves_HPP
#define Moves_HPP
#include "../../regression.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include "../../converter/matrix.hpp"

using namespace cv;

class MovementProcessor {
    public:
        MovementProcessor();

        void appendToMatrix(Mat3D mat) {
            // Convert the matrix to something usable
            Matrix newMat = convertMat3D(mat);
            this->currentMatrix = newMat;

            // append are current matrix to the final matrix
            this->matrix.append(this->matrix, this->currentMatrix);
        }

        Matrix getMatrix() {
            return this->matrix;
        }

        Matrix processMatrix() {
            Regression reg;
            std::vector<float> x, y;

            // Flatten the matrix into two 1D arrays
            for (const auto& mat : this->matrix.M) {
                for (int i = 0; i < mat.size(); i++) {
                    if (i < 3) {
                        x.push_back(mat[i]);
                    } else {
                        y.push_back(mat[i]);
                    }
                }
            }

            // Convert the vectors to arrays
            float* x_arr = &x[0];
            float* y_arr = &y[0];

            // Pass the arrays to takeIn
            reg.takeIn(x_arr, y_arr);
                      
            // predict expected values
            std::vector<float> pred;
            #pragma omp parallel for
            for (int i = 0; i < 5; i++) {
                pred.push_back(reg.predict(x_arr[i]));
            }

            // Convert the predicted values to a matrix
            Matrix predMat = convertMat3D(Mat3D(pred[0], pred[1], pred[2]));
            return predMat;
        }
    
    private:
        ///
        Matrix currentMatrix;

    public:
        Matrix matrix;

};

#endif