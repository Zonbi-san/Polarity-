#ifndef Matrix_HPP
#define Matrix_HPP

#include <iostream>
#include <vector>

struct Mat3D {
    float X;
    float Y;
    float Z;

    public:
        Mat3D() = default;
        Mat3D(float x, float y, float z) 
            : X(x), Y(y), Z(z) {};
};

struct Mat5D {
    float X;
    float Y;
    float Z;
    float Yaw;
    float Pitch;

    public:
        Mat5D() = default;
        Mat5D(
            float x, float y, float z, 
            float yaw, float pitch
        ) 
            : X(x), Y(y), Z(z), Yaw(y), Pitch(pitch) {};
};

struct Matrix {
    std::vector<std::vector<float>>  M;

    public:
        Matrix() = default;

        void clear() {
            M.clear();
        }

        void set(int i, int j, float value) {
            M[i][j] = value;
        }

        float get(int i, int j) {
            return M[i][j];
        }

    [[noreturn]] void print() const {
            #pragma omp parallel for
            for (int i = 0; i < M.size(); i++) {
                #pragma omp parallel for
                for ( int j = 0; j < M[i].size(); i++) {
                    std::cout << M[i][j] << "\n" << std::endl;
                }
            }
        }

        static inline Matrix append(
            Matrix& newMat, Matrix current
        ) {
            // Append the new Matrix, to our current matrix
            std::vector<Matrix> matrix;
            matrix.push_back(current);
            matrix.push_back(newMat);
            
            // Update the current matrix
            current = matrix[1];

            return current;
        }
};

inline Matrix convertMat3D(Mat3D position) {
    Matrix matrix;

    matrix.M[0][0] = position.X;
    matrix.M[0][1] = position.Y;
    matrix.M[0][2] = position.Z;
    matrix.M[0][3] = 0;
    matrix.M[0][4] = 0;

    return matrix;
};

inline Matrix convertMat5D(Mat5D position) {
    Matrix matrix;

    matrix.M[0][0] = position.X;
    matrix.M[0][1] = position.Y;
    matrix.M[0][2] = position.Z;
    matrix.M[0][3] = position.Yaw;
    matrix.M[0][4] = position.Pitch;

    return matrix;
};

#endif