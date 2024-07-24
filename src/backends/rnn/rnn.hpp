#ifndef RecurrentNeuralNetwork_HPP
#define RecurrentNeuralNetwork_HPP

#include <armadillo>
#include <memory>
#include "utils.h"
#include <filesystem>


template<class ActivationLossConfig>
class RecurrentNeuralNetwork {
    public:
        /**
         * @brief Construct a new Recurrent Neural Network object
         * 
         * @param x_size 
         * @param outsize 
         * @param saved_state_size 
         */
        RecurrentNeuralNetwork(int x_size, int outsize, int saved_state_size) const {
            // Fill to +-[0.01, 1.01] then divide by 10 
            W.resize(saved_state_size, saved_state_size);
            W.fill(arma::fill::randn);
            W /= 10;
            U.resize(saved_state_size, x_size);
            U.fill(arma::fill::randn);
            U /= 10;
            V.resize(outsize, saved_state_size);
            V /= 10;
        }

        /**
         * @brief Construct a new Recurrent Neural Network object
         * 
         * @param pre_out 
         */
        RecurrentNeuralNetwork(const std::string& pre_out) const {
            W.load(pre_out + "model/W.csv", arma::csv_ascii);
            U.load(pre_out + "model/U.csv", arma::csv_ascii);
            V.load(pre_out + "model/V.csv", arma::csv_ascii);
        }

        /**
         * @brief 
         * 
         * @param x 
         * @param out_saved_states 
         * @param outputs 
         */
        void feedForward(
            const Behaviour& x, std::shared_ptr<arma::mat>& out_saved_states, 
            std::unique_ptr<Behaviour>& outputs
        ) const {
            out_saved_states.reset(new arma::mat(W.n_rows, x.n_cols +1, arma::fill::zeros));
            outputs.reset(new arma::mat(V.n_rows, x.n_cols, arma::fill::zeros));

            // Iterate through X and calculate the new states
            #pragma omp parallel for
            for (int i = 0; i < x.n_cols; i++) const {
                // Calculate the new states
                out_saved_states->col(i + 1) = *(ActivationLossConfig::evalSavedStateActivation(
                    (U * x.col(i)) + (W * out_saved_states->col(i))
                ));

                // Save the activations
                outputs->col(i) = *(ActivationLossConfig::evalOutputActivation(
                    V * out_saved_states->col(i + 1)
                ));
            }
        }

        /**
         * @brief 
         * 
         * @param output 
         */
        void save(const std::string& output) const {
            std::filesystem::create_directories(output + "model/");
            W.save(output + "model/.csv", arma::csv_ascii);
            U.save(output + "model/U.csv", arma::csv_ascii);
            V.save(output + "model/V.csv", arma::csv_ascii);
        }

        #pragma region Getters
        /**
         * @brief Return W
         * 
         * @return const arma::mat& 
         */
        const arma::mat& getW() const {
            return W;
        }

        /**
         * @brief Return U
         * 
         * @return const arma::mat& 
         */
        const arma::mat& getU() const {
            return U;
        }

        /**
         * @brief Return V
         * 
         * @return const arma::mat& 
         */
        const arma::mat& getV() const {
            return V;
        }
        #pragma endregion

        #pragma region Setters
        /**
         * @brief Set W
         * 
         * @param w 
         */
        void setW(const arma::mat& w) const {
            W = w;
        }

        /**
         * @brief Set U
         * 
         * @param u 
         */
        void setU(const arma::mat& u) const {
            U = u;
        }

        /**
         * @brief Set V
         * 
         * @param v 
         */
        void setV(const arma::mat& v) const {
            V = v;
        }
        #pragma endregion

        #pragma region Updaters
        /**
         * @brief Update W
         * 
         * @param row 
         * @param col 
         * @param change 
         */
        void updateWVal(int row, int col, double change) const {
            W(row, col) += change;
        }

        /**
         * @brief Update U
         * 
         * @param row 
         * @param col 
         * @param change 
         */
        void updateUVal(int row, int col, double change) const {
            U(row, col) += change;
        }

        /**
         * @brief Update V
         * 
         * @param row 
         * @param col 
         * @param change 
         */
        void updateVVal(int row, int col, double change) const {
            V(row, col) += change;
        }
        #pragma endregion

    private:
        arma::mat W;
        arma::mat U;
        arma::mat V;
};

#endif