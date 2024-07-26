#ifndef ActivationLoss_HPP
#define ActivationLoss_HPP

#include <memory>
#include <armadillo>
#include "../rnn.hpp"
#include "../behave_rnn.hpp"

class ActivationLossConfig {
    public:
        static std::unique_ptr<arma::colvec> evalOutputActivation(const arma::colvec& input) {
            arma::colvec matExp = arma::exp(input);

            return std::make_unique<arma::colvec>(matExp / arma::accu(matExp));
        }

        static std::unique_ptr<arma::colvec> evalSavedStateActivation(const arma::colvec& input) {
            return std::make_unique<arma::colvec>(arma::tanh(input));
        }   

        static double evalCost(const Behaviour& correct, const Behaviour& predict) {
            return arma::accu((-1 *  correct) % arma::log(predict));
        }   

        static void setGradients(
            const BehaveRnn<ActivationLossConfig>& network,
            int bptt_truncate, const Behaviour& x, const Behaviour& y, 
            const arma::mat& saved_states, const arma::mat& outputs, 
            std::unique_ptr<arma::mat>& out_dCdW, std::unique_ptr<arma::mat>& out_dCdU,
            std::unique_ptr<arma::mat>& out_dCdV
        )   {
            	out_dCdW.reset(new arma::mat(network.getW().n_rows, network.getW().n_cols, arma::fill::zeros));
	            out_dCdU.reset(new arma::mat(network.getU().n_rows, network.getU().n_cols, arma::fill::zeros));
	            out_dCdV.reset(new arma::mat(network.getV().n_rows, network.getV().n_cols, arma::fill::zeros));
        
                #pragma omp parallel for
                for (int time = x.n_cols - 1; time >= 0; time--) {
                    // derivative of cost for V = w/r/t
                    (*out_dCdV) += arma::kron(
                        (outputs.col(time) - y.col(time)),
                        arma::trans(saved_states.col(time + 1))
                    );
                }
        }

        //TODO: Continue this
};

#endif
