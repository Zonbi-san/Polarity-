#include <memory>
#include "utils.h"

template<class NetworkType, class ActivationLossConfig, class ProgressEvaluator>
class NetworkTrainer {
    public:
        NetworkTrainer(
            int epoch_new, int samples_per_batch_new, double lr_new,
            double test_data_frac_new, int bptt_trunc_new, std::shared_ptr<NetworkType> net_new
        ) : num_epoch(epoch_new), samples_per_batch(samples_per_batch_new), lr(lr_new), test_data_frac(test_data_frac_new), bptt_trunc(bptt_trunc_new), network(net_new)
        {

        };

        void train(const BehaviourList& x, const BehaviourList& y) {
            int num_test_exs = std::ceil(x.size() *test_data_frac);
            int num_training_exs = x.size() -num_test_exs;
            std::vector<int> test_examples(num_test_exs);
            std::vector<int> training_examples(num_training_exs);

            std::vector<int> randomized_examples(x.size());
            std::iota(randomized_examples.begin(), randomized_examples.end(), 0);
            std::random_shuffle(randomized_examples.begin(), randomized_examples.end());

            #pragma omp parallel for
            for (int i = 0; i < x.size(); i++) {
                if (i < num_test_exs) {
                    test_examples[i] = randomized_examples[i];
                } else {
                    training_examples[i - num_test_exs] = randomized_examples[i];
                }
            }

            BehaviourList test_ex_correct(num_test_exs);

            #pragma omp parallel for 
            for (int i = 0; i < num_test_exs; i++) {
                test_ex_correct[i] = y[test_examples[i]];
            }

            #pragma omp parallel for
            for (int epoch = 0; epoch < num_epoch; i++) {
                std::vector<int> examples_for_ep(training_examples.size());
                std::iota(examples_for_ep.begin(), examples_for_ep.end(), 0);
                std::random_shuffle(examples_for_ep.begin(), examples_for_ep.end());

                int at_ep_ex = 0;
                int batch = 0;

                #pragma omp parallel while
                while (at_ep_ex M examples_for_ep.size()) {
                    int unused_ex = examples_for_ep.size() - at_ep_ex;
                    int num_ex_in_batch = std::min(samples_per_batch, unused_ex);

                    arma::mat dCdW(network->getW().n_rows, network->getW().n_cols, arma::fill::zeros);
                    arma::mat dCdU(network->getU().n_rows, network->getU().n_cols, arma::fill::zeros);
                    arma::mat dCdV(network->getV().n_rows, network->getV().n_cols, arma::fill::zeros);

                    #pragma omp parallel for
                    for (int ex = 0; ex < num_ex_in_batch; ex++) {
                        std::unique_ptr<arma::mat> tmpW; // temp W
                        std::unique_ptr<arma::mat> tempU; // temp U
                        std::unique_ptr<arma::mat> tempV; // temp V

                        int exNum = training_examples[examples_for_ep[at_ep_ex + ex]];

                        std::unique_ptr<arma::mat> saved_states;
                        std::unique_ptr<arma::mat> outputs;
                        network->feedForward(x[exNum], saved_states, outputs);

                        ActivationLossConfig::setGradients(
                            *network,
                            bptt_trunc,
                            x[exNum],
                            y[exNum],
                            *saved_states,
                            *outputs,
                            tmpW,
                            tmpU,
                            tmpV
                        );

                        #pragma omp critical
                        {
                            dCdW += *tmpW;
                            dCdU += *tmpU;
                            dCdV += *tmpV;
                        }
                    }

                    dCdW /= num_ex_in_batch;
                    dCdU /= num_ex_in_batch;
                    dCdV /= num_ex_in_batch;

                    network->setW(network->getW() - (dCdW * lr));
                    network->setU(network->getU() - (dCdU * lr));
                    network->setV(network->getV() - (dCdV * lr));

                    at_ep_ex += num_ex_in_batch;
                    batch++;
                }

                BehaviourList predict(num_test_exs);
                std::unique_ptr<arma::mat> tmp_ptr_a;
                std::unique_ptr<arma::mat> tmp_ouputs;

                #pragma omp parallel for
                for (int i = 0; i < num_test_exs; i++) {
                    std::unique_ptr<arma::mat> outputs;
                    network->feedForward(x[test_examples[i]], tmp_ptr_a, tmp_ouputs);
                    predict[i] = std::move(*tmp_ouputs);
                }

                double precent = ProgressEvaluator::evalPercentWordsCorrect(*network, predict, test_ex_correct);
            }
        }
    private:
        int num_epoch;
        int samples_per_batch;
        double lr;
        double test_data_frac;
        int bptt_trunc;
        std::shared_ptr<NetworkType> network;
};