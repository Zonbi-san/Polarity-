#ifndef BehaviouralRnn_HPP
#define BehaviouralRnn_HPP

#include <memory>
#include <ranges>
#include <utility>
#include "utils.h"
#include "rnn.hpp"

template<class ActivationLossConfig>
class BehaveRnn : public RecurrentNeuralNetwork<ActivationLossConfig> {
public:
    BehaveRnn(
        int x_size, int out_size, int saved_state_size,
        std::shared_ptr<const ActionKnowledge> new_knowledge
    ): RecurrentNeuralNetwork<ActivationLossConfig>(x_size, out_size, saved_state_size) {
        knowledge_ = std::move(new_knowledge);

        for (const auto &[fst, snd]: *knowledge_) {
            knowledge_rev_[snd] = fst;
        }
    };

    explicit BehaveRnn(const std::string &prev_output_pref): RecurrentNeuralNetwork<ActivationLossConfig>(
        prev_output_pref) {
        ActionKnowledge knowledge_in;
        std::ifstream ifs(prev_output_pref + "model/knowledge.map");
        std::string _tmp_line;

        while (getline(ifs, _tmp_line)) {
            ActionBehaviourList actions;
            //TODO: Find a correct way to Type Cast and Line Split
            knowledge_in[actions[0]] = dynamic_cast<int>(actions[1]);
            knowledge_rev_[knowledge_in[actions[0]]] = actions[0];
        }

        knowledge_ = std::make_shared<ActionKnowledge>(knowledge_in);
    }

    std::unique_ptr<Action> AActionToAction(const ActionBehaviour &action_behav) const {
        std::unique_ptr<Action> ret(new Action(knowledge_->size(), arma::fill::zeros));

        if (!knowledge_->contains(action_behav)) {
            (*ret)(knowledge_->at(UNKNOWN_CHAR_VAL)) = 1;
        } else {
            (*ret)(knowledge_->at(action_behav)) = 1;
        }

        return ret;
    }

    std::unique_ptr<ActionBehaviour> BActionToBehaviour(const Behaviour &behaviour) const {
        return std::make_unique<ActionBehaviour>(knowledge_rev_.at(behaviour.index_max()()));
    }

    std::unique_ptr<Behaviour> BBehaviourToBehaviour(const ActionBehaviour &act_beh) const {
        std::unique_ptr<Behaviour> ret(
            new Behaviour(knowledge_->size(), act_beh.size(), arma::fill::zeros)
        );

        for (int at = 0; at < act_beh.size(); at++) {
            if (!knowledge_->contains(act_beh)) {
                (*ret)(knowledge_->at(UNKNOWN_CHAR_VAL), at) = 1;
            } else {
                (*ret)(knowledge_->at(act_beh), at) = 1;
            }
        }

        return ret;
    }

    std::unique_ptr<ActionBehaviourList> BehaviourToActionBehaviour(const Behaviour &behaviour) const {
        std::unique_ptr<ActionBehaviourList> ret(new ActionBehaviourList);

        for (int at = 0; at < behaviour.n_cols; at++) {
            ret->push_back(knowledge_rev_.at(behaviour.col(at).index_max()));
        }

        return ret;
    }

    std::unique_ptr<ActionBehaviourList> ABehaviourListToBehaviourList(
        const ActionBehaviourList &behaviours
    ) const {
        std::unique_ptr<ActionBehaviourList> ret(new ActionBehaviourList);

        for (const auto &behaviour: behaviours) {
            ret->push_back(std::move(*(BBehaviourToBehaviour(behaviour))));
        }

        return ret;
    }

    static void BehaviourListToTrainingBehaviourList(
        const BehaviourList &abl,
        std::unique_ptr<BehaviourList> &out_x,
        std::unique_ptr<BehaviourList> &out_y
    ) {
        out_x = std::make_unique<BehaviourList>();
        out_y = std::make_unique<BehaviourList>();

        for (const auto &at: abl) {
            out_x->push_back(at.cols(0, at.n_cols - 2));
            out_y->push_back(at.cols(0, at.n_cols - 2));
        }
    }

    std::unique_ptr<ActionBehaviourListList> BListToABL(const BehaviourList &behaviours) const {
        std::unique_ptr<ActionBehaviourListList> ret(new ActionBehaviourListList);

        for (const auto &behaviour: behaviours) {
            ret->push_back(std::move(*(BehaviourToActionBehaviour(behaviour))));
        }

        return ret;
    }

    std::unique_ptr<ActionBehaviourList> generateBehaviour(const ActionBehaviour &begin, const ActionBehaviour &end,
                                                           int max) const {
        std::unique_ptr<ActionBehaviourList> abl(new ActionBehaviourList);

        abl->push_back(begin);

        arma::colvec last_saved_state(this->W.n_rows, arma::fill::zeros);
        Action last_out = *(AActionToAction(begin));

        while (abs->back() != end && abl->size() < max) {
            last_saved_state = *(ActivationLossConfig::evalSavedStateActivation(
                (this->U * last_out) + (this->W * last_saved_state)
            ));

            last_out = *(ActivationLossConfig::evalOutputActivation(this->V * last_saved_state));
            last_out(knowledge_->at(UNKNOWN_CHAR_VAL)) = 0;
            abl->push_back(*(BActionToBehaviour(last_out)));
        }

        return abl;
    }

    void save(const std::string &output) const {
        RecurrentNeuralNetwork<ActivationLossConfig>::save(output);

        std::ofstream ofs(output + "model/knowledge.map");
        for (const auto &[fst, snd]: *knowledge_) {
            ofs << fst << "" << snd << "\n";
        }
    }

private:
    std::shared_ptr<const ActionKnowledge> knowledge_;
    ActionKnowledgeRev knowledge_rev_;
};

#endif
