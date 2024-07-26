#ifndef UTILS_H
#define UTILS_H
#include <unordered_map>
#include <armadillo>

typedef arma::colvec Action;
typedef arma::mat Behaviour;
typedef std::vector<Behaviour> BehaviourList;

typedef std::string ActionBehaviour;
typedef std::vector<ActionBehaviour> ActionBehaviourList;
typedef std::vector<ActionBehaviourList> ActionBehaviourListList;

typedef std::unordered_map<ActionBehaviour, int> ActionKnowledge;
typedef std::unordered_map<int, ActionBehaviour> ActionKnowledgeRev;
typedef std::vector<std::pair<ActionBehaviour, int>> ActionOccurrenceCountsVec;
typedef std::unordered_map<ActionBehaviour, int> ActionOccurrenceCountsMap;

#define UNKNOWN_CHAR_VAL "UNKNOWN_CHAR"
#endif // UTILS_H
