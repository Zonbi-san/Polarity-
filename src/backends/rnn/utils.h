#include <unordered_map>
#include <armadillo>

typedef arma::colvec Action;
typedef arma::mat Behaviour;
typedef std::vector<Behaviour> BehaviourListList;

typedef std::string TextWord;
typedef std::vector<TextWord> TextBehaviourList;
typedef std::vector<TextBehaviourList> TextBehaviourListList;

typedef std::unordered_map<TextWord, int> TextVocab;
typedef std::unordered_map<int, TextWord> TextVocabRev;
typedef std::vector<std::pair<TextWord, int>> TextOccurrenceCountsVec;
typedef std::unordered_map<TextWord, int> TextOccurrenceCountsMap;

#define UNKNOWN_CHAR_VAL "UNKNOWN_CHAR"