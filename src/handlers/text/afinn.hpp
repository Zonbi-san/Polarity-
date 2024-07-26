#ifndef AFINN_HPP
#define AFINN_HPP
#include "../../../includes/afinn.h"

#include <unordered_map>
#include <string>
#include <fstream>

inline int AFINN::getText(const std::string& text) {
    if (textPolarity.empty()) {
        if (std::ifstream file ("data/AFINN.txt"); file.is_open()) {
            std::string line;
            while (getline(file, line)) {
                unsigned long pos = line.find('\t');
                std::string word = line.substr(0, pos);
                int value = std::stoi(line.substr(pos + 1));
                textPolarity[word] = value;
            }
            file.close();
        }

        if (auto it = textPolarity.find(text); it != textPolarity.end()) {
            return it->second;
        } return 0;
    } else {
        if (auto it = textPolarity.find(text); it != textPolarity.end()) {
            return it->second;
        } return 0;
    }
}

inline int AFINN::getEmoji(const std::string &emoji) {
    if (const auto it = emojiPolarity.find(emoji); it != emojiPolarity.end()) {
        return it->second;
    }
    return 0;
}

#endif // AFINN_HPP
