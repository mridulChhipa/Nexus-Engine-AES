#ifndef KEYEXPANSION_HPP
#define KEYEXPANSION_HPP

#include <vector>
#include <stdexcept>
#include "AESCore.hpp"

namespace AESCore
{

    inline void rot_word(uint8_t *word)
    {
        uint8_t temp = word[0];
        word[0] = word[1];
        word[1] = word[2];
        word[2] = word[3];
        word[3] = temp;
    }

    inline void sub_word(uint8_t *word)
    {
        for (int i = 0; i < 4; i++)
        {
            word[i] = substitution_box[word[i]];
        }
    }

    inline std::vector<uint8_t> ExpandKey(const std::vector<uint8_t> &orig_key)
    {
        if (orig_key.size() != 16)
        {
            throw std::invalid_argument("AES-128 requires a 16-byte key.");
        }

        std::vector<uint8_t> expanded_keys(176);

        for (int i = 0; i < 16; i++)
        {
            expanded_keys[i] = orig_key[i];
        }

        int bytes = 16;
        int iter = 1;
        uint8_t temp[4];

        while (bytes < 176)
        {
            for (int i = 0; i < 4; i++)
            {
                temp[i] = expanded_keys[bytes - 4 + i];
            }

            if (bytes % 16 == 0)
            {
                rot_word(temp);
                sub_word(temp);
                temp[0] ^= round_consts[iter];
                iter++;
            }

            for (int i = 0; i < 4; i++)
            {
                expanded_keys[bytes] = expanded_keys[bytes - 16] ^ temp[i];
                bytes++;
            }
        }

        return expanded_keys;
    }

}

#endif