#ifndef AES_NODE_HPP
#define AES_NODE_HPP

#include <iomanip>
#include <iostream>
#include <cstring>
#include <memory>
#include <vector>

#include "Types.hpp"
#include "BaseNode.hpp"
#include "RawBuffer.hpp"
#include "AESCore.hpp"
#include "KeyExpansion.hpp"

enum class AESMode
{
    Encrypt,
    Decrypt
};

class AESNode : public BaseNode
{
    std::vector<uint8_t> expanded_keys;
    AESMode mode;

    void add_round_key(unsigned char *state, int round)
    {
        for (int i = 0; i < 16; i++)
        {
            state[i] ^= expanded_keys[(round * 16) + i];
        }
    }

    void mix_columns(unsigned char *state)
    {
        unsigned char tmp[16];
        std::memcpy(tmp, state, 16);

        for (int i = 0; i < 16; i += 4)
        {
            state[i] = AESCore::galois_field_mul(2, tmp[i]) ^ AESCore::galois_field_mul(3, tmp[i + 1]) ^ tmp[i + 2] ^ tmp[i + 3];
            state[i + 1] = tmp[i] ^ AESCore::galois_field_mul(2, tmp[i + 1]) ^ AESCore::galois_field_mul(3, tmp[i + 2]) ^ tmp[i + 3];
            state[i + 2] = tmp[i] ^ tmp[i + 1] ^ AESCore::galois_field_mul(2, tmp[i + 2]) ^ AESCore::galois_field_mul(3, tmp[i + 3]);
            state[i + 3] = AESCore::galois_field_mul(3, tmp[i]) ^ tmp[i + 1] ^ tmp[i + 2] ^ AESCore::galois_field_mul(2, tmp[i + 3]);
        }
    }

    void sub_bytes(unsigned char *state)
    {
        for (int i = 0; i < 16; i++)
        {
            state[i] = AESCore::substitution_box[state[i]];
        }
    }

    void shift_rows(unsigned char *state)
    {
        unsigned char temp[16];
        std::memcpy(temp, state, 16);

        state[0] = temp[0];
        state[4] = temp[4];
        state[8] = temp[8];
        state[12] = temp[12];

        state[1] = temp[5];
        state[5] = temp[9];
        state[9] = temp[13];
        state[13] = temp[1];

        state[2] = temp[10];
        state[6] = temp[14];
        state[10] = temp[2];
        state[14] = temp[6];

        state[3] = temp[15];
        state[7] = temp[3];
        state[11] = temp[7];
        state[15] = temp[11];
    }

    void encrypt_block(unsigned char *curr)
    {
        add_round_key(curr, 0);

        for (int round = 1; round < 10; round++)
        {
            sub_bytes(curr);
            shift_rows(curr);
            mix_columns(curr);
            add_round_key(curr, round);
        }

        sub_bytes(curr);
        shift_rows(curr);
        add_round_key(curr, 10);
    }

    void inv_sub_bytes(unsigned char *state)
    {
        for (int i = 0; i < 16; i++)
        {
            state[i] = AESCore::inv_subs_box[state[i]];
        }
    }

    void inv_shift_rows(unsigned char *state)
    {
        unsigned char temp[16];
        std::memcpy(temp, state, 16);

        state[0] = temp[0];
        state[4] = temp[4];
        state[8] = temp[8];
        state[12] = temp[12];

        state[1] = temp[13];
        state[5] = temp[1];
        state[9] = temp[5];
        state[13] = temp[9];

        state[2] = temp[10];
        state[6] = temp[14];
        state[10] = temp[2];
        state[14] = temp[6];

        state[3] = temp[7];
        state[7] = temp[11];
        state[11] = temp[15];
        state[15] = temp[3];
    }

    void inv_mix_columns(unsigned char *state)
    {
        unsigned char tmp[16];
        std::memcpy(tmp, state, 16);

        for (int i = 0; i < 16; i += 4)
        {
            state[i] = AESCore::galois_field_mul(0x0e, tmp[i]) ^ AESCore::galois_field_mul(0x0b, tmp[i + 1]) ^ AESCore::galois_field_mul(0x0d, tmp[i + 2]) ^ AESCore::galois_field_mul(0x09, tmp[i + 3]);
            state[i + 1] = AESCore::galois_field_mul(0x09, tmp[i]) ^ AESCore::galois_field_mul(0x0e, tmp[i + 1]) ^ AESCore::galois_field_mul(0x0b, tmp[i + 2]) ^ AESCore::galois_field_mul(0x0d, tmp[i + 3]);
            state[i + 2] = AESCore::galois_field_mul(0x0d, tmp[i]) ^ AESCore::galois_field_mul(0x09, tmp[i + 1]) ^ AESCore::galois_field_mul(0x0e, tmp[i + 2]) ^ AESCore::galois_field_mul(0x0b, tmp[i + 3]);
            state[i + 3] = AESCore::galois_field_mul(0x0b, tmp[i]) ^ AESCore::galois_field_mul(0x0d, tmp[i + 1]) ^ AESCore::galois_field_mul(0x09, tmp[i + 2]) ^ AESCore::galois_field_mul(0x0e, tmp[i + 3]);
        }
    }

    void decrypt_block(unsigned char *curr)
    {
        add_round_key(curr, 10);
        inv_shift_rows(curr);
        inv_sub_bytes(curr);

        for (int round = 9; round > 0; round--)
        {
            add_round_key(curr, round);
            inv_mix_columns(curr);
            inv_shift_rows(curr);
            inv_sub_bytes(curr);
        }

        add_round_key(curr, 0);
    }

public:
    AESNode(std::string id, const std::vector<uint8_t> &key, AESMode m) : BaseNode(id), mode(m)
    {
        expanded_keys = AESCore::ExpandKey(key);
    }

    [[nodiscard]] Packet process(const Packet &packet) override
    {
        if (!std::holds_alternative<std::shared_ptr<RawBuffer>>(packet.core_data))
        {
            return packet;
        }

        auto originalBuffer = std::get<std::shared_ptr<RawBuffer>>(packet.core_data);
        size_t originalSize = originalBuffer->get_size();

        if (mode == AESMode::Encrypt)
        {
            uint8_t paddingVal = 16 - (originalSize % 16);
            size_t paddedSize = originalSize + paddingVal;

            auto newBuffer = std::make_shared<RawBuffer>(paddedSize);
            newBuffer->copy(originalBuffer->get_data(), originalSize);

            unsigned char *rawPtr = const_cast<unsigned char *>(newBuffer->get_data());

            for (size_t i = originalSize; i < paddedSize; i++)
            {
                rawPtr[i] = paddingVal;
            }

            for (size_t i = 0; i < paddedSize; i += 16)
            {
                encrypt_block(rawPtr + i);
            }

            std::cout << "[AES] Encrypted " << originalSize << " bytes -> " << paddedSize << " bytes (padded).\n";
            return Packet(newBuffer);
        }

        else
        {
            auto workBuffer = std::make_shared<RawBuffer>(originalSize);
            workBuffer->copy(originalBuffer->get_data(), originalSize);
            unsigned char *rawPtr = const_cast<unsigned char *>(workBuffer->get_data());

            for (size_t i = 0; i < originalSize; i += 16)
            {
                decrypt_block(rawPtr + i);
            }

            uint8_t paddingVal = rawPtr[originalSize - 1];

            if (paddingVal > 0 && paddingVal <= 16)
            {
                size_t trueSize = originalSize - paddingVal;

                auto finalBuffer = std::make_shared<RawBuffer>(trueSize);
                finalBuffer->copy(rawPtr, trueSize);

                std::cout << "[AES] Decrypted " << originalSize << " bytes -> " << trueSize << " bytes (unpadded).\n";
                return Packet(finalBuffer);
            }

            return Packet(workBuffer);
        }
    }

    NodeStatus get_status() const override
    {
        return NodeStatus::Ready;
    }
};

#endif