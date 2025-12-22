#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <filesystem>

#include "../include/NexusEngine.hpp"
#include "../include/AESNode.hpp"
#include "../include/LoggerNode.hpp"
#include "../include/RawBuffer.hpp"
#include "../include/Types.hpp"

std::shared_ptr<RawBuffer> load_file(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    auto buffer = std::make_shared<RawBuffer>(size);

    unsigned char *ptr = const_cast<unsigned char *>(buffer->get_data());

    if (!file.read(reinterpret_cast<char *>(ptr), size))
    {
        throw std::runtime_error("Failed to read file data");
    }

    return buffer;
}

void save_file(const std::string &filename, std::shared_ptr<RawBuffer> buffer)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to create file: " + filename);
    }

    file.write(reinterpret_cast<const char *>(buffer->get_data()), buffer->get_size());
    std::cout << "[IO] Saved " << buffer->get_size() << " bytes to " << filename << "\n";
}

int main()
{
    try
    {
        std::cout << "=== Nexus AES-128 System ===\n\n";

        std::string secretFile = "secret_plans.txt";
        {
            std::ofstream outfile(secretFile);
            outfile << "The target location is 45.912, -12.001. Attack at dawn.";
        }

        std::vector<uint8_t> key = {
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xa, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

        NexusEngine engine;

        std::cout << "--- Phase 1: Encryption ---\n";

        auto plainBuffer = load_file(secretFile);

        engine << std::make_unique<AESNode>("AES-Encrypter", key, AESMode::Encrypt)
               << std::make_unique<LoggerNode>("EncryptionLogger");

        Packet p1(plainBuffer);
        engine(p1);

        AESNode encryptor("Encryptor", key, AESMode::Encrypt);
        Packet encryptedPacket = encryptor.process(Packet(plainBuffer));

        if (std::holds_alternative<std::shared_ptr<RawBuffer>>(encryptedPacket.core_data))
        {
            auto encBuf = std::get<std::shared_ptr<RawBuffer>>(encryptedPacket.core_data);
            save_file("secret_plans.enc", encBuf);
        }

        std::cout << "\n--- Phase 2: Decryption ---\n";

        auto encBufferFromFile = load_file("secret_plans.enc");

        AESNode decryptor("Decryptor", key, AESMode::Decrypt);
        Packet decryptedPacket = decryptor.process(Packet(encBufferFromFile));

        if (std::holds_alternative<std::shared_ptr<RawBuffer>>(decryptedPacket.core_data))
        {
            auto decBuf = std::get<std::shared_ptr<RawBuffer>>(decryptedPacket.core_data);
            save_file("secret_plans_decrypted.txt", decBuf);

            std::cout << "\nDecrypted Message: \"";
            const unsigned char *data = decBuf->get_data();
            for (size_t i = 0; i < decBuf->get_size(); i++)
            {
                std::cout << data[i];
            }
            std::cout << "\"\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "CRITICAL ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
