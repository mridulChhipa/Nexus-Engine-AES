#include <iostream>
#include <memory>
#include <string>

#include "../include/NexusEngine.hpp"
#include "../include/ComputeNode.hpp"
#include "../include/LoggerNode.hpp"
#include "../include/RawBuffer.hpp"

int main()
{
    NexusEngine engine;

    auto compute_ptr = std::make_unique<ComputeNode>("Multiplier");
    auto logger_ptr = std::make_unique<LoggerNode>("HistoryTracker");

    engine << std::move(compute_ptr) << std::move(logger_ptr);

    Packet my_packet(4892);
    engine(std::move(my_packet));

    Packet next_packet("Nexus");
    engine(std::move(next_packet));

    RawBuffer buffer(NexusUtils::min_buff_size() / 128);
    unsigned char rawData[] = {0x10, 0x20, 0x30, 0x40};
    buffer.copy(rawData, NexusUtils::min_buff_size() / 128);
    buffer.reverse();

    const unsigned char *data_ptr = buffer.get_data();
    for (size_t i = 0; i < NexusUtils::min_buff_size() / 128; i++)
    {
        std::cout << "0x" << std::hex << std::uppercase
                  << static_cast<int>(*(data_ptr + i)) << " ";
    }
    std::cout << "\n";

    return 0;
}