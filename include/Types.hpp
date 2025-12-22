#ifndef TYPES_HPP
#define TYPES_HPP

#include <any>
#include <string>
#include <memory>
#include <variant>
#include <vector>

#include "RawBuffer.hpp"

using DataVariant = std::variant<int, double, std::string, std::shared_ptr<RawBuffer>>;

enum class NodeStatus
{
    Ready,
    Processing,
    Error,
    Finished
};

struct Packet
{
    DataVariant core_data;
    std::any meta_data;

    Packet() : core_data(0) {}

    Packet(DataVariant data) : core_data(std::move(data)) {}

    Packet(std::shared_ptr<RawBuffer> buf)
        : core_data(std::move(buf)) {}
};

#endif 