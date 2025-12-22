#ifndef BASE_NODE_HPP
#define BASE_NODE_HPP

#include <cstdint>
#include <iostream>
#include "Types.hpp"

class BaseNode
{
protected:
    std::string node_name;

    uint32_t id;
    static inline uint32_t num_nodes = 0;

public:
    BaseNode(std::string name) : node_name(std::move(name)), id(num_nodes++)
    {
        std::cout << "Node [" << node_name << "] created with ID: [" << id << "]\n";
    }

    virtual ~BaseNode() = default;

    // [[nodiscard]] ensures the return value (Packet) isn't ignored
    [[nodiscard]] virtual Packet process(const Packet &input) = 0;

    virtual NodeStatus get_status() const = 0;
    uint32_t get_id()
    {
        return id;
    }
};

#endif