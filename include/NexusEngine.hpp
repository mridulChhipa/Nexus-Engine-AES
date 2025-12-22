#include <memory>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

#include "BaseNode.hpp"
#include "Utils.hpp"

class NexusEngine
{
    std::vector<std::unique_ptr<BaseNode>> nodes;
    std::unordered_map<std::string, BaseNode *> node_map;
    std::map<unsigned int, BaseNode *> ids;
    std::priority_queue<int> error_logs;

public:
    NexusEngine()
    {
        std::cout << "Nexus Engine v" << NexusUtils::calculate_version(1, 0) << "initialised \n";
    }

    NexusEngine &operator<<(std::unique_ptr<BaseNode> node)
    {
        if (!node)
            return *this;

        ids[node->get_id()] = node.get();

        nodes.push_back(std::move(node));
        return *this;
    }

    void operator()(Packet initial_packet)
    {
        std::cout << "Nexus Engine has started \n";
        Packet curr = std::move(initial_packet);

        for (auto &node : nodes)
        {
            curr = node->process(curr);
            std::visit([](auto &&inp)
                       { std::cout << "Current value: " << inp << "\n"; }, curr.core_data);
        }
    }

    void add(std::unique_ptr<BaseNode> node)
    {
        node_map[node->get_status() == NodeStatus::Ready ? "Ready" : "Busy"] = node.get();
        nodes.push_back(std::move(node));
    }

    void run(Packet initial_packet)
    {
        Packet curr = std::move(initial_packet);
        for (const auto &node : nodes)
        {
            curr = node->process(curr);
            std::visit([](auto &&inp)
                       { std::cout << "Current value: " << inp << "\n"; }, curr.core_data);
        }
    }
};
