#include "BaseNode.hpp"
#include <deque>
#include <list>

class LoggerNode : public BaseNode
{
    const size_t recent_limit = 5;
    std::deque<std::string> logs;
    std::list<std::string> history;

public:
    LoggerNode(std::string name) : BaseNode(std::move(name)) {}

    [[nodiscard]] Packet process(const Packet &packet) override
    {
        std::string entry = std::visit([](auto &&data) -> std::string
                                       { 
            using T = std::decay_t<decltype(data)>; 
            if constexpr (std::is_same_v<T, std::string>) return data;
            else if constexpr (std::is_same_v<T, std::shared_ptr<RawBuffer>>) 
                return "[RawBuffer size=" + std::to_string(data ? data->get_size() : 0) + "]";
            else return std::to_string(data); }, packet.core_data);

        history.emplace_back("Processed: " + entry);
        logs.emplace_back(entry);

        if (logs.size() > recent_limit)
            logs.pop_front();

        std::cout << "[LOG] " << entry << " added to history \n";
        return packet;
    }

    NodeStatus get_status() const override
    {
        return NodeStatus::Ready;
    }
};
