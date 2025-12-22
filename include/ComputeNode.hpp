#include <functional>
#include <stdexcept>
#include "BaseNode.hpp"

template <typename T>
struct Calculator
{
    static T add(T a, T b) { return a + b; }
};

template <>
struct Calculator<std::string>
{
    static std::string add(std::string a, std::string b) { return a + " " + b; }
};

class ComputeNode : public BaseNode
{
    std::function<DataVariant(const DataVariant &)> operation;
    NodeStatus curr_status;

public:
    ComputeNode(std::string name) : BaseNode(std::move(name)), curr_status(NodeStatus::Ready)
    {
        operation = [](const DataVariant &input) -> DataVariant
        {
            return std::visit([](auto &&inp) -> DataVariant
                              {
                using T = std::decay_t<decltype(inp)>;
                if constexpr (std::is_arithmetic_v<T>)
                {
                    return inp * 2;
                }
                else
                {
                    return inp + " operation";
                } }, input);
        };
    }

    [[nodiscard]] Packet process(const Packet &packet) override
    {
        curr_status = NodeStatus::Processing;

        try
        {
            std::cout << "Processing Node [" << id << "] having type [" << packet.core_data.index() << "]\n";
            DataVariant res = operation(packet.core_data);
            curr_status = NodeStatus::Finished;

            return Packet(res);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            throw;
        }
    }

    NodeStatus get_status() const override
    {
        return curr_status;
    }
};
