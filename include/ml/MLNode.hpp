#ifndef ML_NODE_HPP
#define ML_NODE_HPP

#include "../BaseNode.hpp"
#include "../RawBuffer.hpp"
#include "../Types.hpp"

#include "MLTypes.hpp"

#include <any>
#include <memory>
#include <stdexcept>

class MLNode : public BaseNode
{
protected:
    NodeStatus status;
    std::string error;

    bool is_buffer(const Packet &packet) const
    {
        return std::holds_alternative<std::shared_ptr<RawBuffer>>(packet.core_data);
    }

    std::shared_ptr<RawBuffer> get_buffer(const Packet &packet) const
    {
        if (!is_buffer(packet))
        {
            throw std::runtime_error("Packet does not contain a RawBuffer.");
        }

        return std::get<std::shared_ptr<RawBuffer>>(packet.core_data);
    }

    template <typename T>
    T get_metadata(const Packet &packet) const
    {
        try
        {
            return std::any_cast<T>(packet.meta_data);
        }
        catch (const std::bad_any_cast &e)
        {
            throw std::runtime_error("Failed to cast metadata to the requested type: " + std::string(e.what()));
        }
    }

    template <typename T>
    bool has_metadata(const Packet &packet) const
    {
        return packet.meta_data.type() == typeid(T);
    }
    
    template <typename MetaT>
    Packet create_packet(std::shared_ptr<RawBuffer> buffer, const MetaT &metadata) const
    {
        Packet p(buffer);
        p.meta_data = metadata;
        return p;
    }
    
    void set_error(const std::string &err)
    {
        status = NodeStatus::Error;
        error = err;
        std::cerr << "[" << node_name << "] Error: " << err << "\n";
    }

public:
    MLNode(std::string name) : BaseNode(std::move(name)), status(NodeStatus::Ready) {}
    
    virtual ~MLNode() = default;
    
    std::string get_last_error() const
    {
        return error;
    }
};

#endif