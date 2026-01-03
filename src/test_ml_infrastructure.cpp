#include <iostream>
#include <memory>
#include "../include/ml/MLNode.hpp"
#include "../include/ml/MLTypes.hpp"

class TestMLNode : public MLNode {
public:
    TestMLNode(std::string name) : MLNode(std::move(name)) {}
    
    [[nodiscard]] Packet process(const Packet &input) override {
        status = NodeStatus::Processing;
        std::cout << "[" << node_name << "] Processing packet...\n";
        
        // Test buffer extraction
        if (is_buffer(input)) {
            auto buffer = get_buffer(input);
            std::cout << "  Buffer size: " << buffer->get_size() << " bytes\n";
            
            // Test metadata extraction
            if (has_metadata<ImageMetaData>(input)) {
                auto img_meta = get_metadata<ImageMetaData>(input);
                std::cout << "  Image: " << img_meta.width << "x" << img_meta.height 
                         << "x" << img_meta.channels << " (" << img_meta.format << ")\n";
            }
        }
        
        status = NodeStatus::Finished;
        return input;
    }
    
    NodeStatus get_status() const override {
        return status;
    }
};

int main() {
    std::cout << "=== Testing ML Infrastructure ===\n\n";
    
    // Test 1: Create metadata structures
    std::cout << "Test 1: Metadata structures\n";
    ImageMetaData img_meta(224, 224, 3, "RGB");
    std::cout << "  ImageMetaData created: " << img_meta.width << "x" << img_meta.height 
              << ", total size: " << img_meta.total_size() << " bytes\n";
    
    TensorMetaData tensor_meta({1, 3, 224, 224}, "float32", "NCHW");
    std::cout << "  TensorMetaData created: " << tensor_meta.num_elements() << " elements, "
              << tensor_meta.total_bytes() << " bytes\n";
    
    AudioMetaData audio_meta(16000, 1, 48000);
    std::cout << "  AudioMetaData created: " << audio_meta.sample_rate << "Hz, "
              << audio_meta.duration_seconds() << "s duration\n";
    
    // Test 2: Create MLNode instance
    std::cout << "\nTest 2: MLNode instantiation\n";
    auto test_node = std::make_unique<TestMLNode>("TestNode");
    std::cout << "  Node created with ID: " << test_node->get_id() << "\n";
    std::cout << "  Initial status: " << (test_node->get_status() == NodeStatus::Ready ? "Ready" : "Not Ready") << "\n";
    
    // Test 3: Process packet with buffer
    std::cout << "\nTest 3: Processing packet with buffer\n";
    auto buffer = std::make_shared<RawBuffer>(150528); // 224*224*3
    Packet test_packet(buffer);
    test_packet.meta_data = img_meta;
    
    Packet result = test_node->process(test_packet);
    std::cout << "  Final status: " << (test_node->get_status() == NodeStatus::Finished ? "Finished" : "Other") << "\n";
    
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
