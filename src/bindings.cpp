#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatic conversion for std::vector, std::string
#include <pybind11/functional.h>

#include "../include/RawBuffer.hpp"
#include "../include/AESNode.hpp"
#include "../include/Types.hpp"

namespace py = pybind11;

std::shared_ptr<RawBuffer> aes_wrapper(AESNode &node, std::shared_ptr<RawBuffer> inputBuffer)
{
    Packet p(inputBuffer);
    Packet result = node.process(p);
    if (std::holds_alternative<std::shared_ptr<RawBuffer>>(result.core_data))
    {
        return std::get<std::shared_ptr<RawBuffer>>(result.core_data);
    }

    throw std::runtime_error("AES processing failed or returned non-buffer data.");
}

PYBIND11_MODULE(nexus, m)
{
    m.doc() = "Nexus Engine: High-Performance C++ Crypto Backend";

    py::class_<RawBuffer, std::shared_ptr<RawBuffer>>(m, "RawBuffer", py::buffer_protocol())
        .def(py::init<size_t>())
        .def("get_size", &RawBuffer::get_size)
        // Enable Python to set data: buffer.set_bytes(b"hello")
        .def("set_bytes", [](RawBuffer &self, py::bytes data)
             {
             std::string s = data; // Copy from Python bytes to C++ string
             self.copy(reinterpret_cast<const unsigned char*>(s.c_str()), s.size()); })
        // Enable Python to read data: print(buffer.to_bytes())
        .def("to_bytes", [](RawBuffer &self)
             { return py::bytes(reinterpret_cast<const char *>(self.get_data()), self.get_size()); })
        // MAGIC: Allow Numpy/Python to view this memory directly!
        .def_buffer([](RawBuffer &m) -> py::buffer_info
                    { return py::buffer_info(
                          const_cast<unsigned char *>(m.get_data()),      // Pointer to buffer
                          sizeof(unsigned char),                          // Size of one element
                          py::format_descriptor<unsigned char>::format(), // Python format code ('B')
                          1,                                              // Number of dimensions
                          {m.get_size()},                                 // Buffer dimensions
                          {sizeof(unsigned char)}                         // Strides (in bytes)
                      ); });

    py::enum_<AESMode>(m, "AESMode")
        .value("Encrypt", AESMode::Encrypt)
        .value("Decrypt", AESMode::Decrypt)
        .export_values();

    py::class_<AESNode>(m, "AESNode")
        .def(py::init<std::string, std::vector<uint8_t>, AESMode>())

        .def("process", &aes_wrapper, "Process a RawBuffer through AES");
}