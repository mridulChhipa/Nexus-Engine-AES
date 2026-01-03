#ifndef ML_TYPES_HPP
#define ML_TYPES_HPP

#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

struct ImageMetaData
{
    int width;
    int height;
    int channels;
    std::string format;

    ImageMetaData() : width(0), height(0), channels(0), format("RGB") {}
    ImageMetaData(int w, int h, int c, const std::string &fmt)
        : width(w), height(h), channels(c), format(std::move(fmt)) {}

    size_t total_size() const
    {
        return width * height * channels;
    }
};

struct TensorMetaData
{
    std::vector<uint64_t> shape;
    std::string dtype;
    std::string layout;

    TensorMetaData() : dtype("float32"), layout("NCHW") {}
    TensorMetaData(const std::vector<uint64_t> &s, const std::string &dt, const std::string &lay)
        : shape(std::move(s)), dtype(dt), layout(lay) {}

    size_t num_elements() const
    {
        size_t count = 1;
        for (uint64_t dim : shape)
        {
            count *= dim;
        }
        return count;
    }

    size_t element_size() const
    {
        if (dtype == "float32" || dtype == "int32")
            return 4;
        else if (dtype == "float64" || dtype == "int64")
            return 8;
        else if (dtype == "uint8")
            return 1;

        throw std::runtime_error("Unsupported data type: " + dtype);
    }

    size_t total_bytes() const
    {
        return num_elements() * element_size();
    }
};

struct AudioMetaData
{
    int sample_rate;
    int channels;
    int samples;
    std::string format;

    AudioMetaData() : sample_rate(16000), channels(1), samples(0), format("float32") {}
    AudioMetaData(int sr, int ch, int ns, std::string dt = "float32")
        : sample_rate(sr), channels(ch), samples(ns), format(std::move(dt)) {}

    double duration_seconds() const
    {
        return static_cast<double>(samples) / sample_rate;
    }
};

struct TextMetaData
{
    std::string enc;
    std::string tokenizer;
    size_t length;

    TextMetaData() : enc("utf-8"), tokenizer("word"), length(0) {}
};

#endif