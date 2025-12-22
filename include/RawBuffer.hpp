#ifndef RAWBUFFER_HPP
#define RAWBUFFER_HPP

#include <algorithm>
#include <cstring>

class RawBuffer
{
    unsigned char *data;

    /*
    For lengths and sizes representing bytes, etc using size_t is preffered
    sizeof() return type is size_t
    */
    size_t size;

public:
    RawBuffer(size_t s) : size(s)
    {
        data = new unsigned char[size];
    }

    ~RawBuffer()
    {
        delete[] data;
    }

    // Here the const ensures that the copy method cannot modify source
    void copy(const unsigned char *source, size_t len)
    {
        size_t n = std::min(len, size);
        for (size_t i = 0; i < n; i++)
        {
            *(data + i) = *(source + i);
        }
    }

    void reverse()
    {
        unsigned char *start = data;
        unsigned char *end = data + size - 1;

        while (start < end)
        {
            std::swap(*start, *end);
            start++;
            end--;
        }
    }

    /*
    First const ensures that no one can modify the data returned as it is
    a pointer and thus the values stored in the pointer can be modified which
    is prevented using the first const
    The second const (i.e the one after get_data()) ensures that the get_data method
    can't modify any of the objects members
    */
    const unsigned char *get_data() const
    {
        return data;
    }

    size_t get_size() const
    {
        return size;
    }
};

#endif