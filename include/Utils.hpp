#include <string_view>
namespace NexusUtils
{
    constexpr int calculate_version(int major, int minor)
    {
        return major * 100 + minor;
    }

    [[nodiscard]] constexpr size_t min_buff_size()
    {
        return 1024;
    }
}