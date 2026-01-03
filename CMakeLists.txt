cmake_minimum_required(VERSION 3.20)
project(NexusEngine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(NexusEngine src/main.cpp)

if(MSVC)
    target_compile_options(NexusEngine PRIVATE /W4)
else()
    target_compile_options(NexusEngine PRIVATE -Wall -Wextra)
endif()