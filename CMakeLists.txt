cmake_minimum_required(VERSION 3.20)
project(NexusEngine)

# 1. Set C++ Standard to C++20
# This is required for std::span, concepts, and modern features used in the project
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Include the 'include' directory
# This allows your code to use #include "BaseNode.hpp" instead of "../include/BaseNode.hpp"
include_directories(include)

# 3. Define the Executable
# We only compile main.cpp because all other implementations are in the .hpp files
add_executable(NexusEngine src/main.cpp)

# 4. Enable Compiler Warnings (Best Practice)
if(MSVC)
    target_compile_options(NexusEngine PRIVATE /W4)
else()
    target_compile_options(NexusEngine PRIVATE -Wall -Wextra)
endif()