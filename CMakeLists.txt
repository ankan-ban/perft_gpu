cmake_minimum_required(VERSION 3.18)
project(perft_gpu LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CXX_STANDARD 17)

add_executable(perft_gpu
    perft.cu
    util.cpp
    GlobalVars.cpp
    Magics.cpp
)

target_include_directories(perft_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# No separable compilation needed (no CDP)
set_target_properties(perft_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
