cmake_minimum_required(VERSION 3.21)
project(CUDA_motecarlo)

set(CMAKE_CXX_STANDARD 14)

add_executable(CUDA_motecarlo
        monte_carlo_cpp/main.cpp)
