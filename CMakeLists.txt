cmake_minimum_required(VERSION 3.16)
project(learn-cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES "75") # gtx 1650 turing arch sm75 ; rtx 3090 ampere arch sm80 86
# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL") # \MD

add_subdirectory(chapter1)