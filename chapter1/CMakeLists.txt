cmake_minimum_required(VERSION 3.16)
project(MyGuassian CUDA CXX)

set(SOURCES
    src/src.cu
    src/src_img.cpp
    src/src_global_mem.cu
    src/src_shared_mem.cu
)

add_executable(MyGuassian ${SOURCES})

find_package(OpenCV REQUIRED)
target_include_directories(MyGuassian PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(MyGuassian PRIVATE ${OpenCV_LIBS})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")