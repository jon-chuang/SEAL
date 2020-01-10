# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(SEALExamples VERSION 3.4.5 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 -gencode=arch=compute_61,code=sm_61 -rdc=true --cudart shared")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lnvToolsExt -L/usr/local/cuda-10.2/targets/x86_64-linux/lib")

# Executable will be in ../bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SEALExamples_SOURCE_DIR}/../bin)

add_executable(sealexamples examples.cpp)
target_sources(sealexamples
    PRIVATE
        1_bfv_basics.cpp
        2_encoders.cpp
        3_levels.cpp
        4_ckks_basics.cpp
        5_rotation.cpp
        6_performance.cpp
        7_cuda_test.cpp
)

# Import Microsoft SEAL
find_package(SEAL 3.4.5 EXACT REQUIRED
    # Providing a path so this can be built without installing Microsoft SEAL
    PATHS ${SEALExamples_SOURCE_DIR}/../src/cmake
)

# Link Microsoft SEAL
target_link_libraries(sealexamples SEAL::seal)
