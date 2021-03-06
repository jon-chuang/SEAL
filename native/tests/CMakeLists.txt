# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(SEALTest LANGUAGES CXX)

# set(CMAKE_CXX_COMPILER /usr/local/bin/syclcc-clang)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__ --std=c++14")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 -gencode=arch=compute_61,code=sm_61 -rdc=true --cudart shared")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lnvToolsExt -L/usr/local/cuda-10.0/targets/x86_64-linux/lib")

# Executable will be in ../bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SEALTest_SOURCE_DIR}/../bin)

add_executable(sealtest seal/testrunner.cpp)
# set_target_properties(sealtest PROPERTIES CXX_STANDARD 14)

# Import Microsoft SEAL
find_package(SEAL 3.4.5 EXACT REQUIRED
    # Providing a path so this can be built without installing Microsoft SEAL
    PATHS ${SEALTest_SOURCE_DIR}/../src/cmake
)

# Add source files
add_subdirectory(seal)

# Only build GTest
option(BUILD_GMOCK OFF)
option(INSTALL_GTEST OFF)
mark_as_advanced(BUILD_GMOCK INSTALL_GTEST)

# Add GTest
set(GTEST_DIR ${SEALTest_SOURCE_DIR}/thirdparty/googletest)
if(NOT EXISTS ${GTEST_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "Could not find `${GTEST_DIR}/CMakeLists.txt`. Run `git submodule update --init` and retry.")
endif()

add_subdirectory(${GTEST_DIR})

# Link Microsoft SEAL and GTest
target_link_libraries(sealtest SEAL::seal gtest)
