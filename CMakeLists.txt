cmake_minimum_required(VERSION 3.25)

project(lumi_swarm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

find_program(CLANG_FORMAT_EXE NAMES clang-format)

file(GLOB_RECURSE LUMI_FORMAT_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/*.cc"
  "${CMAKE_SOURCE_DIR}/*.cpp"
  "${CMAKE_SOURCE_DIR}/*.cxx"
  "${CMAKE_SOURCE_DIR}/*.h"
  "${CMAKE_SOURCE_DIR}/*.hh"
  "${CMAKE_SOURCE_DIR}/*.hpp"
  "${CMAKE_SOURCE_DIR}/*.hxx"
  "${CMAKE_SOURCE_DIR}/*.cu"
  "${CMAKE_SOURCE_DIR}/*.cuh"
)
list(FILTER LUMI_FORMAT_SOURCES EXCLUDE REGEX "/(build|cmake-build[^/]*)/")
list(FILTER LUMI_FORMAT_SOURCES EXCLUDE REGEX "/\\.git/")

if(CLANG_FORMAT_EXE AND LUMI_FORMAT_SOURCES)
  add_custom_target(format
    COMMAND "${CLANG_FORMAT_EXE}" -i ${LUMI_FORMAT_SOURCES}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Formatting C++ and CUDA sources"
    VERBATIM
  )
elseif(CLANG_FORMAT_EXE)
  add_custom_target(format
    COMMAND "${CMAKE_COMMAND}" -E echo "No C++ or CUDA sources to format"
    VERBATIM
  )
else()
  add_custom_target(format
    COMMAND "${CMAKE_COMMAND}" -E echo "clang-format not found"
    COMMAND "${CMAKE_COMMAND}" -E false
    VERBATIM
  )
endif()

add_executable(lumi_build_smoke
  tests/test_build.cpp
)

add_executable(lumi_cuda_smoke
  tests/test_cuda_smoke.cu
)

add_test(NAME lumi_build_smoke COMMAND lumi_build_smoke)
add_test(NAME lumi_cuda_smoke COMMAND lumi_cuda_smoke)
