cmake_minimum_required(VERSION 3.10)

project(ransac_lines_fitting_gpu LANGUAGES CXX)

set(LIBRARY_NAME ${PROJECT_NAME})

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA not found")
  return()
endif()

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

add_compile_options(
  "$<$<CONFIG:DEBUG>:-DDEBUG>"
)

add_subdirectory(src)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)
