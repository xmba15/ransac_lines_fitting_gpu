cmake_minimum_required(VERSION 3.10)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/LinesFittingInterface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SACModel.cpp
)

add_library(lines_fitting_cuda
  SHARED
    ${SOURCE_FILES}
)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/LinesFittingInterface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SACModel.cpp
    PROPERTIES LANGUAGE CUDA
)

target_include_directories(lines_fitting_cuda
  PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_compile_features(lines_fitting_cuda PRIVATE cxx_std_14)
target_compile_options(lines_fitting_cuda
  PRIVATE
     $<$<COMPILE_LANGUAGE:CUDA>:-expt-extended-lambda>
)
