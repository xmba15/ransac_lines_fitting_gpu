cmake_minimum_required(VERSION 3.10)

add_executable(2d_lines_fitting_app
  ${CMAKE_CURRENT_LIST_DIR}/2DLinesFittingApp.cpp
)

target_link_libraries(2d_lines_fitting_app
  PUBLIC
    ${LIBRARY_NAME}
)


add_executable(3d_lines_fitting_app
  ${CMAKE_CURRENT_LIST_DIR}/3DLinesFittingApp.cpp
)

target_link_libraries(3d_lines_fitting_app
  PUBLIC
    ${LIBRARY_NAME}
)
