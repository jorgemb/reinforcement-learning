# CMakeList.txt : CMake project for reinforcement-learning, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project ("reinforcement-learning")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# find_package(Catch2 REQUIRED)
add_subdirectory(lib/Catch2)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/lib/Catch2/extras ${CMAKE_MODULE_PATH})

# Testing library
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
include(Catch)

# Add other projects
add_subdirectory(tic-tac-toe/)


