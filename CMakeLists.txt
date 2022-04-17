# CMakeList.txt : CMake project for reinforcement-learning, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")
  
project ("reinforcement-learning")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

find_package(Boost 1.78.0 REQUIRED)
find_package(Catch2 REQUIRED)

# Testing library
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
include(Catch)

# Add other projects
add_subdirectory(tic-tac-toe/)


