cmake_minimum_required (VERSION 3.8)

# Add vcpkg 
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

project ("reinforcement-learning")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

find_package(Catch2 REQUIRED)
# add_subdirectory(lib/Catch2)
# set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/lib/Catch2/extras ${CMAKE_MODULE_PATH})

# Testing library
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
include(Catch)

# Add other projects
add_subdirectory(tic-tac-toe/)


