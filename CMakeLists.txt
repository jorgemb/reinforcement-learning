cmake_minimum_required (VERSION 3.16)

# Add vcpkg 
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

project ("reinforcement-learning")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Add dependencies
find_package(Catch2 REQUIRED)

# Testing library
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
include(Catch)

# Add other projects
add_subdirectory(kbandit/)

