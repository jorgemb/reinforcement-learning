cmake_minimum_required (VERSION 3.16)

# Add vcpkg 
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

project ("reinforcement-learning")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Install prefix
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install/)

# Add dependencies
find_package(Catch2 REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

# Testing library
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
include(Catch)

# Add other projects
add_subdirectory(kbandit/)
