﻿cmake_minimum_required (VERSION 3.16)

# Add vcpkg 
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

project ("reinforcement-learning")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Install prefix
#set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install/ CACHE STRING "Install path prefix")

# Add dependencies
find_package(Catch2 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(xtl CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(sciplot CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)
find_package(SFML COMPONENTS system window graphics CONFIG REQUIRED)

if(NOT WIN32)
  find_package(TBB REQUIRED)
endif()

# Testing library
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
include(Catch)

# Add other projects
add_subdirectory(kbandit/)
add_subdirectory(mdp/)
add_subdirectory(draw/)