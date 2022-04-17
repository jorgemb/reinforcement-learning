# CMakeList.txt : CMake project for reinforcement-learning, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project ("reinforcement-learning")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Conan - Dependency management
# if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
#    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
#    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
#                   "${CMAKE_BINARY_DIR}/conan.cmake")
# endif()

# include(${CMAKE_BINARY_DIR}/conan.cmake)

# foreach(TYPE ${CMAKE_CONFIGURATION_TYPES})
#     conan_cmake_autodetect(settings BUILD_TYPE ${TYPE})
#     conan_cmake_install(PATH_OR_REFERENCE ${CMAKE_SOURCE_DIR}
#                         BUILD missing
#                         REMOTE conancenter
#                         SETTINGS ${settings})
# endforeach()

# Installing dependencies
# DEBUG
# conan install . --install-folder build --build=missing -s build_type=Debug
# RELEASE
# conan install . --install-folder build --build=missing -s build_type=Release

# Dependencies
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
# include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
# conan_basic_setup()

find_package(Boost 1.78.0 REQUIRED)
find_package(Catch2 REQUIRED)

# Testing library
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
include(Catch)

# Add other projects
add_subdirectory(tic-tac-toe/)


