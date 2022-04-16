# CMakeList.txt : CMake project for reinforcement-learning, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project ("reinforcement-learning")

set(CMAKE_CXX_STANDARD 20)
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

 include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
 conan_basic_setup()

# Testing library
find_package(Boost 1.78.0 REQUIRED)

# Add other projects
add_subdirectory(tic-tac-toe/)


