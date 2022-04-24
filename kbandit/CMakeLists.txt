cmake_minimum_required (VERSION 3.16)

project ("k-armed-bandit")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# LIBKBANDIT
set(LIBKBANDIT_SOURCES "k-bandit.cpp")
IF(WIN32)
  add_library(kbandit STATIC ${LIBKBANDIT_SOURCES})
else()
  add_library(kbandit SHARED ${LIBKBANDIT_SOURCES})
endif()

target_include_directories(kbandit PUBLIC "include")
set_target_properties(kbandit PROPERTIES
  PUBLIC_HEADER "include/kbandit/k-bandit.h"
)

# Run bandits
add_executable(run-kbandits "run-kbandit.cpp")
target_link_libraries(run-kbandits PRIVATE kbandit fmt::fmt)

if (MSVC) 
  target_link_options(run-kbandits PRIVATE $<$<CONFIG:DEBUG>:/PROFILE>)
endif()

# Tests
add_executable(test-kbandit "test-kbandit.cpp")
target_link_libraries(test-kbandit PRIVATE kbandit Catch2::Catch2)
catch_discover_tests(test-kbandit)

# Install targets
install(
  TARGETS kbandit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include/kbandit/
)