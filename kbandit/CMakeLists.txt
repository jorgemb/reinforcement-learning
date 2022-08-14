cmake_minimum_required (VERSION 3.16)

project ("k-armed-bandit")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# LIBKBANDIT
set(LIBKBANDIT_SOURCES
        "src/k-bandit.cpp"
        "src/k-bandit-agent.cpp"
)
set(LIBKBANDIT_HEADERS
  "include/kbandit/k-bandit.h"
  "include/kbandit/k-bandit-agent.h"
)


IF(WIN32)
  add_library(kbandit STATIC ${LIBKBANDIT_SOURCES})
else()
  add_library(kbandit SHARED ${LIBKBANDIT_SOURCES})
endif()

target_include_directories(kbandit PUBLIC "include")
set_target_properties(kbandit PROPERTIES
  PUBLIC_HEADER "${LIBKBANDIT_HEADERS}"
)

# Run bandits
add_executable(run-kbandits "run-kbandit.cpp")
target_link_libraries(run-kbandits PRIVATE kbandit fmt::fmt sciplot::sciplot)

# .. add profiling information
if (MSVC) 
  target_link_options(run-kbandits PRIVATE $<$<CONFIG:DEBUG>:/PROFILE>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

# Tests
add_executable(test-kbandit "tests/test-kbandit.cpp")
target_link_libraries(test-kbandit PRIVATE kbandit Catch2::Catch2WithMain)
catch_discover_tests(test-kbandit)

# Install targets
install(
  TARGETS kbandit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include/kbandit/
)