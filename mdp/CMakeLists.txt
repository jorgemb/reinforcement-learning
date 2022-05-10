cmake_minimum_required (VERSION 3.16)

project ("mdp")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Run MDP
add_executable("run-mdp" run_mdp.cpp)
target_link_libraries("run-mdp" PRIVATE fmt::fmt Boost::graph)

# Gridworld
set(LIBGRIDWORLD_SOURCES "src/gridworld.cpp")
set(LIBGRIDWORLD_HEADERS "include/mdp/gridworld.h")
add_library(gridworld STATIC ${LIBGRIDWORLD_SOURCES} ${LIBGRIDWORLD_HEADERS})
target_include_directories(gridworld PUBLIC include)
set_target_properties(gridworld PROPERTIES
    PUBLIC_HEADER "${LIBGRIDWORLD_HEADERS}"
)

add_executable("test-gridworld" test-gridworld.cpp)
target_link_libraries("test-gridworld" PRIVATE gridworld Catch2::Catch2)
catch_discover_tests("test-gridworld")