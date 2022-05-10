cmake_minimum_required (VERSION 3.16)

project ("mdp")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Run MDP tests
add_executable("run-mdp" run_mdp.cpp)
target_link_libraries("run-mdp" PRIVATE fmt::fmt Boost::graph)