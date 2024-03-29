cmake_minimum_required (VERSION 3.16)

project ("mdp")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# LibMDP
set(LIBMDP_SOURCES
        src/gridworld.cpp
        src/actions.cpp
        src/agents.cpp)
set(LIBMDP_HEADERS
        include/mdp/mdp.h
        include/mdp/gridworld.h
        include/mdp/graph.h
        include/mdp/graph_policy.h
        include/mdp/actions.h
        include/mdp/agents.h)
IF(WIN32)
    add_library(mdp STATIC ${LIBMDP_SOURCES} ${LIBMDP_HEADERS})
else()
    add_library(mdp SHARED ${LIBMDP_SOURCES} ${LIBMDP_HEADERS})
endif()
target_include_directories(mdp PUBLIC include)
target_link_libraries(mdp PUBLIC Boost::graph)
set_target_properties(mdp PROPERTIES
        PUBLIC_HEADER "${LIBMDP_HEADERS}"
        )

install(
        TARGETS mdp
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/mdp/
)

# Run MDP
add_executable("run-mdp" run_mdp.cpp)
target_link_libraries("run-mdp" PRIVATE mdp fmt::fmt Boost::graph)

# Run Gridworld Agents
add_executable("run-gridworld-agents" run_gridworld_agents.cpp)
target_link_libraries("run-gridworld-agents" PRIVATE mdp fmt::fmt Boost::boost)
#target_link_options("run-gridworld-agents" PRIVATE /PROFILE) # Profile with VisualStudio

# Tests :: Gridworld
add_executable("test-gridworld" tests/test-gridworld.cpp)
target_link_libraries("test-gridworld" PRIVATE mdp Catch2::Catch2WithMain)
catch_discover_tests("test-gridworld")

# Tests :: GraphMDP
add_executable(test-graphmdp tests/test-graphmdp.cpp)
target_link_libraries(test-graphmdp PRIVATE mdp Catch2::Catch2WithMain)
catch_discover_tests(test-graphmdp)

add_executable(test-graphmdp-greedy tests/test-graphmdp-greedy.cpp)
target_link_libraries(test-graphmdp-greedy PRIVATE mdp Catch2::Catch2WithMain)
catch_discover_tests(test-graphmdp-greedy)

# Tests :: Agents
add_executable(test-agents tests/test-agents.cpp)
target_link_libraries(test-agents PRIVATE mdp Catch2::Catch2WithMain sciplot::sciplot)
catch_discover_tests(test-agents)