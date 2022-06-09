cmake_minimum_required (VERSION 3.16)

project ("mdp")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# LibMDP
set(LIBMDP_SOURCES src/gridworld.cpp)
set(LIBMDP_HEADERS include/mdp/mdp.h include/mdp/gridworld.h include/mdp/graph_mdp.h include/mdp/graph_mdp_policy.h)
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

# Tests :: Gridworld
add_executable("test-gridworld" tests/test-gridworld.cpp)
target_link_libraries("test-gridworld" PRIVATE mdp Catch2::Catch2)
catch_discover_tests("test-gridworld")

# Tests :: GraphMDP
add_executable(test-graphmdp tests/test-graphmdp.cpp)
target_link_libraries(test-graphmdp PRIVATE mdp Catch2::Catch2)
catch_discover_tests(test-graphmdp)