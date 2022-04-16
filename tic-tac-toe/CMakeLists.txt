cmake_minimum_required(VERSION 3.8)

project("tic-tac-toe" VERSION 0.1)

# Executables
add_executable(run-tictactoe "run.cpp" "tictactoe.h" "test-tictactoe.cpp")

# Tests
add_executable(test-tictactoe "test-tictactoe.cpp")
# target_include_directories(test-tictactoe PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(test-tictactoe PRIVATE Boost::boost Boost::test)
