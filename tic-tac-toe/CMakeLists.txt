cmake_minimum_required(VERSION 3.8)

project("tic-tac-toe" VERSION 0.1)

# Executables
add_executable(run-tictactoe "run.cpp" "tictactoe.h" "test-tictactoe.cpp")

# Tests
# add_executable(test-tictactoe "test-tictactoe.cpp")
# target_link_libraries(test-tictactoe PRIVATE Catch2::Catch2WithMain)
