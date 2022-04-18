cmake_minimum_required (VERSION 3.16)

project ("k-armed-bandit")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

add_library(kbandit STATIC
	"k-bandit.cpp"
)
target_include_directories(kbandit PUBLIC "include")

# Tests
add_executable(test-kbandit "test-kbandit.cpp")
target_link_libraries(test-kbandit PRIVATE kbandit Catch2::Catch2)
catch_discover_tests(test-kbandit)
