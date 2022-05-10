#include "mdp/gridworld.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

using namespace Catch::literals;
using rl::mdp::Gridworld;

TEST_CASE("Basic gridworld", "[gridworld]"){
    size_t rows = 5, columns = 5;
    Gridworld g(rows, columns, 42);
    using Action = Gridworld::Action;
    using State = Gridworld::State;

    std::array<Action, 4> available_actions {Action::LEFT, Action::RIGHT, Action::UP, Action::DOWN};

    SECTION("Standard properties") {
        REQUIRE(g.get_rows() == rows);
        REQUIRE(g.get_columns() == columns);
    }

    SECTION("Default actions"){
        // All actions should be deterministic
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < columns; ++j) {
                for(auto action: available_actions) {
                    DYNAMIC_SECTION("Action for " << i << "," << j << " - " << action) {
                        auto state = State{i, j};
                        auto [new_state, reward] = g.get_transition(state, action);
                        REQUIRE(reward == 0.0_a);

                        // Determine the real new state
                        switch(action){
                                case Action::LEFT: REQUIRE(new_state == State{i, j==0 ? j : j-1}); break;
                                case Action::RIGHT: REQUIRE(new_state == State{i, j>=columns ? j : j+1}); break;
                                case Action::UP: REQUIRE(new_state == State{i==0 ? i : i-1,  j}); break;
                                case Action::DOWN: REQUIRE(new_state == State{i>=rows ? i : i+1, j}); break;
                        }
                    }
                }
            }
        }
    }

    SECTION("Deterministic transitions") {
        // Everything from 0,1 leads to 4,1 and +10 reward
        g.add_transition(State{0, 1}, Action::LEFT, State{4, 1}, 10., 1.0);
        g.add_transition(State{0, 1}, Action::RIGHT, State{4, 1}, 10., 1.0);
        g.add_transition(State{0, 1}, Action::UP, State{4, 1}, 10., 1.0);
        g.add_transition(State{0, 1}, Action::DOWN, State{4, 1}, 10., 1.0);
        for (auto action: available_actions) {
            DYNAMIC_SECTION("Check transitions (0,1) -> (4,1) :: " << action) {
                auto transition = g.get_transition(State{0, 1}, action);
                REQUIRE(transition.first == State{4, 1});
                REQUIRE(transition.second == 10.0_a);
            }
        }
    }

    SECTION("Non-deterministic transitions"){
        // Create non-deterministic transition
        State from{0, 0};
        Action action{Action::LEFT};
        State newA{1,0}, newB{2, 0};
        g.add_transition(from, action, newA, 5.0, 0.5);
        g.add_transition(from, action, newB, 10.0, 0.5);

        // Do many tests to see if probabilities match the provided ones
        size_t total_tests = 10000;
        size_t times_A = 0, times_B = 0;
        for (int i = 0; i < total_tests; ++i) {
            auto transition = g.get_transition(from, action);
            if(transition.first == newA) {
                ++times_A;
            }
            else {
                ++times_B;
            }
        }

        REQUIRE(static_cast<double>(times_A) / total_tests == Approx(0.5).margin(0.01));
        REQUIRE(static_cast<double>(times_B) / total_tests == Approx(0.5).margin(0.01));
    }

    SECTION("MDP values"){
        SECTION("Expected reward") {
            // Expected reward for default transition
            REQUIRE(g.expected_reward(State{0, 0}, Action::LEFT) == 0.0_a);

            // Expected reward for added transition
            g.add_transition(State{0, 0}, Action::LEFT, State{1,1}, 10., 1.0);
            REQUIRE(g.expected_reward(State{0, 0}, Action::LEFT) == 10.0_a);

            // Expected reward for non-deterministic transition
            g.add_transition(State{0, 0}, Action::LEFT, State{2, 2}, 100, 3.0);
            double expected = (10.0 * 0.25) + (100.0 * 0.75);
            REQUIRE(g.expected_reward(State{0, 0}, Action::LEFT) == Approx(expected));
        }
    }
}