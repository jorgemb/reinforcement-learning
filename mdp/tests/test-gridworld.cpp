#include "mdp/gridworld.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <vector>
#include <limits>

using namespace Catch::literals;
using rl::mdp::Gridworld;
using rl::mdp::AvailableGridworldActions;

TEST_CASE("Basic Gridworld", "[gridworld]"){
    size_t rows = 5, columns = 5;
    Gridworld g(rows, columns, 42);
    using Action = Gridworld::Action;
    using State = Gridworld::State;

    SECTION("Standard properties") {
        REQUIRE(g.get_rows() == rows);
        REQUIRE(g.get_columns() == columns);
    }

    SECTION("Default actions"){
        // All actions should be deterministic
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < columns; ++j) {
                for(auto action: AvailableGridworldActions) {
                    DYNAMIC_SECTION("Action for " << i << "," << j << " - " << action) {
                        auto state = State{i, j};
                        auto [new_state, reward] = g.get_transition(state, action);

                        // Check if in the edge, which would mean that the returned state is the same
                        if(new_state == state) REQUIRE(reward == -1.0_a);
                        else REQUIRE(reward == 0.0_a);

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
        for (auto action: AvailableGridworldActions) {
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
            g.add_transition(State{0, 0}, Action::LEFT, State{3, 3}, 50, 6.0);
            double expected = (10.0 * 0.1) + (100.0 * 0.3) + (50 * 0.6);
            REQUIRE(g.expected_reward(State{0, 0}, Action::LEFT) == Approx(expected));
        }

        SECTION("State-transition probability"){
            // Expected reward for known transition
            REQUIRE(g.state_transition_probability(
                    State{0,0}, Action::RIGHT, State{0, 1}) == 1.0);

            // Expected reward for impossible transition
            REQUIRE(g.state_transition_probability(
                    State{0, 0}, Action::RIGHT, State{1,1}) == 0.0);

            // Expected reward for different transitions
            g.add_transition(State{0,0}, Action::RIGHT, State{0, 1}, 10, 1);
            g.add_transition(State{0,0}, Action::RIGHT, State{1, 1}, 10, 3);
            g.add_transition(State{0,0}, Action::RIGHT, State{1, 0}, 10, 4);
            g.add_transition(State{0,0}, Action::RIGHT, State{1, 0}, 3, 2);
            REQUIRE(g.state_transition_probability(
                    State{0,0}, Action::RIGHT, State{0, 1}) == Approx(0.1));
            REQUIRE(g.state_transition_probability(
                    State{0,0}, Action::RIGHT, State{1, 1}) == Approx(0.3));
            REQUIRE(g.state_transition_probability(
                    State{0,0}, Action::RIGHT, State{1, 0}) == Approx(0.6));
        }
    }

    SECTION("State iteration"){
        using namespace Catch::Matchers;

        SECTION("Full states list") {
            // Create states
            std::vector<State> states_vector(rows * columns);
            size_t r = 0, c = 0;
            std::generate(states_vector.begin(), states_vector.end(), [=, &r, &c]() {
                return State{r++ % rows, c++ % columns};
            });

            auto states_matcher = UnorderedEquals(states_vector);
            REQUIRE_THAT(g.get_states(), states_matcher);
        }

        SECTION("Actions list"){
            // Create actions vector
            std::vector<Action> actions_vector(AvailableGridworldActions.begin(), AvailableGridworldActions.end());
            auto actions_matcher = UnorderedEquals(actions_vector);

            for(const State& s: g.get_states()){
                REQUIRE_THAT(g.get_actions(s), actions_matcher);
            }
        }
    }
}

using rl::mdp::GridworldRandomAgent;
TEST_CASE("Basic GridworldRandomAgent", "[gridworld_agent]"){
    using State = rl::mdp::GridworldState;
    using Action = rl::mdp::GridworldAction;
    GridworldRandomAgent agent(State{0, 0}, 42);

    size_t samples = 10000;
    std::map<Action, size_t> results;
    for (int i = 0; i < samples; ++i) {
        Action a = agent.next_action();
        results[a] += 1;
    }

    // Verify equal results
    auto expected_probability = Approx(1.0 / AvailableGridworldActions.size()).margin(0.01);
    for(auto [action, hits]: results){
        REQUIRE(hits / static_cast<double>(samples) == expected_probability);
    }
}