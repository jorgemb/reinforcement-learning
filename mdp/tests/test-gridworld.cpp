#include "mdp/gridworld.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <array>
#include <algorithm>
#include <vector>

using namespace Catch::literals;
using rl::mdp::Gridworld;
using rl::mdp::AvailableGridworldActions;

using Action = Gridworld::Action;
using State = Gridworld::State;
using StateRewardProbability = rl::mdp::Gridworld::StateRewardProbability;

TEST_CASE("Gridworld", "[gridworld]") {
    size_t rows = 4, columns = 4;
    Gridworld g(rows, columns);

    SECTION("Standard properties") {
        REQUIRE(g.get_rows() == rows);
        REQUIRE(g.get_columns() == columns);
    }

    SECTION("Default actions") {
        // All actions should be deterministic
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < columns; ++j) {
                for (auto action: AvailableGridworldActions) {
                    DYNAMIC_SECTION("Action for " << i << "," << j << " - " << action) {
                        auto state = State{i, j};
                        auto transitions = g.get_transitions(state, action);
                        REQUIRE(transitions.size() == 1);
                        auto [new_state, reward, probability] = transitions[0];

                        // Verify new_state transition
                        REQUIRE(new_state.row >= 0);
                        REQUIRE(new_state.row < rows);
                        REQUIRE(new_state.column >= 0);
                        REQUIRE(new_state.column < columns);


                        // Check if in the edge, which would mean that the returned state is the same
                        if (new_state == state) REQUIRE(reward == -1.0_a);
                        else
                            REQUIRE(reward == 0.0_a);

                        // Determine the real new state
                        switch (action) {
                            case Action::LEFT:
                                REQUIRE(new_state == State{i, j == 0 ? j : j - 1});
                                break;
                            case Action::RIGHT:
                                REQUIRE(new_state == State{i, j >= columns-1 ? j : j + 1});
                                break;
                            case Action::UP:
                                REQUIRE(new_state == State{i == 0 ? i : i - 1, j});
                                break;
                            case Action::DOWN:
                                REQUIRE(new_state == State{i >= rows-1 ? i : i + 1, j});
                                break;
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
                auto [state, reward, probability] = g.get_transitions(State{0, 1}, action)[0];
                REQUIRE(state == State{4, 1});
                REQUIRE(reward == 10.0_a);
                REQUIRE(probability == 1.0_a);
            }
        }
    }

    SECTION("Non-deterministic transitions") {
        // Create non-deterministic transition
        State from{0, 0};
        Action action{Action::LEFT};

        std::vector<StateRewardProbability> test_transitions{
                StateRewardProbability{State{1, 0}, 5.0, 1},
                StateRewardProbability{State{2, 0}, 10.0, 1},
                StateRewardProbability{State{3, 3}, 20.0, 1}
        };

        // Add transitions
        for (const auto &srp: test_transitions) {
            g.add_transition(from, action, std::get<0>(srp), std::get<1>(srp), std::get<2>(srp));
        }

        auto expected_probability = 1.0 / static_cast<double>(test_transitions.size());
        std::transform(test_transitions.begin(), test_transitions.end(), test_transitions.begin(),
                       [expected_probability](const auto &srp) {
                           StateRewardProbability new_srp{srp};
                           std::get<2>(new_srp) = expected_probability;
                           return new_srp;
                       });

        auto matcher = Catch::Matchers::UnorderedEquals(test_transitions);
        REQUIRE_THAT(g.get_transitions(from, action), matcher);
    }

    SECTION("MDP values") {
        SECTION("Expected reward") {
            // Expected reward for default transition
            REQUIRE(g.expected_reward(State{0, 0}, Action::LEFT) == 0.0_a);

            // Expected reward for added transition
            g.add_transition(State{0, 0}, Action::LEFT, State{1, 1}, 10., 1.0);
            REQUIRE(g.expected_reward(State{0, 0}, Action::LEFT) == 10.0_a);

            // Expected reward for non-deterministic transition
            g.add_transition(State{0, 0}, Action::LEFT, State{2, 2}, 100, 3.0);
            g.add_transition(State{0, 0}, Action::LEFT, State{3, 3}, 50, 6.0);
            double expected = (10.0 * 0.1) + (100.0 * 0.3) + (50 * 0.6);
            REQUIRE(g.expected_reward(State{0, 0}, Action::LEFT) == Approx(expected));
        }

        SECTION("State-transition probability") {
            // Expected reward for known transition
            REQUIRE(g.state_transition_probability(
                    State{0, 0}, Action::RIGHT, State{0, 1}) == 1.0);

            // Expected reward for impossible transition
            REQUIRE(g.state_transition_probability(
                    State{0, 0}, Action::RIGHT, State{1, 1}) == 0.0);

            // Expected reward for different transitions
            g.add_transition(State{0, 0}, Action::RIGHT, State{0, 1}, 10, 1);
            g.add_transition(State{0, 0}, Action::RIGHT, State{1, 1}, 10, 3);
            g.add_transition(State{0, 0}, Action::RIGHT, State{1, 0}, 10, 4);
            g.add_transition(State{0, 0}, Action::RIGHT, State{1, 0}, 3, 2);
            REQUIRE(g.state_transition_probability(
                    State{0, 0}, Action::RIGHT, State{0, 1}) == Approx(0.1));
            REQUIRE(g.state_transition_probability(
                    State{0, 0}, Action::RIGHT, State{1, 1}) == Approx(0.3));
            REQUIRE(g.state_transition_probability(
                    State{0, 0}, Action::RIGHT, State{1, 0}) == Approx(0.6));
        }
    }

    SECTION("State iteration") {
        using namespace Catch::Matchers;

        SECTION("Full states list") {
            // Create states
            std::vector<State> states_vector(rows * columns);
            size_t r = 0, c = 0;
            std::generate(states_vector.begin(), states_vector.end(), [=, &r, &c]() {
                State s{r, c};
                ++c;
                r += c / columns;
                c = c % columns;
                return s;
            });

            auto states_matcher = UnorderedEquals(states_vector);
            auto g_states = g.get_states();
            REQUIRE_THAT(g.get_states(), states_matcher);

            // Verify state bounds
            for(const auto& s: g_states){
                auto [i, j] = s;
                REQUIRE(i >= 0);
                REQUIRE(i < rows);
                REQUIRE(j >= 0);
                REQUIRE(j < columns);
            }
        }

        SECTION("Actions list") {
            // Create actions vector
            std::vector<Action> actions_vector(AvailableGridworldActions.begin(), AvailableGridworldActions.end());
            auto actions_matcher = UnorderedEquals(actions_vector);

            for (const State &s: g.get_states()) {
                REQUIRE_THAT(g.get_actions(s), actions_matcher);
            }
        }
    }
}

using ActionProbability = rl::mdp::GreedyPolicy::ActionProbability;

TEST_CASE("Gridworld Policy", "[gridworld]"){
    // Initialize elements
    size_t rows = 4, columns = 4;
    auto g = std::make_shared<Gridworld>(rows, columns);

    SECTION("Default values"){
        rl::mdp::GreedyPolicy policy(g, 1.0);
        SECTION("Probabilities") {
            double default_probability = 1.0 / static_cast<double>(AvailableGridworldActions.size());
            std::vector<ActionProbability> default_action_probabilities;
            std::transform(AvailableGridworldActions.begin(), AvailableGridworldActions.end(),
                           std::back_inserter(default_action_probabilities),
                           [default_probability](const auto& val){
                return ActionProbability{val, default_probability};
            } );
            auto action_probability_matcher = Catch::Matchers::UnorderedEquals(default_action_probabilities);

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < columns; ++j) {
                    auto action_probabilities = policy.get_action_probabilities(State{i, j});

                    REQUIRE(action_probabilities.size() == 4);
                    REQUIRE_THAT(action_probabilities, action_probability_matcher);
                }
            }
        }

        SECTION("Value function"){
            SECTION("Default value function"){
                for(const auto& s: g->get_states()){
                    REQUIRE(policy.value_function(s) == 0.0_a);
                }
            }
        }
    }

    SECTION("Expected values"){
        // Set the expected world
        for(const auto& s: g->get_states()){
            for(const auto& a: g->get_actions(s)){
                // Check if it is terminal state or other
                if(s == State{0, 0} || s == State{3, 3}){
                    // Terminal state
                    g->add_transition(s, a, s, 0.0, 1.0);
                } else {
                    // Other state
                    auto [s_i, r, p] = g->get_transitions(s, a)[0];
                    g->add_transition(s, a, s_i, -1.0, 1.0);


                }
            }
        }

        SECTION("Policy evaluation"){
            // Values taken from Sutton & Barto [figure 4.2]
            rl::mdp::GreedyPolicy policy(g, 1.0);

            // First evaluation
            double change = policy.policy_evaluation();
            REQUIRE(std::abs(change) == 1.0_a);

            for(const auto& s: g->get_states()){
                if(s == State{0, 0} || s == State{3, 3}){
                    REQUIRE(policy.value_function(s) == 0.0_a);
                    INFO("Final state");
                } else {
                    REQUIRE(policy.value_function(s) == -1.0_a);
                    INFO("Normal state");
                }
            }

            // Second evaluation
            change = policy.policy_evaluation();
            REQUIRE(std::abs(change) == 1.0_a);

            std::vector possible_values{0.0_a, -1.75_a, -2.0_a};
            for(const auto& s: g->get_states()){
                INFO("State: " << s << " value:" << policy.value_function(s));
                REQUIRE(std::find(possible_values.cbegin(), possible_values.cend(), policy.value_function(s)) != possible_values.cend());
            }
        }

        SECTION("Policy improvement"){
            rl::mdp::GreedyPolicy policy(g, 1.0);
            policy.policy_evaluation();
            policy.update_policy();

            // Verify actions
            using ActionList = std::vector<Action>;
            using Line = std::vector<ActionList>;
            for(const auto& s: g->get_states()){
                for(const auto& [a, p]: policy.get_action_probabilities(s)){
                    if(p == 0.0_a) continue;
                    INFO("State: " << s);
                    INFO("Probability: " << p);

                    // Check type of state
                    if(s == State{0, 1}){
                        REQUIRE(a == Action::LEFT);
                    } else if(s == State{1, 0}){
                        REQUIRE(a == Action::UP);
                    } else if(s == State{3, 2}){
                        REQUIRE(a == Action::RIGHT);
                    } else if(s == State{2, 3}){
                        REQUIRE(a == Action::DOWN);
                    }
                }
            }
        }
    }
}