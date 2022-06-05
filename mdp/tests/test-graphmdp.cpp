#include "mdp/graph_mdp.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <numeric>
#include <vector>

using namespace Catch::literals;
using rl::mdp::GraphMDP;
using rl::mdp::GraphMDP_Greedy;

using State = std::string;
enum class GraphAction{ LEFT, RIGHT };
using Action = GraphAction;

template<>
std::vector<GraphAction> rl::mdp::get_actions_list() {
    return {GraphAction::LEFT, GraphAction::RIGHT};
}

std::ostream& operator<<(std::ostream& os, const Action& a){
    if(a == Action::LEFT) os << "LEFT";
    else os << "RIGHT";

    return os;
}

TEST_CASE("GraphMDP", "[graphmdp]") {
    GraphMDP<State, Action> g;
    std::vector<State> states{"BAD", "A", "B", "C", "D", "E", "GOOD"};

    SECTION("Default values") {
        REQUIRE(g.get_states().empty());
        REQUIRE_THROWS(g.get_actions("ANY").empty());
        REQUIRE_THROWS(g.get_transitions("ANY", Action::RIGHT).empty());
    }

    SECTION("States"){
        // Insert a transition over every pair of states
        auto iter_a = states.begin(), iter_b = std::next(iter_a), iter_end = states.end();
        for(; iter_b != iter_end; ++iter_a, ++iter_b){
            g.add_transition(*iter_a, Action::RIGHT, *iter_b, 10.0, 1.0);
        }

        // Default size
        REQUIRE(g.get_states().size() == states.size());
        auto state_match = Catch::Matchers::UnorderedEquals(states);
        REQUIRE_THAT(g.get_states(), state_match);
    }

    SECTION("Transitions") {
        std::string A{"A"}, B{"B"};
        SECTION("Basic") {
            g.add_transition(A, Action::RIGHT, B, 0.0, 1.0);
            auto transitions = g.get_transitions(A, Action::RIGHT);

            REQUIRE(transitions.size() == 1);
            auto [s, r, p] = transitions[0];
            REQUIRE(s == B);
            REQUIRE(r == 0.0_a);
            REQUIRE(p == 1.0_a);
        }

        SECTION("Multiple transitions"){
            g.add_transition(A, Action::RIGHT, B, 10.0, 1.0);
            g.add_transition(A, Action::RIGHT, B, 10.0, 1.0);

            // Verify probabilities
            auto transitions = g.get_transitions(A, Action::RIGHT);
            REQUIRE(transitions.size() == 2);
            for(auto [s, r, p]: transitions){
                REQUIRE(s == B);
                REQUIRE(r == 10.0_a);
                REQUIRE(p == 0.5_a);
            }

            // Verify actions
            auto actions = g.get_actions(A);
            REQUIRE(actions.size() == 1);
            REQUIRE(actions[0] == Action::RIGHT);

            // Verify multiple actions
            g.add_transition(A, Action::LEFT, B, 10.0, 1.0);
            actions = g.get_actions(A);
            REQUIRE(actions.size() == 2);
            std::vector<Action> available_actions{Action::RIGHT, Action::LEFT};
            auto actions_matcher = Catch::Matchers::UnorderedEquals(available_actions);
            REQUIRE_THAT(actions, actions_matcher);
        }

        SECTION("Transition probability"){
            g.add_transition("A", Action::LEFT, "B", 100.0, 2);
            g.add_transition("A", Action::LEFT, "B", 30.0, 1);
            g.add_transition("A", Action::LEFT, "A", 10.0, 7);
            REQUIRE(g.state_transition_probability("A", Action::LEFT, "B") == 0.3_a);
            REQUIRE(g.state_transition_probability("B", Action::RIGHT, "A") == 0.0_a);
        }
    }

    SECTION("Expected reward"){
        g.add_transition("A", Action::LEFT, "B", 100.0, 3);
        g.add_transition("A", Action::LEFT, "A", 10.0, 7);

        REQUIRE(g.expected_reward("A", Action::LEFT) == Approx(100.0 * 0.3 + 10.0 * 0.7));
    }

    SECTION("Terminal states"){
        g.add_transition("A", Action::RIGHT, "B", 10, 1);
        g.add_transition("B", Action::RIGHT, "C", 10, 1);
        g.add_transition("B", Action::RIGHT, "D", 10, 1);
        g.set_terminal_state("C", 10.0);
        g.set_terminal_state("D", 0.0);

        SECTION("Is terminal") {
            REQUIRE_FALSE(g.is_terminal_state("A"));
            REQUIRE_FALSE(g.is_terminal_state("B"));
            REQUIRE(g.is_terminal_state("C"));
        }

        SECTION("Transitions") {
            for (const auto &a: rl::mdp::get_actions_list<Action>()) {
                auto transitions = g.get_transitions("C", a);
                REQUIRE(transitions.size() == 1);

                auto [s, r, p] = transitions[0];
                REQUIRE(s == "C");
                REQUIRE(r == 10.0_a);
                REQUIRE(p == 1.0_a);
            }

            REQUIRE_THROWS(g.add_transition("C", Action::RIGHT, "A", 1.0, 1.0));
        }

        SECTION("List") {
            std::vector<State> expected_terminal_states{"C", "D"};
            auto expected_terminal_matcher = Catch::Matchers::UnorderedEquals(expected_terminal_states);
            REQUIRE_THAT(g.get_terminal_states(), expected_terminal_matcher);
        }
    }
}

TEST_CASE("GraphMDP_GreedyPolicy", "[graphmdp]"){
    // Set up the graph MDP
    auto g = std::make_shared<GraphMDP<State,Action>>();

    // According to Example 6.2 from RL Book
    std::array<State, 6> states{"A", "B", "C", "D", "E", "GOOD"};
    auto node_iter = states.begin(), next_node_iter = std::next(states.begin());
    for(; next_node_iter != states.end(); ++node_iter, ++next_node_iter){
        double r_right = *next_node_iter == "GOOD" ? 1.0 : -1.0;

        g->add_transition(*node_iter, Action::RIGHT, *next_node_iter, r_right, 1.0);
        g->add_transition(*next_node_iter, Action::LEFT, *node_iter, -1.0, 1.0);
    }
    g->set_terminal_state("GOOD", 0.0);

    // Policy
    GraphMDP_Greedy<State, Action> policy(g, 1.0);

    SECTION("Default values"){
        SECTION("Probabilities") {
            auto default_probability = Approx(1.0 / static_cast<double>(rl::mdp::get_actions_list<Action>().size()));
            for (const auto &s: states) {
                INFO("State is " << s);

                // Terminal states should not have probabilities
                if(g->is_terminal_state(s)){
                    REQUIRE_THROWS(policy.get_action_probabilities(s));
                    continue;
                }

                auto action_probabilities = policy.get_action_probabilities(s);
                REQUIRE_FALSE(action_probabilities.empty());
                for (const auto &[a, p]: action_probabilities) {
                    if(s == "A"){
                        REQUIRE(p == 1.0_a);
                    } else {
                        REQUIRE(p == default_probability);
                    }
                }
            }
        }

        SECTION("Value function)"){
            for (const auto& s: states) {
                REQUIRE(policy.value_function(s) == 0.0_a);
            }
        }
    }

    SECTION("Policy iteration"){
        SECTION("1st iteration"){
            // Evaluation
            auto change = policy.policy_evaluation();
            REQUIRE(change == 1.0_a);
            for(const auto& s: states) {
                INFO("State is " << s);
                auto value_function = policy.value_function(s);
                if (g->is_terminal_state(s)) {
                    REQUIRE(value_function == 0.0_a);
                } else if (s == "E") {
                    REQUIRE(value_function == 0.0_a);
                } else {
                    REQUIRE(value_function == -1.0_a);
                }
            }

            // Make greedy
            REQUIRE(policy.update_policy());
            for(const auto& s: states){
                if(g->is_terminal_state(s)) continue;
                INFO("State is " << s);

                auto action_prob = policy.get_action_probabilities(s);
                if(s == "E" || s == "D") {
                    for (const auto &[a, p]: action_prob) {
                        if (a == Action::RIGHT) REQUIRE(p == 1.0_a);
                        if (a == Action::LEFT) REQUIRE(p == 0.0_a);
                    }
                } else if (s == "A") {
                    REQUIRE(action_prob.size() == 1);
                    auto [a,p] = action_prob[0];
                    REQUIRE(a == Action::RIGHT);
                    REQUIRE(p == 1.0_a);
                } else {
                    for(const auto& [a, p]: action_prob){
                        INFO("Action is " << a);
                        REQUIRE(p == 0.5_a);
                    }
                }
            }
        }

        SECTION("Final iteration") {
            // Iterate until there are no more policy changes
            bool policy_changed = true;
            size_t iterations = 0;
            while (policy_changed || iterations < 10) {
                policy.policy_evaluation();
                policy_changed = policy.update_policy();
                ++iterations;
            }
            INFO("Total iterations: " << iterations);

            // Value function
            INFO("A - " << policy.value_function("A"));
            INFO("B - " << policy.value_function("B"));
            INFO("C - " << policy.value_function("C"));
            INFO("D - " << policy.value_function("D"));
            INFO("E - " << policy.value_function("E"));
            INFO("GOOD - " << policy.value_function("GOOD"));


            for (const auto &s: states) {
                if (g->is_terminal_state(s)) continue;

                INFO("State is " << s);
                for(const auto& [a, p]: policy.get_action_probabilities(s)){
                    if(a == Action::RIGHT) REQUIRE(p == 1.0_a);
                    else REQUIRE(p == 0.0_a);
                }
            }
        }
    }
}
