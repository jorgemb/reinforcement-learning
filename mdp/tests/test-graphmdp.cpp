#include "mdp/graph_mdp.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <array>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace Catch::literals;
using rl::mdp::GraphMDP;

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
    std::vector<State> states{"Bad", "A", "B", "C", "D", "E", "Good"};

    SECTION("Default values") {
        REQUIRE(g.get_states().size() == 0);
        REQUIRE_THROWS(g.get_actions("ANY").size() == 0);
        REQUIRE_THROWS(g.get_transitions("ANY", Action::RIGHT).size() == 0);
    }

    SECTION("States"){
        // Insert a transition over every pair of states
        std::vector<State> _ignore;
        std::adjacent_difference(states.cbegin(), states.cend(), std::back_inserter(_ignore),
                                 [&g](const auto& sA, const auto& sB){
            g.add_transition(sA, Action::RIGHT, sB, 10.0, 1.0);
            return sA;
        });

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
