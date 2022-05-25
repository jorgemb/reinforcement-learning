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

std::ostream& operator<<(std::ostream& os, const Action& a){
    if(a == Action::LEFT) os << "LEFT";
    else os << "RIGHT";

    return os;
}

TEST_CASE("GraphMDP", "[graphmdp]") {
    GraphMDP<State, Action> g({Action::LEFT, Action::RIGHT});
    std::vector<State> states{"BAD", "A", "B", "C", "D", "E", "GOOD"};

    SECTION("Default values") {
        REQUIRE(g.get_states().size() == 0);
        REQUIRE_THROWS(g.get_actions("ANY").size() == 0);
        REQUIRE_THROWS(g.get_transitions("ANY", Action::RIGHT).size() == 0);
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

        // Terminal states
        State terminal{"BAD"};
        REQUIRE_THROWS(g.set_terminal_state("ANY", 0.0));
        REQUIRE_FALSE(g.is_terminal_state(terminal));
        g.set_terminal_state(terminal, -10.0);
        REQUIRE(g.is_terminal_state(terminal));
        for(const auto& a: {Action::LEFT, Action::RIGHT}){
            auto transitions = g.get_transitions(terminal, a);
            REQUIRE(transitions.size() == 1);
            auto [s, r, p] = transitions[0];
            REQUIRE(s == terminal);
            REQUIRE(r == -10.0_a);
            REQUIRE(p == 1.0_a);
        }

        // Adding transitions to terminal states is illegal
        REQUIRE_THROWS(g.add_transition(terminal, Action::LEFT, "A", 1.0, 1.0));

        // Check terminal states list
        std::vector<State> terminal_states{"BAD", "GOOD"};
        for(auto s: terminal_states) g.set_terminal_state(s, 0.0);
        auto terminal_match = Catch::Matchers::UnorderedEquals(terminal_states);
        REQUIRE_THAT(g.get_terminal_states(), terminal_match);
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
            g.add_transition(A, Action::RIGHT, A, 10.0, 3.0);
            g.add_transition(A, Action::RIGHT, B, 10.0, 1.0);
            g.add_transition(A, Action::RIGHT, B, 10.0, 1.0);

            REQUIRE(g.state_transition_probability(A, Action::RIGHT, B) == 0.40_a);
        }
    }

    SECTION("Reward"){
        State A{"A"}, B{"B"}, C{"C"};
        g.add_transition(A, Action::LEFT, A, -10, 0.25);
        g.add_transition(A, Action::LEFT, B, 50, 0.50);
        g.add_transition(A, Action::LEFT, C, 5, 0.25);

        double expected_reward = -10.0*0.25 + 50*0.5 + 5*0.25;
        REQUIRE(g.expected_reward(A, Action::LEFT) == Approx(expected_reward));
    }

}
