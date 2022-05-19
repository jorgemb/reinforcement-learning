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
    }

}
