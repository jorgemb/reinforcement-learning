#include <mdp/graph.h>
#include <mdp/actions.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <numeric>
#include <vector>
#include <sstream>
#include <map>

using namespace Catch::literals;
using rl::mdp::GraphMDP;
using rl::mdp::MDPEnvironment;
using rl::mdp::ActionTraits;

TEST_CASE("GraphMDP", "[graphmdp]") {
    using State = std::string;
    using Action = rl::mdp::TwoWayAction;
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

        SECTION("Is initial"){
            g.set_initial_state("A");
            for(const State& s: g.get_states()){
                if(s == "A"){
                    REQUIRE(g.is_initial_state(s));
                } else {
                    REQUIRE_FALSE(g.is_initial_state(s));
                }
            }
        }

        SECTION("Transitions") {
            for (const auto &a: ActionTraits<Action>::available_actions()) {
                auto transitions = g.get_transitions("C", a);
                REQUIRE(transitions.size() == 1);

                auto [s, r, p] = transitions[0];
                REQUIRE(s == "C");
                REQUIRE(r == 10.0_a);
                REQUIRE(p == 1.0_a);
            }

            REQUIRE_THROWS(g.add_transition("C", Action::RIGHT, "A", 1.0, 1.0));
        }

        SECTION("List terminal states") {
            std::vector<State> expected_terminal_states{"C", "D"};
            auto expected_terminal_matcher = Catch::Matchers::UnorderedEquals(expected_terminal_states);
            REQUIRE_THAT(g.get_terminal_states(), expected_terminal_matcher);
        }

        SECTION("List initial states"){
            g.set_initial_state("A");
            g.set_initial_state("B");
            std::vector<State> initial_states{"A", "B"};

            auto expected_initial_matcher = Catch::Matchers::UnorderedEquals(initial_states);
            REQUIRE_THAT(g.get_initial_states(), expected_initial_matcher);
        }
    }
}


TEST_CASE("MDP Environment", "[graphmdp, mdp]"){
    using State = std::string;
    using Action = rl::mdp::TwoWayAction;

    using MDP = GraphMDP<State, Action>;
    using Environment = rl::mdp::MDPEnvironment<MDP>;

    // Create initial elements
    auto g = std::make_shared<MDP>();
    std::array<State, 3> states{"A", "B", "C"};
    for (size_t i = 0; i < states.size(); ++i) {
        size_t left = (i + 2) % 3;
        size_t right = (i + 1) % 3;

        g->add_transition(states[i], Action::LEFT, states[left], 0.0, 1.0);
        g->add_transition(states[i], Action::LEFT, states[i], 0.0, 1.0);
        g->add_transition(states[i], Action::RIGHT, states[right], 0.0, 1.0);
        g->add_transition(states[i], Action::RIGHT, states[i], 0.0, 1.0);
    }
    Environment e(g, 42);

    SECTION("Start environment"){
        REQUIRE_THROWS(e.start());

        // Set initial state
        State initial{"B"};
        g->set_initial_state(initial);
        REQUIRE(e.start() == initial);
    }

    for(const auto& action: ActionTraits<Action>::available_actions()) {
        DYNAMIC_SECTION("Step environment - " << action) {
            State initial{"B"};
            g->set_initial_state(initial);

            std::map<State, size_t> state_counts;
            size_t total = 1000;
            for(size_t i = 0; i < total; ++i){
                // Do single step
                static_cast<void>(e.start()); // Discard initial state
                auto [next_state, reward, is_final] = e.step(action);
                REQUIRE_FALSE(is_final);
                REQUIRE(reward == 0.0_a);

                // Count state
                state_counts[next_state] += 1;
            }

            // Check states ratio
            auto transitions = g->get_transitions(initial, action);
            auto expected_ratio =
                    Approx(static_cast<double>(total) / static_cast<double>(transitions.size() * total))
                    .margin(0.01);

            std::set<State> transition_states;
            std::transform(transitions.begin(), transitions.end(),
                           std::inserter(transition_states, transition_states.begin()),
                           [](const auto& srp){ return MDP::srp_state(srp); });

            for(const auto& current_state: states){
                if(transition_states.find(current_state) != transition_states.end()){
                    // Expected state
                    double state_ratio = static_cast<double>(state_counts[current_state]) / static_cast<double>(total);
                    REQUIRE(state_ratio == expected_ratio);
                } else {
                    // State should not have appeared
                    REQUIRE(state_counts[current_state] == 0);
                }
            }
        }
    }

    SECTION("End state"){
        State initial{"B"}, final{"A"};
        g->set_initial_state("B");
        g->set_terminal_state("A", 0.0);

        State current_state = e.start();
        while(current_state != final){
            auto [s_i, r, is_final] = e.step(Action::RIGHT);
            if(is_final) {
                REQUIRE(r == 0.0_a);
                REQUIRE(s_i == final);
            }

            current_state = s_i;
        }
    }
}
