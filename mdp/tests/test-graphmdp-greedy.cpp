#include <mdp/graph.h>
#include <mdp/graph_policy.h>
#include <mdp/actions.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

using namespace Catch::literals;
using rl::mdp::GraphMDP;
using rl::mdp::GraphMDP_Greedy;

using State = std::string;
using Action = rl::mdp::TwoWayAction;
using rl::mdp::ActionTraits;

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
            auto default_probability = Approx(1.0 / static_cast<double>(ActionTraits<Action>::available_actions().size()));
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

