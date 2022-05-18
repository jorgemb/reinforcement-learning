#include "mdp/gridworld.h"

#include <memory>
#include <iostream>
#include <fmt/core.h>
using namespace rl::mdp;

void print_value_function(const std::shared_ptr<Gridworld> &gridworld, const GridworldGreedyPolicy &policy);

int main(){
    // Do a run for a Gridworld
    auto gridworld = std::make_shared<Gridworld>(4, 4);
    using State = GridworldState;
    using Action = GridworldAction;

    // .. create transitions
    gridworld->set_terminal_state(State{0,0}, 1.0);
    gridworld->set_terminal_state(State{3, 3}, 1.0);
    for(const auto& s: gridworld->get_states()){
        for(const auto& a: gridworld->get_actions(s)){
            if(gridworld->is_terminal_state(s)) continue;

            // Add the transitions to the rest of the states
            auto [s_i, reward, probability] = gridworld->get_transitions(s, a)[0];
//            if(gridworld->is_terminal_state(s_i)){
//                gridworld->add_transition(s, a, s_i, 1.0, 1.0);
//            } else {
//                gridworld->add_transition(s, a, s_i, -1.0, 1.0);
//            }
                gridworld->add_transition(s, a, s_i, -1.0, 1.0);
        }
    }

    // Create a policy
    GridworldGreedyPolicy policy(gridworld, 1.0);
    print_value_function(gridworld, policy);

    std::cout << "\nFirst evaluation\n";
    policy.policy_evaluation();
//    policy.update_policy();
    std::cout << policy << std::endl;
    print_value_function(gridworld, policy);


    std::cout << "\nSecond evaluation\n";
    policy.policy_evaluation();
//    policy.update_policy();
    std::cout << policy << std::endl;
    print_value_function(gridworld, policy);

    std::cout << "\nInf evaluation\n";
    int n = 0;
    while(policy.policy_evaluation() >= 0.0001 && n++ < 1000){
        policy.policy_evaluation();
    }
//    policy.update_policy();
    fmt::print("...after {:d} iterations", n);
    std::cout << policy << std::endl;
    print_value_function(gridworld, policy);

    return 0;
}

void print_value_function(const std::shared_ptr<Gridworld> &gridworld, const GridworldGreedyPolicy &policy) {
    for (size_t i = 0; i < gridworld->get_rows(); ++i) {
        for (size_t j = 0; j < gridworld->get_columns(); ++j) {
            fmt::print("{: .2f} ", policy.value_function(GridworldState{i, j}));
        }
        fmt::print("\n");
    }
}
