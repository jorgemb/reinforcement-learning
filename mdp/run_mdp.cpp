#include "mdp/gridworld.h"

#include <memory>
#include <iostream>
#include <fmt/core.h>

using namespace rl::mdp;

int main(){
    // Do a run for a Gridworld
    auto gridworld = std::make_shared<Gridworld>(4, 4);
    using State = GridworldState;
    using Action = GridworldAction;

    // .. create transitions
    for(const auto& s: gridworld->get_states()){
        for(const auto& a: gridworld->get_actions(s)){
            // Check if it is terminal state or other
            if(s == State{0, 0} || s == State{3, 3}){
                // Terminal state
                gridworld->add_transition(s, a, s, 0.0, 1.0);
            } else {
                // Other state
                auto [s_i, r, p] = gridworld->get_transitions(s, a)[0];
                gridworld->add_transition(s, a, s_i, -1.0, 1.0);
            }
        }
    }

    // Create a policy
    GridworldGreedyPolicy policy(gridworld, 1.0);
    std::cout << "First evaluation\n";
    policy.policy_evaluation();
    policy.update_policy();
    std::cout << policy << std::endl;

    std::cout << "\nSecond evaluation\n";
    policy.policy_evaluation();
    policy.update_policy();
    std::cout << policy << std::endl;

    std::cout << "\nThird evaluation\n";
    policy.policy_evaluation();
    policy.update_policy();
    std::cout << policy << std::endl;

    return 0;
}