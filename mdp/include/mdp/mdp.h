//
// Created by jorge on 10/05/2022.
//

#ifndef REINFORCEMENT_LEARNING_MDP_H
#define REINFORCEMENT_LEARNING_MDP_H

namespace rl::mdp{
    template <class TState, class TAction, class TReward=double, class TProbability=double>
    class MDP{
    public:
        // Type definitions
        using State = TState;
        using Action = TAction;
        using Reward = TReward;
        using Probability = TProbability;
    };
}

#endif //REINFORCEMENT_LEARNING_MDP_H
