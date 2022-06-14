#ifndef REINFORCEMENT_LEARNING_AGENTS_H
#define REINFORCEMENT_LEARNING_AGENTS_H

#include <mdp/gridworld.h>
#include <mdp/mdp.h>

namespace rl::mdp {

    /// Represents a basic gridworld agent with a random policy
    class BasicGridworldAgent: public MDPAgent<GridworldState, GridworldAction> {
        /// Returns the first action the agent takes according to the given initial state.
        /// \param initial_state
        /// \return
        Action start(const State& initial_state) override {

        }

        /// Given the reward of the previous action and the following state compute the next action
        /// \param reward Reward of the previous action
        /// \param next_state State after the previous action
        /// \return Next action to perform
        Action step(const Reward& reward, const State& next_state) override {

        }

        /// Called when entering the final state, provides the reward of the last action taken
        /// \param reward Reward of the previous action
        void end(const Reward& reward) override {

        }

    };

} // namespace rl::mdp

#endif //REINFORCEMENT_LEARNING_AGENTS_H
