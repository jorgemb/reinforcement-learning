#ifndef REINFORCEMENT_LEARNING_AGENTS_H
#define REINFORCEMENT_LEARNING_AGENTS_H

#include <mdp/gridworld.h>
#include <mdp/mdp.h>

#include <random>
#include <vector>

namespace rl::mdp {

    /// Represents a basic gridworld agent with a random policy
    class BasicGridworldAgent: public MDPAgent<GridworldState, GridworldAction> {
    public:
        using RandomEngine = std::default_random_engine;
        using MDPAgent::Action;
        using MDPAgent::State;
        using MDPAgent::Reward;
        using MDPAgent::Probability;

        /// Default constructor
        /// \param seed Seed for generator, use 0 for random seed
        explicit BasicGridworldAgent(RandomEngine::result_type seed = 0);

        /// Returns the first action the agent takes according to the given initial state.
        /// \param initial_state
        /// \return
        Action start(const State& initial_state) override;

        /// Given the reward of the previous action and the following state compute the next action
        /// \param reward Reward of the previous action
        /// \param next_state State after the previous action
        /// \return Next action to perform
        Action step(const Reward& reward, const State& next_state) override;

        /// Called when entering the final state, provides the reward of the last action taken
        /// \param reward Reward of the previous action
        void end(const Reward& reward) override;

        /// Returns the total reward at the moment
        /// \return
        [[nodiscard]]
        Reward get_reward() const { return m_total_reward; }

    protected:
        std::vector<GridworldAction> m_actions;

        RandomEngine m_random_engine;
        std::uniform_int_distribution<uint32_t> m_distribution;

        Reward m_total_reward;
    };



} // namespace rl::mdp

#endif //REINFORCEMENT_LEARNING_AGENTS_H
