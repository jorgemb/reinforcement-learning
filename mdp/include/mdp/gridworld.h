//
// Created by Jorge Martinez Bonilla on 10/05/2022.
//

#ifndef REINFORCEMENT_LEARNING_GRIDWORLD_H
#define REINFORCEMENT_LEARNING_GRIDWORLD_H

#include "mdp/mdp.h"

#include <map>
#include <tuple>
#include <random>
#include <limits>
#include <ostream>

namespace rl::mdp {
    enum class GridworldAction {
        LEFT,
        RIGHT,
        UP,
        DOWN
    };

    /// Class for representing the current state in Gridworld
    class GridworldState {
    public:
        size_t row, column;

        auto operator<(const GridworldState &other) const {
            return std::make_pair(row, column) < std::make_pair(other.row, other.column);
        }

        auto operator==(const GridworldState &other) const { return row == other.row && column == other.column; }
    };

    class Gridworld: public MDP<GridworldState, GridworldAction> {
    public:
        using RandomEngine = std::default_random_engine;

        /// Creates a new Gridworld with the given amount of rows and columns
        /// \param rows
        /// \param columns
        Gridworld(size_t rows, size_t columns,
                  RandomEngine::result_type seed = std::numeric_limits<RandomEngine::result_type>::max());

        /// Returns the number of rows
        /// \return
        [[nodiscard]]
        size_t get_rows() const;

        /// Returns the number of columns
        /// \return
        [[nodiscard]]
        size_t get_columns() const;

        /// Returns the Transition from a state action pair. If there are several states
        /// it returns a non-deterministic one
        /// \param state_action
        /// \return
        [[nodiscard]]
        Transition get_transition(const State &state, const Action &action) const override;

        /// Adds a transition with the given probability
        /// \param state
        /// \param action
        /// \param new_state
        /// \param reward
        /// \param probability
        void add_transition(const State &state,
                            const Action &action,
                            const State &new_state,
                            const Reward &reward,
                            const Probability &probability) override;

        /// Returns the expected reward of the State-Action pair
        /// \param state
        /// \param action
        /// \return
        double expected_reward(const State &state, const Action &action) const override;

    private:
        using DynamicsMap = std::multimap<StateAction, StateRewardProbability>;
        DynamicsMap m_dynamics;
        size_t m_rows, m_columns;
        mutable RandomEngine m_random_engine;

        /// Returns the default transition for the state-action pair
        /// \param state
        /// \param action
        /// \return
        [[nodiscard]]
        Transition transition_default(const State &state, const Action &action) const;

        /// Selects a random transition given the possible outcomes of the state-action pair
        /// \param state
        /// \param action
        /// \return
        [[nodiscard]]
        Gridworld::Transition transition_non_deterministic(const State &state, const Action &action) const;
    };

} // namespace rl::mdp

/// Outputs the action of the Gridworld
/// \param os
/// \param action
/// \return
std::ostream &operator<<(std::ostream &os, const rl::mdp::Gridworld::Action &action);

/// Outputs the current State
/// \param os
/// \param state
/// \return
std::ostream &operator<<(std::ostream &os, const rl::mdp::Gridworld::State &state);



#endif //REINFORCEMENT_LEARNING_GRIDWORLD_H
