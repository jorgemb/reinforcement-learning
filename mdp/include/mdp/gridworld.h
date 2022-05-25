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
#include <array>
#include <memory>

namespace rl::mdp {
    /// Actions for the Gridworld
    enum class GridworldAction {
        LEFT,
        RIGHT,
        UP,
        DOWN
    };

    template<>
    inline std::vector<GridworldAction> get_actions_list() {
        return {GridworldAction::LEFT, GridworldAction::RIGHT, GridworldAction::UP, GridworldAction::DOWN};
    }

    /// Class for representing the current state in Gridworld
    class GridworldState {
    public:
        GridworldState() = default;
        GridworldState(size_t r, size_t c): row(r), column(c) {}

        size_t row, column;

        auto operator<(const GridworldState &other) const {
            return std::make_pair(row, column) < std::make_pair(other.row, other.column);
        }

        auto operator==(const GridworldState &other) const { return row == other.row && column == other.column; }
    };

    /// Represents a grid based MDP with transitions between cells
    class Gridworld: public MDP<GridworldState, GridworldAction> {
    public:
        using RandomEngine = std::default_random_engine;

        /// Creates a new Gridworld with the given amount of rows and columns
        /// \param rows
        /// \param columns
        Gridworld(size_t rows, size_t columns);

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
        std::vector<StateRewardProbability> get_transitions(const State &state, const Action &action) const override;

        /// Adds a transition with the given weight (NOTE: This is later normalized to sum 1)
        /// \param state
        /// \param action
        /// \param new_state
        /// \param reward
        /// \param weight
        void add_transition(const State &state,
                            const Action &action,
                            const State &new_state,
                            const Reward &reward,
                            const Probability &weight) override;

        /// Returns the expected reward of the State-Action pair
        /// \param state
        /// \param action
        /// \return
        [[nodiscard]]
        Reward expected_reward(const State &state, const Action &action) const override;

        /// Returns probability of going to a state from a state-action pair.
        /// \param from_state
        /// \param action
        /// \param to_state
        /// \return
        [[nodiscard]]
        Probability state_transition_probability(const State &from_state, const Action &action,
                                            const State &to_state) const override;


        /// Returns a vector with all the possible states that the MDP can contain.
        /// \return
        [[nodiscard]]
        std::vector<State> get_states() const override;

        /// Returns a list with the available actions for a given state.
        /// \param state
        /// \return
        [[nodiscard]]
        std::vector<Action> get_actions(const State &state) const override;

        /// Marks a state as a terminal state. This makes all transitions out of this state to point to it again
        /// with reward zero.
        /// \param s
        void set_terminal_state(const State& s, const Reward& default_reward) override;

        /// Returns true if the given State is a terminal state.
        /// \param s
        /// \return
        [[nodiscard]]
        bool is_terminal_state(const State& s) const override;

        /// Returns a list of the terminal states.
        /// \return
        [[nodiscard]]
        std::vector<State> get_terminal_states() const override;

    private:
        using DynamicsMap = std::multimap<StateAction, StateRewardProbability>;
        DynamicsMap m_dynamics;
        size_t m_rows, m_columns;

        std::vector<State> m_terminal_states;

        /// Returns the default transition for the state-action pair
        /// \param state
        /// \param action
        /// \return
        [[nodiscard]]
        StateRewardProbability transition_default(const State &state, const Action &action) const;
    };

    class GridworldGreedyPolicy: public MDPPolicy<GridworldState, GridworldAction>{
    public:
        /// Default constructor with rows and columns.
        /// \param rows
        /// \param columns
        explicit GridworldGreedyPolicy(std::shared_ptr<Gridworld> gridworld, double gamma);

        /// Return the possible actions and its probabilities based on the current state.
        /// \param state
        /// \return
        [[nodiscard]]
        std::vector<ActionProbability> get_action_probabilities(const State &state) const override;

        /// Returns the gridworld associated to the policy.
        /// \return
        [[nodiscard]]
        std::shared_ptr<Gridworld> get_gridworld() const;

        /// Approximates the value function doing a single policy evaluation.
        /// \param epsilon
        /// \return
        double policy_evaluation();

        /// Makes the policy greedy according to the value function
        /// \return
        void update_policy();

        /// Returns the value function result given a state.
        /// \param state
        /// \return
        [[nodiscard]]
        Reward value_function(const State &state) const override;

    private:
        std::shared_ptr<Gridworld> m_gridworld;
        size_t m_rows, m_columns;
        double m_gamma;
        std::vector<Probability> m_value_function_table;

        // Way of representing action-probability pairs
        using ActionProbabilityMap = std::map<Action, Probability>;
        std::map<State, ActionProbabilityMap> m_state_action_probability_map;

        /// Returns a copy a the value from the value function table
        /// \param state
        /// \return
        [[nodiscard]]
        Probability value_from_table(const State& state) const{ return m_value_function_table[state.row * m_columns + state.column]; };

        /// Returns a reference to the value function table
        /// \param state
        /// \return
        Probability& value_from_table(const State& state) { return m_value_function_table[state.row * m_columns + state.column]; };
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

/// Outputs the current policy for a gridworld
/// \param os
/// \param greedy_policy
/// \return
std::ostream& operator<<(std::ostream& os, const rl::mdp::GridworldGreedyPolicy& greedy_policy);



#endif //REINFORCEMENT_LEARNING_GRIDWORLD_H
