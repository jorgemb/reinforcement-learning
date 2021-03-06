#ifndef REINFORCEMENT_LEARNING_MDP_GRIDWORLD_H
#define REINFORCEMENT_LEARNING_MDP_GRIDWORLD_H

#include <mdp/mdp.h>
#include <mdp/actions.h>

#include <map>
#include <set>
#include <tuple>
#include <random>
#include <limits>
#include <ostream>
#include <array>
#include <memory>

namespace rl::mdp {
    /// Actions for the Gridworld
    using GridworldAction = FourWayAction;

    /// Class for representing the current state in Gridworld
    struct GridworldState {
        GridworldState() = default;
        GridworldState(size_t r, size_t c): row(r), column(c) {}

        size_t row, column;

        /// Comparison operator for ordering
        /// \param other
        /// \return
        auto operator<(const GridworldState &other) const {
            return row<other.row || (other.row >= row && column < other.column);
        }

        /// Equality operator
        /// \param other
        /// \return
        auto operator==(const GridworldState &other) const { return row == other.row && column == other.column; }

        /// Inequality operator
        /// \param other
        /// \return
        auto operator!=(const GridworldState& other) const { return !this->operator==(other); }
    };

    /// Represents a grid based MDP with transitions between cells
    class Gridworld: public MDP<GridworldState, GridworldAction> {
    public:
        /// Creates a new Gridworld with the given amount of rows and columns
        /// \param rows
        /// \param columns
        Gridworld(size_t rows, size_t columns)
        : m_rows(rows), m_columns(columns), m_cost_of_living{}, m_bounds_penalty{-1} { }

        /// SETTINGS ///

        /// Set cost of living parameter.
        /// \param cost_of_living
        void cost_of_living(Reward cost_of_living){ m_cost_of_living = cost_of_living; }

        /// Set the out-of-bounds penalty
        /// \param bounds_penalty
        void bounds_penalty(Reward bounds_penalty){ m_bounds_penalty = bounds_penalty; }

        /// MDP ///

        /// Returns the number of rows
        /// \return
        [[nodiscard]]
        size_t get_rows() const { return m_rows; }

        /// Returns the number of columns
        /// \return
        [[nodiscard]]
        size_t get_columns() const { return m_columns; }

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
        /// \param s_term State to mark as terminal
        /// \param default_reward Reward to use as default in-reward
        void set_terminal_state(const State& s_term,
                                std::optional<Reward> default_reward) override;

        /// Returns true if the given State is a terminal state.
        /// \param s
        /// \return
        [[nodiscard]]
        bool is_terminal_state(const State& s) const override;

        /// Returns a list of the terminal states.
        /// \return
        [[nodiscard]]
        std::vector<State> get_terminal_states() const override;

        /// Sets the given state as an initial state
        /// \param s
        void set_initial_state(const State& s) override{
            if(is_wall_state(s) || is_terminal_state(s))
                throw std::invalid_argument("Terminal or wall states cannot be marked as initial");
            m_initial_states.insert(s);
        }

        /// Returns if the given state is an initial state
        /// \param s
        /// \return
        [[nodiscard]]
        bool is_initial_state(const State& s) const override{
            return m_initial_states.find(s) != m_initial_states.cend();
        }

        /// Returns a list with the initial states
        /// \return
        [[nodiscard]]
        std::vector<State> get_initial_states() const override{
            return {m_initial_states.begin(), m_initial_states.end()};
        }

        /// Sets a states as wall. Meaning all transitions to this revert to previous state with the given penalty.
        /// \param wall
        /// \param penalty
        void set_wall_state(const State& wall, Reward penalty);

        /// Returns true if the state is a Wall
        /// \param s
        /// \return
        bool is_wall_state(const State& s) const{
            return m_wall_states.find(s) != m_wall_states.end();
        }

        /// Returns aa list with all states marked as walls
        /// \return
        [[nodiscard]]
        std::vector<State> get_wall_states() const{
            return {m_wall_states.begin(), m_wall_states.end() };
        }


    private:
        using DynamicsMap = std::multimap<StateAction, StateRewardProbability>;
        DynamicsMap m_dynamics;

        size_t m_rows, m_columns;
        Reward m_cost_of_living, m_bounds_penalty;

        std::set<State> m_terminal_states, m_initial_states, m_wall_states;

        /// Returns the default transition for the state-action pair
        /// \param state
        /// \param action
        /// \return
        [[nodiscard]]
        StateRewardProbability transition_default(const State &state, const Action &action) const;

        /// Removes added transitions between states (reverting to default)
        /// \param source
        /// \param action
        /// \param target
        void remove_added_transition(const State& source, const Action& action, const State& target);
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
        double policy_evaluation() override;

        /// Makes the policy greedy according to the value function
        /// \return True if the policy changed
        bool update_policy() override;

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



#endif //REINFORCEMENT_LEARNING_MDP_GRIDWORLD_H
