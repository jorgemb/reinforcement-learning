//
// Created by Jorge Martinez Bonilla on 10/05/2022.
//

#ifndef REINFORCEMENT_LEARNING_GRIDWORLD_H
#define REINFORCEMENT_LEARNING_GRIDWORLD_H

#include <map>
#include <tuple>
#include <random>
#include <limits>
#include <ostream>

class Gridworld {
public:
    enum class Action{
        LEFT,
        RIGHT,
        UP,
        DOWN
    };

    /// Class for representing the current state in Gridworld
    class State{
    public:
        size_t row, column;

        auto operator<(const State& other) const { return std::make_pair(row, column) < std::make_pair(other.row, other.column); }
        auto operator==(const State& other) const {return row == other.row && column == other.column; }
    };

    using Reward = int;
    using Probability = double;

    using StateAction = std::pair<State, Action>;
    using Transition = std::pair<State, Reward>;
    using TransitionProbability = std::pair<Transition, Probability>;
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
    Transition get_transition(const State& state, const Action& action) const;

    /// Adds a transition with the given probability
    /// \param state
    /// \param action
    /// \param new_state
    /// \param reward
    /// \param probability
    void add_transition(const State& state, const Action& action, const State& new_state, const Reward& reward,
                        const Probability& probability = 1.0);
private:
    using DynamicsMap = std::multimap<StateAction, TransitionProbability>;
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

/// Outputs the action of the Gridworld
/// \param os
/// \param action
/// \return
inline std::ostream& operator<<(std::ostream& os, Gridworld::Action action){
    using Action = Gridworld::Action;
    switch (action)  {
        case Action::LEFT: os << "LEFT"; break;
        case Action::RIGHT: os << "RIGHT"; break;
        case Action::UP: os << "UP"; break;
        case Action::DOWN: os << "DOWN"; break;
    }

    return os;
}

/// Outputs the current State
/// \param os
/// \param state
/// \return
inline std::ostream& operator<<(std::ostream& os, Gridworld::State state){
    os << "(" << state.row << "," << state.column << ")";
    return os;
}


#endif //REINFORCEMENT_LEARNING_GRIDWORLD_H
