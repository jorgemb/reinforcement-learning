//
// Created by Jorge Martinez Bonilla on 10/05/2022.
//

#include "../include/mdp/gridworld.h"

#include <numeric>
#include <stdexcept>

using namespace rl::mdp;

Gridworld::Gridworld(size_t rows, size_t columns, RandomEngine::result_type seed) : m_rows(rows), m_columns(columns),
m_random_engine(seed){
    if(seed == std::numeric_limits<RandomEngine::result_type>::max()){
        m_random_engine.seed(std::random_device{}());
    }
}

size_t Gridworld::get_rows() const {
    return m_rows;
}

size_t Gridworld::get_columns() const {
    return m_columns;
}

Gridworld::Transition Gridworld::get_transition(const Gridworld::State &state, const Gridworld::Action &action) const {
    StateAction stateAction{state, action};
    size_t number_transitions = m_dynamics.count(stateAction);

    // Default case - No actions found for current state
    if(number_transitions == 0){
        return transition_default(state, action);
    }

    // Deterministic case
    if (number_transitions == 1){
        auto element_iter = m_dynamics.find(stateAction);
        auto [transition, probability] = element_iter->second;
        return transition;
    }

    // Non-deterministic case
    return transition_non_deterministic(state, action);
}

Gridworld::Transition
Gridworld::transition_default(const Gridworld::State &state, const Gridworld::Action &action) const {
    switch(action){
        case Action::LEFT:
            return Transition{
                State{state.row, state.column == 0 ? 0 : state.column-1},
                0.0};
        case Action::RIGHT:
            return Transition {
                State{state.row, state.column >= m_columns ? m_columns-1 : state.column+1},
                0.0};
        case Action::UP:
            return Transition {
                State{ state.row == 0 ? 0 : state.row-1, state.column},
                0.0};
        case Action::DOWN:
            return Transition {
                    State{ state.row >= m_rows ? m_rows-1 : state.row+1, state.column},
                    0.0};
    }

    throw std::logic_error("Invalid action selected");
}

Gridworld::Transition
Gridworld::transition_non_deterministic(const Gridworld::State &state, const Gridworld::Action &action) const {
    // Get range of options
    auto [start_iter, end_iter] = m_dynamics.equal_range(StateAction{state, action});
    Probability total_probability = std::accumulate(start_iter, end_iter, 0.0,
                                                    [](const auto& value, const auto& prob){
        TransitionProbability transition_prob{prob.second};
        return value + transition_prob.second;
    });

    // Calculate probability distribution
    std::uniform_real_distribution distribution(0., total_probability);
    Probability random_value = distribution(m_random_engine);

    // Select one of the actions
    Probability current_value = 0.0;
    for(auto iter=start_iter; iter != end_iter; ++iter){
        TransitionProbability transition_probability = iter->second;
        current_value += transition_probability.second;

        if(random_value <= current_value)
            return transition_probability.first;
    }

    throw std::logic_error("Non-deterministic transition error");
}

void Gridworld::add_transition(const Gridworld::State &state, const Gridworld::Action &action,
                               const Gridworld::State &new_state, const Gridworld::Reward &reward,
                               const Gridworld::Probability &probability) {
    StateAction state_action{state, action};
    Transition transition{new_state, reward};
    TransitionProbability transition_probability{transition, probability};

    m_dynamics.emplace(state_action, transition_probability);
}

std::ostream &operator<<(std::ostream &os, const Gridworld::Action& action) {
    using Action = Gridworld::Action;
    switch (action) {
        case Action::LEFT:
            os << "LEFT";
            break;
        case Action::RIGHT:
            os << "RIGHT";
            break;
        case Action::UP:
            os << "UP";
            break;
        case Action::DOWN:
            os << "DOWN";
            break;
    }

    return os;
}

std::ostream &operator<<(std::ostream &os, const Gridworld::State& state) {
    os << "(" << state.row << "," << state.column << ")";
    return os;
}