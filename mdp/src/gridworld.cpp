#include "../include/mdp/gridworld.h"

#include <numeric>
#include <algorithm>
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
        return srp_transition(element_iter->second);
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
        StateRewardProbability transition_prob{prob.second};
        return value + srp_probability(transition_prob);
    });

    // Calculate probability distribution
    std::uniform_real_distribution distribution(0., total_probability);
    Probability random_value = distribution(m_random_engine);

    // Select one of the actions
    Probability current_value = 0.0;
    for(auto iter=start_iter; iter != end_iter; ++iter){
        StateRewardProbability transition_probability = iter->second;
        current_value += srp_probability(transition_probability);

        if(random_value <= current_value)
            return srp_transition(transition_probability);
    }

    throw std::logic_error("Non-deterministic transition error");
}

void Gridworld::add_transition(const Gridworld::State &state, const Gridworld::Action &action,
                               const Gridworld::State &new_state, const Gridworld::Reward &reward,
                               const Gridworld::Probability &probability) {
    StateAction state_action{state, action};
    StateRewardProbability transition_probability{new_state, reward, probability};

    m_dynamics.emplace(state_action, transition_probability);
}

double Gridworld::expected_reward(const GridworldState &state, const GridworldAction &action) const {
    StateAction state_action{state, action};
    auto transition_count = m_dynamics.count(state_action);

    // Default reward
    if(transition_count == 0){
        return 0.0;
    }

    // Deterministic
    if(transition_count == 1){
        auto transition = m_dynamics.find(state_action);
        return srp_reward(transition->second);
    }

    // Non-deterministic
    Reward total_probability{};
    auto [start_iter, end_iter] = m_dynamics.equal_range(StateAction{state, action});
    Reward expected_reward = std::accumulate(start_iter, end_iter, 0.0,
                    [&total_probability](const auto& value, const auto& iter_value){
                        total_probability += srp_probability(iter_value.second);
        return value + srp_probability(iter_value.second) * srp_reward(iter_value.second);
    });

    return expected_reward / total_probability;
}

double Gridworld::state_transition_probability(const GridworldState &from_state,
                                               const GridworldAction &action,
                                               const GridworldState &to_state) const {
    // Get all transitions from state-action pair
    auto [start_iter, end_iter] = m_dynamics.equal_range(StateAction{from_state, action});

    // Check if it is a default probability or set state
    // ... default
    if(start_iter == m_dynamics.end()) {
        auto [default_state, default_reward] = transition_default(from_state, action);
        if (default_state == to_state) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    // ... set
    Probability total_probability = 0.0, to_state_probability = 0.0;
    for(auto iter = start_iter; iter != end_iter; ++iter){
        StateRewardProbability srp = iter->second;

        if(srp_state(srp) == to_state){
            to_state_probability += srp_probability(srp);
        }
        total_probability += srp_probability(srp);
    }

    // Calculate final probability
    if(total_probability != 0.0){
        return to_state_probability / total_probability;
    }

    return 0.0;
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