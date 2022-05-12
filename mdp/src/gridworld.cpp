#include "../include/mdp/gridworld.h"

#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iterator>

using namespace rl::mdp;

Gridworld::Gridworld(size_t rows, size_t columns) : m_rows(rows), m_columns(columns){ }

size_t Gridworld::get_rows() const {
    return m_rows;
}

size_t Gridworld::get_columns() const {
    return m_columns;
}

std::vector<Gridworld::StateRewardProbability> Gridworld::get_transitions(const Gridworld::State &state, const Gridworld::Action &action) const {
    StateAction stateAction{state, action};
    size_t number_transitions = m_dynamics.count(stateAction);

    // Default case - No actions found for current state
    if(number_transitions == 0){
        return { transition_default(state, action) };
    }

    // Deterministic case
    if (number_transitions == 1) {
        auto element_iter = m_dynamics.find(stateAction);
        return {element_iter->second};
    }

    // Non-deterministic case
    auto [start_iter, end_iter] = m_dynamics.equal_range(StateAction{state, action});
    auto total_probability = std::accumulate(start_iter, end_iter, 0.0, [](const auto& val, const auto& iter){
        return val + srp_probability(iter.second);
    });

    std::vector<StateRewardProbability> srp_list;
    std::transform(start_iter, end_iter, std::back_inserter(srp_list), [total_probability](const auto& iter){
        StateRewardProbability new_srp{iter.second};
        srp_probability(new_srp) /= total_probability;
        return new_srp;
    });
    return srp_list;
}

Gridworld::StateRewardProbability
Gridworld::transition_default(const Gridworld::State &state, const Gridworld::Action &action) const {
    StateRewardProbability srp;

    // Select according to action
    switch(action) {
        case Action::LEFT:
            srp = StateRewardProbability{
                    State{state.row, state.column == 0 ? 0 : state.column - 1},
                    0.0,
                    1.0};
            break;
        case Action::RIGHT:
            srp = StateRewardProbability{
                    State{state.row, state.column >= m_columns ? m_columns - 1 : state.column + 1},
                    0.0,
                    1.0};
            break;
        case Action::UP:
            srp = StateRewardProbability{
                    State{state.row == 0 ? 0 : state.row - 1, state.column},
                    0.0,
                    1.0};
            break;
        case Action::DOWN:
            srp = StateRewardProbability{
                    State{state.row >= m_rows ? m_rows - 1 : state.row + 1, state.column},
                    0.0,
                    1.0};
            break;
    }

    // If the new-state == the given state it means we are at the edge
    // so the reward is -1
    if( state == srp_state(srp)){
        srp_reward(srp) = -1.0;
    }

    return srp;
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
        auto [default_state, default_reward, default_probability] = transition_default(from_state, action);
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
    } else {
        return 0.0;
    }
}

std::vector<Gridworld::State> Gridworld::get_states() const {
    // Create the states vector and return
    std::vector<State> states;
    states.reserve(get_rows() * get_columns());

    for(size_t row=0; row != get_rows(); ++row){
        for(size_t col=0; col != get_columns(); ++col){
            states.emplace_back(row, col);
        }
    }

    return states;
}

std::vector<Gridworld::Action> Gridworld::get_actions(const GridworldState &state) const {
    return {AvailableGridworldActions.begin(), AvailableGridworldActions.end()};
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
