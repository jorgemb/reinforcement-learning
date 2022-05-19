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
                    State{state.row, state.column >= m_columns - 1 ? m_columns - 1 : state.column + 1},
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
                    State{state.row >= m_rows - 1 ? m_rows - 1 : state.row + 1, state.column},
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
                               const Gridworld::Probability &weight) {
    // Check if it is a terminal state
    if(is_terminal_state(state)) throw std::invalid_argument("Adding transition to terminal state");

    StateAction state_action{state, action};
    StateRewardProbability transition_probability{new_state, reward, weight};

    m_dynamics.emplace(state_action, transition_probability);
}

Gridworld::Reward Gridworld::expected_reward(const GridworldState &state, const GridworldAction &action) const {
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

Gridworld::Probability Gridworld::state_transition_probability(const GridworldState &from_state,
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

void Gridworld::set_terminal_state(const GridworldState &s, const Reward& default_reward) {
    // Check if it is already added
    if(is_terminal_state(s)) return;

    // Remove all transitions coming from this state, and add a single one that returns to the same state
    for(const auto& action: AvailableGridworldActions){
        auto [start, end] = m_dynamics.equal_range(StateAction{s, action});
        m_dynamics.erase(start, end);

        add_transition(s, action, s, default_reward, 1.0);
    }

    // Add the state to the terminal states list
    m_terminal_states.push_back(s);
}

bool Gridworld::is_terminal_state(const GridworldState &s) const {
    return std::find(m_terminal_states.cbegin(), m_terminal_states.cend(), s) != m_terminal_states.cend();
}

std::vector<Gridworld::State> Gridworld::get_terminal_states() const {
    return m_terminal_states;
}

GridworldGreedyPolicy::GridworldGreedyPolicy(std::shared_ptr<Gridworld> gridworld, double gamma):
m_gridworld(std::move(gridworld)),
m_rows(m_gridworld->get_rows()), m_columns(m_gridworld->get_columns()), m_gamma(gamma),
m_value_function_table(m_rows * m_columns, 0.0) {
    // Initialize the state action probability map
    for(const auto& state: m_gridworld->get_states()){
        auto actions = m_gridworld->get_actions(state);
        Probability starting_probability = 1.0 / static_cast<Probability>(actions.size());

        for(auto a: actions){
            m_state_action_probability_map[state][a] = starting_probability;
        }
    }
}

std::vector<GridworldGreedyPolicy::ActionProbability> GridworldGreedyPolicy::get_action_probabilities(const GridworldState &state) const {
    std::vector<ActionProbability> action_probability;
    std::transform(m_state_action_probability_map.at(state).cbegin(), m_state_action_probability_map.at(state).cend(),
                   std::back_inserter(action_probability), [](const auto& val){
        return ActionProbability{val.first, val.second};
    });

    return action_probability;
}

double GridworldGreedyPolicy::value_function(const GridworldState &state) const {
    return value_from_table(state);
}

double GridworldGreedyPolicy::policy_evaluation() {
    Probability delta = 0.0;
    auto states = m_gridworld->get_states();
    auto value_table_copy{ m_value_function_table };

    // Iterate on each state
    for(const auto& s: states){
        // Skip terminal states
        if(m_gridworld->is_terminal_state(s)) continue;

        Reward expected_value = 0.0;
        for(const auto& [a, p]: get_action_probabilities(s)){
            auto srp_list = m_gridworld->get_transitions(s, a);
            Reward expected_reward = std::accumulate(srp_list.cbegin(), srp_list.cend(), 0.0,
                                               [this](const auto& val, const auto& iter){
                auto [s_i, reward, probability] = iter;
                return val + probability * (reward + m_gamma * value_from_table(s_i));
            });

            expected_value += expected_reward * p;
        }

        auto [row, col] = s;
        value_table_copy[row * m_columns + col] = expected_value;
        delta = std::max(delta, std::abs(value_from_table(s) - expected_value));
    }

    m_value_function_table = std::move(value_table_copy);

    return delta;
}

void GridworldGreedyPolicy::update_policy() {
    // Iterate on each state-action
    for(const auto& s: m_gridworld->get_states()){
        Action best_action{};
        Probability best_action_reward = -std::numeric_limits<Probability>::infinity();

        for(const auto& a: m_gridworld->get_actions(s)) {
            auto srp_list = m_gridworld->get_transitions(s, a);
            Reward expected_reward = std::accumulate(srp_list.cbegin(), srp_list.cend(), 0.0,
                                                 [this](const auto &val, const auto &iter) {
                                                     auto [s_i, reward, probability] = iter;
                                                     return val +
                                                            probability * (reward + m_gamma * value_from_table(s_i));
                                                 });

            // Get best action
            if(expected_reward > best_action_reward){
                best_action = a;
                best_action_reward = expected_reward;
            }
        }

        // Set the probabilities to the best action
        // ToTry: This could use a softmax function to set the probabilities according to the expected reward of each action
        for(const auto& [action, probability]: m_state_action_probability_map[s]){
            m_state_action_probability_map[s][action] = action == best_action ? 1.0 : 0.0;
        }
    }
}

std::shared_ptr<Gridworld> GridworldGreedyPolicy::get_gridworld() const{
    return m_gridworld;
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

std::ostream &operator<<(std::ostream &os, const GridworldGreedyPolicy &greedy_policy) {
    auto gridworld = greedy_policy.get_gridworld();
    for (size_t row = 0; row < gridworld->get_rows(); ++row) {
        for (size_t col = 0; col < gridworld->get_columns(); ++col) {
            Gridworld::State s{row, col};
            auto actions = greedy_policy.get_action_probabilities(s);

            // Print the best action
            auto best_action = std::max_element(actions.cbegin(), actions.cend(),
                                                [](const auto& a, const auto& b){
                return a.second < b.second;
            });

            auto [action, probability] = *best_action;
            switch (action) {
                case rl::mdp::GridworldAction::LEFT:
                    os << '<';
                    break;
                case rl::mdp::GridworldAction::RIGHT:
                    os << '>';
                    break;
                case rl::mdp::GridworldAction::UP:
                    os << '^';
                    break;
                case rl::mdp::GridworldAction::DOWN:
                    os << 'v';
                    break;
            }
        }
        // Add newline at the end of the row
        os << '\n';
    }

    return os;
}
