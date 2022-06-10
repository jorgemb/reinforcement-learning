#ifndef REINFORCEMENT_LEARNING_GRAPH_POLICY_H
#define REINFORCEMENT_LEARNING_GRAPH_POLICY_H

namespace rl::mdp {

    /// Class to represent an stochastic policy for an MDP
    /// \tparam TState
    /// \tparam TAction
    template<class TState, class TAction>
    class GraphMDP_Greedy : rl::mdp::MDPPolicy<TState, TAction> {
    public:
        // Class definitions
        using typename rl::mdp::MDPPolicy<TState, TAction>::State;
        using typename rl::mdp::MDPPolicy<TState, TAction>::Action;
        using typename rl::mdp::MDPPolicy<TState, TAction>::Reward;
        using typename rl::mdp::MDPPolicy<TState, TAction>::Probability;
        using typename rl::mdp::MDPPolicy<TState, TAction>::ActionProbability;

        using PGraphMDP = std::shared_ptr<rl::mdp::GraphMDP<TState, TAction>>;

        /// Default constructor with pointer to graph
        /// \param graph_mdp
        /// \param gamma
        GraphMDP_Greedy(PGraphMDP graph_mdp, double gamma) : m_graph_mdp(graph_mdp), m_gamma(gamma) {
            // Add one for each element in the list
            for (const auto &state: graph_mdp->get_states()) {
                // Create action probabilities only for non terminal states

                if (!graph_mdp->is_terminal_state(state)) {
                    // Calculate initial probability
                    auto available_actions = graph_mdp->get_actions(state);
                    Probability probability = 1.0 / static_cast<Probability>(available_actions.size());
                    std::vector<ActionProbability> ap_vector;
                    std::transform(available_actions.begin(), available_actions.end(),
                                   std::back_inserter(ap_vector),
                                   [probability](const auto &a) {
                                       return ActionProbability{a, probability};
                                   });

                    m_state_action_map[state] = std::move(ap_vector);
                }

                // Value function for all states
                m_value_function[state] = Reward{};
            }
        }

        /// Return the possible actions and its probabilities based on the current state.
        /// \param state
        /// \return
        std::vector<ActionProbability> get_action_probabilities(const State &state) const override {
            return m_state_action_map.at(state);
        }

        /// Returns the value function result given a state.
        /// \param state
        /// \return
        Reward value_function(const State &state) const override {
            return m_value_function.at(state);
        }

        /// Approximates the value function doing a single policy evaluation.
        /// \param epsilon
        /// \return
        double policy_evaluation() override {
            // Copy value function
            auto value_function_copy(m_value_function);
            Reward delta{};

            // Iterate through states
            for (auto v_iter = value_function_copy.begin(); v_iter != value_function_copy.end(); ++v_iter) {
                auto [state, value] = *v_iter;

                // Do not iterate for terminal states
                if (m_graph_mdp->is_terminal_state(state)) continue;

                // Calculate new state value
                Reward new_value{};
                for (const auto &[action, probability]: m_state_action_map.at(state)) {
                    auto transitions = m_graph_mdp->get_transitions(state, action);
                    Reward action_value = std::transform_reduce(transitions.begin(), transitions.end(),
                                                                Reward{}, std::plus<>(),
                                                                [this](const auto &srp) {
                                                                    auto [s_i, r, p] = srp;
                                                                    return p * (r + m_gamma * m_value_function.at(s_i));
                                                                });

                    new_value += probability * action_value;
                }

                // Store and check change
                v_iter->second = new_value;
                delta = std::max(delta, std::abs(new_value - m_value_function.at(state)));
            }

            // Update the new value function
            m_value_function = std::move(value_function_copy);

            return delta;
        }

        /// Makes the policy greedy according to the value function
        /// \return
        bool update_policy() override {
            // Store if policy has changed
            bool policy_changed = false;

            // Greedify the policy
            for (auto &[state, action_prob_list]: m_state_action_map) {
                std::set<Action> max_actions;
                Reward max_value = -std::numeric_limits<Reward>::infinity();

                // Calculate new state value
                for (const auto &[action, probability]: action_prob_list) {
                    auto transitions = m_graph_mdp->get_transitions(state, action);
                    Reward action_value = std::transform_reduce(
                            transitions.begin(), transitions.end(),
                            Reward{}, std::plus<>(),
                            [this](const auto &srp) {
                                auto [s_i, r, p] = srp;
                                auto ret = p * (r + m_gamma * m_value_function.at(s_i));
                                return ret;
//                                return p * (r + m_gamma * m_value_function.at(s_i));
                            });

                    // Check if it is the best action
                    if (action_value > max_value) {
                        max_actions.clear();
                        max_actions.insert(action);
                        max_value = action_value;
                    } else if (action_value == max_value) {
                        max_actions.insert(action);
                    }
                }

                // Calculate new probability of actions
                auto action_prob_list_copy = action_prob_list;

                Probability new_probability = 1.0 / static_cast<Probability>(max_actions.size());
                std::transform(action_prob_list.begin(), action_prob_list.end(), action_prob_list.begin(),
                               [&max_actions, new_probability](const auto &action_prob) {
                                   auto [a, p] = action_prob;
                                   if (max_actions.find(a) == max_actions.end()) {
                                       return std::make_pair(a, 0.0);
                                   } else {
                                       return std::make_pair(a, new_probability);
                                   }
                               });

                if (!policy_changed && action_prob_list != action_prob_list_copy) policy_changed = true;
            }

            return policy_changed;
        }

    private:
        using ActionProbabilityList = std::vector<ActionProbability>;
        std::map<State, ActionProbabilityList> m_state_action_map;
        std::map<State, Reward> m_value_function;
        double m_gamma;

        PGraphMDP m_graph_mdp;
    };

} // Namespace rl::mdp

#include "graph.h"

#endif //REINFORCEMENT_LEARNING_GRAPH_POLICY_H
