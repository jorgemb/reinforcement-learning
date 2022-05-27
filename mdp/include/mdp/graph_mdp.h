#ifndef REINFORCEMENT_LEARNING_GRAPH_MDP_H
#define REINFORCEMENT_LEARNING_GRAPH_MDP_H

#include "mdp/mdp.h"
#include <boost/graph/adjacency_list.hpp>
#include <map>
#include <iterator>
#include <memory>
#include <numeric>
#include <functional>

namespace rl::mdp {
    /// Class for representing an MDP that uses a graph as underlying type
    /// \tparam TState
    /// \tparam TAction
    template <class TState, class TAction>
    class GraphMDP: public MDP<TState, TAction> {
    public:
        using typename MDP<TState, TAction>::State;
        using typename MDP<TState, TAction>::Action;
        using typename MDP<TState, TAction>::Reward;
        using typename MDP<TState, TAction>::Probability;

        using typename MDP<TState, TAction>::StateAction;
        using typename MDP<TState, TAction>::StateRewardProbability;

        /// Constructor accepting list of available actions
        /// \param available_actions
        GraphMDP(): m_available_actions(get_actions_list<TAction>()) {}

        /// Returns a transitions from a State-Action pair
        /// \param state
        /// \param action
        /// \return
        [[nodiscard]]
        std::vector<StateRewardProbability> get_transitions(const State& state, const Action& action) const override{
            GraphVertex v = m_state_to_vertex.at(state);
            auto[iter, end] = boost::out_edges(v, m_dynamics);

            // Return the transitions that match the given action
            std::vector<StateRewardProbability> ret;
            Probability total_probability{0.0};
            while(iter != end){
                if(m_dynamics[*iter].action == action){
                    GraphVertex target = boost::target(*iter, m_dynamics);
                    total_probability += m_dynamics[*iter].probability;

                    ret.push_back(StateRewardProbability{
                        m_dynamics[target].state,
                        m_dynamics[*iter].reward,
                        m_dynamics[*iter].probability
                    });
                }
                iter = std::next(iter);
            }

            // Normalize probabilities
            std::transform(ret.begin(), ret.end(), ret.begin(), [total_probability](const auto& srp){
                auto [s, r, p] = srp;
                return StateRewardProbability{s, r, p/total_probability};
            });

            return std::move(ret);
        }

        /// Adds a transition with the given probability
        /// \param state
        /// \param action
        /// \param new_state
        /// \param reward
        /// \param probability
        void add_transition(const State& state,
                                    const Action& action,
                                    const State& new_state,
                                    const Reward& reward,
                                    const Probability& weight) override{
            // Cannot add transition to terminal states
            if(is_terminal_state(state)) throw std::invalid_argument("Adding transition to terminal state");

            // Get vertices
            GraphVertex A = get_or_create_vertex(state);
            GraphVertex B = get_or_create_vertex(new_state);

            // Create the new transition
            boost::add_edge(A, B, {action, reward, weight}, m_dynamics);
        }

        /// Calculates the expected reward of a given State-Action pair
        /// \param state
        /// \param action
        /// \return
        Reward expected_reward(const State& state, const Action& action) const override{
            Reward expected_reward{};
            for(const auto& [s, r, p]: get_transitions(state, action)){
                expected_reward += r * p;
            }

            return expected_reward;
        }

        /// Probability of going to a given state from a state-action pair
        /// \param from_state
        /// \param action
        /// \param to_state
        /// \return
        Probability state_transition_probability(const State& from_state,
                                                         const Action& action,
                                                         const State& to_state) const override{
            Probability prob{};
            for(const auto& [s, r, p]: get_transitions(from_state, action)){
                if(s == to_state){
                    prob += p;
                }
            }

            return prob;
        }

        /// Returns a vector with all the possible states that the MDP can contain.
        /// \return
        std::vector<State> get_states() const override{
            std::vector<State> states;
            std::transform(m_state_to_vertex.cbegin(), m_state_to_vertex.cend(), std::back_inserter(states),
                           [](const auto& iter){
                return iter.first;
            });

            return states;
        }

        /// Marks a state as a terminal state. This makes all transitions out of this state to point to it again
        /// with the given reward.
        /// \param s
        void set_terminal_state(const State& s, const Reward& default_reward) override{
            // Initial check
            if(is_terminal_state(s)) return;

            // Remove all outgoing transitions and set only transitions to itself
            auto v = m_state_to_vertex.at(s);
            boost::clear_out_edges(v, m_dynamics);
            for(const auto& a: m_available_actions){
                add_transition(s, a, s, default_reward, 1.0);
            }

            // Add to list of terminal states
            m_terminal_states.insert(s);
        }

        /// Returns true if the given State is a terminal state.
        /// \param s
        /// \return
        bool is_terminal_state(const State& s) const override{
            return m_terminal_states.find(s) != m_terminal_states.end();
        }

        /// Returns a list of the terminal states.
        /// \return
        std::vector<State> get_terminal_states() const override{
            return {m_terminal_states.begin(), m_terminal_states.end()};
        }

        /// Returns a list with the available actions for a given state.
        /// \param state
        /// \return
        std::vector<Action> get_actions(const State& state) const override{
            // Get vertex
            GraphVertex v = m_state_to_vertex.at(state);
            std::set<Action> available_actions; // Use a set to avoid duplicates
            auto[iter, end] = boost::out_edges(v, m_dynamics);
            std::transform(iter, end, std::inserter(available_actions, available_actions.begin()),
                           [this](const GraphEdge& e) {
                               return m_dynamics[e].action;
                           });

            return {available_actions.begin(), available_actions.end()};
        }
    protected:
        // Graph definitions
        struct VertexProperties{
            State state;
        };

        struct EdgeProperties{
            Action action;
            Reward reward;
            Probability probability;
        };


        // .. Graph
        using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperties, EdgeProperties>;
        using GraphVertex = typename Graph::vertex_descriptor;
        using GraphEdge = typename Graph::edge_descriptor;

        // Internal data
        Graph m_dynamics;
        std::map<State, GraphVertex> m_state_to_vertex;
        const std::vector<Action> m_available_actions;
        std::set<State> m_terminal_states;

    private:
        /// Gets or creates a new vertex in the graph, maintaining the state-vertex map
        /// \param s
        /// \return
        GraphVertex get_or_create_vertex(const State& s){
            auto iter = m_state_to_vertex.find(s);
            if(iter == m_state_to_vertex.end() ){
                // Create a new one
                GraphVertex v = boost::add_vertex(m_dynamics);
                m_dynamics[v].state = s;
                m_state_to_vertex[s] = v;
                return v;
            } else {
                return iter->second;
            }
        }

    };

    template <class TState, class TAction>
    class GraphMDP_Greedy: MDPPolicy<TState, TAction>{
    public:
        // Class definitions
        using typename MDPPolicy<TState, TAction>::State;
        using typename MDPPolicy<TState, TAction>::Action;
        using typename MDPPolicy<TState, TAction>::Reward;
        using typename MDPPolicy<TState, TAction>::Probability;
        using typename MDPPolicy<TState, TAction>::ActionProbability;

        using PGraphMDP = std::shared_ptr<GraphMDP<TState, TAction>>;

        /// Default constructor with pointer to graph
        /// \param graph_mdp
        /// \param gamma
        GraphMDP_Greedy(PGraphMDP graph_mdp, double gamma): m_graph_mdp(graph_mdp), m_gamma(gamma) {
            auto actions = get_actions_list<Action>();
            Probability default_probability = 1.0 / static_cast<Probability>(actions.size());

            // Create default action_probability list
            std::vector<ActionProbability> default_action_probability;
            std::transform(actions.cbegin(), actions.cend(), std::back_inserter(default_action_probability),
                           [default_probability](const auto& a){
                return ActionProbability{a, default_probability};
            });

            // Add one for each element in the list
            for(const auto& state: graph_mdp->get_states()){
                // Create action probabilities only for non terminal states
                if(!graph_mdp->is_terminal_state(state)) {
                    m_state_action_map[state] = default_action_probability;
                }

                // Value function for all states
                m_value_function[state] = Reward{};
            }
        }

        /// Return the possible actions and its probabilities based on the current state.
        /// \param state
        /// \return
        std::vector<ActionProbability> get_action_probabilities(const State& state) const override{
            return m_state_action_map.at(state);
        }

        /// Returns the value function result given a state.
        /// \param state
        /// \return
        Reward value_function(const State& state) const override{
            return m_value_function.at(state);
        }

        /// Approximates the value function doing a single policy evaluation.
        /// \param epsilon
        /// \return
        double policy_evaluation() override{
            // Copy value function
            auto value_function_copy(m_value_function);
            Reward max_change{};

            // Iterate through states
            for(auto v_iter=value_function_copy.begin(); v_iter != value_function_copy.end(); ++v_iter){
                auto [state, value] = *v_iter;

                // Do not iterate for terminal states
                if(m_graph_mdp->is_terminal_state(state)) continue;

                // Calculate new state value
                Reward new_value{};
                for(const auto& [action, probability]: m_state_action_map.at(state)){
                    auto transitions = m_graph_mdp->get_transitions(state, action);
                    Reward action_value = std::transform_reduce(transitions.begin(), transitions.end(), new_value, std::plus<Reward>(),
                                          [this](const auto& srp){
                        auto [s_i, r, p] = srp;
                        return p * (r + m_gamma * m_value_function.at(s_i));
                    });

                    new_value += probability * action_value;
                }

                // Store and check change
                v_iter->second = new_value;
                max_change = std::max(max_change, std::abs(new_value - m_value_function.at(state)));
            }

            // Update the new value function
            m_value_function = std::move(value_function_copy);

            return max_change;
        }

        /// Makes the policy greedy according to the value function
        /// \return
        void update_policy() override{
            // Greedify the policy
            for(auto& [state, action_prob_list]: m_state_action_map){
                std::set<Action> max_actions;
                Reward max_value = -std::numeric_limits<Reward>::infinity();

                // Calculate new state value
                for(const auto& [action, probability]: action_prob_list){
                    auto transitions = m_graph_mdp->get_transitions(state, action);
                    Reward action_value = std::transform_reduce(transitions.begin(), transitions.end(), 0.0, std::plus<Reward>(),
                                                                [this](const auto& srp){
                                                                    auto [s_i, r, p] = srp;
                                                                    return p * (r + m_gamma * m_value_function.at(s_i));
                                                                });

                    // Check if it is the best action
                    if(action_value > max_value){
                        max_actions.clear();
                        max_actions.insert(action);
                        max_value = action_value;
                    } else if(action_value == max_value){
                        max_actions.insert(action);
                    }
                }

                // Calculate new probability of actions
                Probability new_probability = 1.0 / static_cast<Probability>(max_actions.size());
                std::transform(action_prob_list.begin(), action_prob_list.end(), action_prob_list.begin(),
                               [&max_actions, new_probability](const auto& action_prob){
                    auto [a, p] = action_prob;
                    if(max_actions.find(a) == max_actions.end()){
                        return std::make_pair(a, 0.0);
                    } else {
                        return std::make_pair(a, new_probability);
                    }
                });
            }
        }

    private:
        using ActionProbabilityList = std::vector<ActionProbability>;
        std::map<State, ActionProbabilityList> m_state_action_map;
        std::map<State, Reward> m_value_function;
        double m_gamma;

        PGraphMDP m_graph_mdp;
    };

} // rl::mdp

#endif //REINFORCEMENT_LEARNING_GRAPH_MDP_H
