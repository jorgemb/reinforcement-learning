#ifndef REINFORCEMENT_LEARNING_GRAPH_MDP_H
#define REINFORCEMENT_LEARNING_GRAPH_MDP_H

#include "mdp/mdp.h"
#include <boost/graph/adjacency_list.hpp>
#include <map>
#include <iterator>

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
        GraphMDP(std::initializer_list<Action> available_actions): m_available_actions(std::move(available_actions)) {}

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
            return {};
        }

        /// Probability of going to a given state from a state-action pair
        /// \param from_state
        /// \param action
        /// \param to_state
        /// \return
        Probability state_transition_probability(const State& from_state,
                                                         const Action& action,
                                                         const State& to_state) const override{
            return {};
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
            return;
        }

        /// Returns true if the given State is a terminal state.
        /// \param s
        /// \return
        bool is_terminal_state(const State& s) const override{
            return {};
        }

        /// Returns a list of the terminal states.
        /// \return
        std::vector<State> get_terminal_states() const override{
            return {};
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
        std::vector<Action> m_available_actions;

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

} // rl::mdp

#endif //REINFORCEMENT_LEARNING_GRAPH_MDP_H
