#ifndef REINFORCEMENT_LEARNING_MDP_H
#define REINFORCEMENT_LEARNING_MDP_H

#include <tuple>
#include <vector>

namespace rl::mdp{
    template <class TState, class TAction, class TReward=double, class TProbability=double>
    class MDP{
    public:
        // Basic definitions
        using State = TState;
        using Action = TAction;
        using Reward = TReward;
        using Probability = TProbability;

        // Complex definitions
        using StateAction = std::pair<TState, TAction>;
        using Transition = std::pair<TState, TReward>;
        using StateRewardProbability = std::tuple<TState, TReward, TProbability>;

        // Virtual destructor
        virtual ~MDP() = default;

        // MDP functionality

        /// Returns a transitions from a State-Action pair
        /// \param state
        /// \param action
        /// \return
        [[nodiscard]]
        virtual std::vector<StateRewardProbability> get_transitions(const State& state, const Action& action) const = 0;

        /// Adds a transition with the given probability
        /// \param state
        /// \param action
        /// \param new_state
        /// \param reward
        /// \param weight
        virtual void add_transition(const State& state,
                                    const Action& action,
                                    const State& new_state,
                                    const Reward& reward,
                                    const Probability& weight) = 0;

        /// Calculates the expected reward of a given State-Action pair
        /// \param state
        /// \param action
        /// \return
        virtual Reward expected_reward(const State& state, const Action& action) const = 0;

        /// Probability of going to a given state from a state-action pair
        /// \param from_state
        /// \param action
        /// \param to_state
        /// \return
        virtual Probability state_transition_probability(const State& from_state,
                                                         const Action& action,
                                                         const State& to_state) const = 0;

        /// Returns a vector with all the possible states that the MDP can contain.
        /// \return
        virtual std::vector<State> get_states() const = 0;

        /// Marks a state as a terminal state. This makes all transitions out of this state to point to it again
        /// with the given reward.
        /// \param s
        virtual void set_terminal_state(const State& s, const Reward& default_reward) = 0;

        /// Returns true if the given State is a terminal state.
        /// \param s
        /// \return
        virtual bool is_terminal_state(const State& s) const = 0;

        /// Returns a list of the terminal states.
        /// \return
        virtual std::vector<State> get_terminal_states() const = 0;

        /// Returns a list with the available actions for a given state.
        /// \param state
        /// \return
        virtual std::vector<Action> get_actions(const State& state) const = 0;

        // Useful methods for extracting data from StateRewardProbability
        static State srp_state(const StateRewardProbability& srp){ return std::get<0>(srp); }
        static State& srp_state(StateRewardProbability& srp){ return std::get<0>(srp); }

        static Reward srp_reward(const StateRewardProbability& srp){ return std::get<1>(srp); }
        static Reward& srp_reward(StateRewardProbability& srp){ return std::get<1>(srp); }

        static Probability srp_probability(const StateRewardProbability& srp){ return std::get<2>(srp); }
        static Probability& srp_probability(StateRewardProbability& srp){ return std::get<2>(srp); }

        static Transition srp_transition(const StateRewardProbability& srp){
            return Transition{srp_state(srp), srp_reward(srp)};
        }
    };


    /// Defines an agent to traverse an MDP
    /// \tparam TState
    /// \tparam TAction
    /// \tparam TReward
    /// \tparam TProbability
    template <class TState, class TAction, class TReward=double, class TProbability=double>
    class MDPAgent{
    public:
        /// Returns the next action to take based on the internal information
        /// \return
        virtual TAction next_action() = 0;

        /// Adds the result of a transition from an action
        /// \param new_state
        /// \param reward
        virtual void add_transition_result(const TState& new_state, const TReward& reward) = 0;
    };


    template <class TState, class TAction, class TReward=double, class TProbability=double>
    class MDPPolicy{
    public:
        // Definitions
        using State = TState;
        using Action = TAction;
        using Reward = TReward;
        using Probability = TProbability;

        using ActionProbability = std::pair<TAction, TProbability>;


        /// Return the possible actions and its probabilities based on the current state.
        /// \param state
        /// \return
        virtual std::vector<ActionProbability> get_action_probabilities(const State& state) const = 0;

        /// Returns the value function result given a state.
        /// \param state
        /// \return
        virtual Reward value_function(const State& state) const = 0;
    };
} // namespace rl::mdp

#endif //REINFORCEMENT_LEARNING_MDP_H
