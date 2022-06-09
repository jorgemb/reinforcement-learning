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

        /// Sets the given state as an initial state
        /// \param s
        virtual void set_initial_state(const State& s) = 0;

        /// Returns if the given state is an initial state
        /// \param s
        /// \return
        virtual bool is_initial_state(const State& s) const = 0;

        /// Returns a list with the initial states
        /// \return
        virtual std::vector<State> get_initial_states() const = 0;

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
        // DEFINITIONS
        using State = TState;
        using Action = TAction;
        using Reward = TReward;
        using Probability = TProbability;

        /// Returns the first action the agent takes according to the given initial state.
        /// \param initial_state
        /// \return
        virtual Action start(const State& initial_state) = 0;


        /// Given the reward of the previous action and the following state compute the next action
        /// \param reward Reward of the previous action
        /// \param next_state State after the previous action
        /// \return Next action to perform
        virtual Action step(const Reward& reward, const State& next_state) = 0;

        /// Called when entering the final state, provides the reward of the last action taken
        /// \param reward Reward of the previous action
        virtual void end(const Reward& reward) = 0;
    };


    /// Defines the basic functions for an stochastic policy in an MDP
    /// \tparam TState
    /// \tparam TAction
    /// \tparam TReward
    /// \tparam TProbability
    template <class TState, class TAction, class TReward=double, class TProbability=double>
    class MDPPolicy{
    public:
        // Definitions
        using State = TState;
        using Action = TAction;
        using Reward = TReward;
        using Probability = TProbability;

        using ActionProbability = std::pair<TAction, TProbability>;

        /// Approximates the value function doing a single policy evaluation.
        /// \param epsilon
        /// \return
        virtual double policy_evaluation() = 0;

        /// Makes the policy greedy according to the value function
        /// \return Returns true if the policy changed
        virtual bool update_policy() = 0;

        /// Return the possible actions and its probabilities based on the current state.
        /// \param state
        /// \return
        virtual std::vector<ActionProbability> get_action_probabilities(const State& state) const = 0;

        /// Returns the value function result given a state.
        /// \param state
        /// \return
        virtual Reward value_function(const State& state) const = 0;
    };

    template <class TState, class TAction, class TReward=double, class TProbability=double>
    class MDPEnvironment{
    public:
        // DEFINITIONS
        using State = TState;
        using Action = TAction;
        using Reward = TReward;
        using Probability = TProbability;



        /// Starts the environment and returns the initial state
        /// \return
        virtual State start() = 0;

        /// Makes a step on the environment using the given action.
        /// \param action Action to perform next step
        /// \param out_reward Return of reward obtained by the action
        /// \param out_next_state Next state after this action
        /// \return True if the next state is a terminal one
        virtual bool step(const Action& action, Reward& out_reward, State& out_next_state) = 0;
    };


    /// Template function to get a vector of the action space for a type.
    /// \tparam ActionType
    /// \return
    template<class ActionType>
    std::vector<ActionType> get_actions_list() = delete;
} // namespace rl::mdp

#endif //REINFORCEMENT_LEARNING_MDP_H
