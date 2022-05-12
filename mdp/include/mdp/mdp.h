#ifndef REINFORCEMENT_LEARNING_MDP_H
#define REINFORCEMENT_LEARNING_MDP_H

#include <tuple>

namespace rl::mdp{
    template <class TState, class TAction, class TReward=double, class TProbability=double>
    class MDP{
    public:
        // Basic definitions
        typedef TState State;
        typedef TAction Action;
        typedef TReward Reward;
        typedef TProbability Probability;

        // Complex definitions
        typedef std::pair<State, Action> StateAction;
        typedef std::pair<State, Reward> Transition;
        typedef std::tuple<State, Reward, Probability> StateRewardProbability;

        // Virtual destructor
        virtual ~MDP() = default;

        // MDP functionality

        /// Returns a transition from a State-Action pair
        /// \param state
        /// \param action
        /// \return
        [[nodiscard]]
        virtual Transition get_transition(const State& state, const Action& action) const = 0;

        /// Adds a transition with the given probability
        /// \param state
        /// \param action
        /// \param new_state
        /// \param reward
        /// \param probability
        virtual void add_transition(const State& state,
                                    const Action& action,
                                    const State& new_state,
                                    const Reward& reward,
                                    const Probability& probability) = 0;

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

    protected:
        // Useful methods
        static State srp_state(const StateRewardProbability& srp){
            return std::get<0>(srp);
        }

        static Reward srp_reward(const StateRewardProbability& srp){
            return std::get<1>(srp);
        }

        static Probability srp_probability(const StateRewardProbability& srp){
            return std::get<2>(srp);
        }

        static Transition srp_transition(const StateRewardProbability& srp){
            return Transition{srp_state(srp), srp_reward(srp)};
        }
    };
}

#endif //REINFORCEMENT_LEARNING_MDP_H
