#ifndef REINFORCEMENT_LEARNING_MDP_H
#define REINFORCEMENT_LEARNING_MDP_H

#include <tuple>
#include <vector>
#include <memory>
#include <random>
#include <optional>
#include <algorithm>
#include <stdexcept>

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
        /// with zero reward. If default reward is provided, it is used for setting
        /// the reward of reaching this state from other states.
        /// \param s State to mark as terminal
        /// \param default_reward
        virtual void set_terminal_state(const State& s, std::optional<Reward> default_reward) = 0;

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

        /// Virtual destructor
        virtual ~MDPAgent() = default;
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

        /// Virtual destructor
        virtual ~MDPPolicy() = default;
    };

    /// Environment using an MDP as source of information
    /// \tparam MDP
    template<class MDP>
    class MDPEnvironment {
    public:
        // DEFINITIONS
        using State = typename MDP::State;
        using Action = typename MDP::Action;
        using Reward = typename MDP::Reward;
        using Probability = typename MDP::Probability;

        using RandomEngine = std::default_random_engine;

        /// Create the environment with the given MDP
        /// \param mdp
        /// \param seed Seed for the random generator, use 0 for a random one
        explicit MDPEnvironment(std::shared_ptr<MDP> mdp, RandomEngine::result_type seed = 0):
        m_mdp(mdp), m_random_engine(seed), m_random_distribution(0.0){
            // Check if a new seed is necessary
            if(seed == 0){
                m_random_engine.seed(std::random_device{}());
            }
        }

        /// Starts the environment and returns the initial state
        /// \return
        [[nodiscard]]
        virtual State start(){
            // Get initial states and check not empty
            auto initial_states = m_mdp->get_initial_states();
            auto num_states = initial_states.size();
            if(num_states == 0) {
                throw std::invalid_argument("MDP has not initial states");
            } else if(initial_states.size() == 1){
                // Return the only initial state
                m_last_state = *initial_states.begin();
            } else {
                // Return a random state
                std::sample(initial_states.begin(), initial_states.end(),
                            &m_last_state, 1,
                            m_random_engine);
            }

            return m_last_state;
        }


        /// Makes a step on the environment using the given action.
        /// \param action Action to perform next step
        /// \return Tuple with [next state, reward of current action, bool is_final]
        [[nodiscard]]
        virtual std::tuple<State, Reward, bool> step(const Action &action){
            // Initialize probability
            Probability accumulated_probability = 0;
            Probability target_probability = m_random_distribution(m_random_engine);

            // Get transitions from the MDP
            auto transitions = m_mdp->get_transitions(m_last_state, action);
            for(const auto& [s_i, r, p]: transitions){
                accumulated_probability += p;
                if(accumulated_probability >= target_probability){
                    m_last_state = s_i;
                    return std::make_tuple(s_i, r, m_mdp->is_terminal_state(s_i));
                }
            }

            throw std::range_error("Transition probability does not sum 1.0");
        }

        /// Virtual destructor
        virtual ~MDPEnvironment() = default;

    protected:
        std::shared_ptr<MDP> m_mdp;
        State m_last_state;

        RandomEngine m_random_engine;
        std::uniform_real_distribution<Probability> m_random_distribution;
    };


    /// Represents an experiment of an Agent traversing an Environment
    /// \tparam Environment
    /// \tparam Agent
    template <class Environment, class Agent>
    class MDPExperiment{
    public:
        // DEFINITIONS
        using State = typename Environment::State;
        using Action = typename Environment::Action;
        using Reward = typename Environment::Reward;
        using Probability = typename Environment::Probability;

        /// Results of an episode
        struct EpisodeResults{
            /// Default constructor for results
            EpisodeResults(): last_state{}, total_reward{}, total_steps{0}, reached_terminal_state{false} {}

            State last_state;
            Reward total_reward;

            size_t total_steps;
            bool reached_terminal_state;
        };

        /// Experiment constructor
        /// \param max_steps Maximum steps allowed for a single episode run
        explicit MDPExperiment(size_t max_steps): m_max_steps(max_steps) {}

        /// Performs an episode in the environment using the provided agent
        /// \param environment
        /// \param agent
        virtual EpisodeResults do_episode(std::shared_ptr<Environment> environment, std::shared_ptr<Agent> agent){
            EpisodeResults results;

            // Initialize environment
            bool is_terminal_state = false;
            results.last_state = environment->start();
            Action current_action = agent->start(results.last_state);

            // Perform step actions
            while(results.total_steps < m_max_steps && !is_terminal_state){
                // Environment
                auto [s_i, reward, is_terminal] = environment->step(current_action);
                results.last_state = s_i;
                results.total_reward += reward;
                is_terminal_state = is_terminal;

                // Agent
                if(is_terminal){
                    agent->end(reward);
                } else {
                    current_action = agent->step(reward, s_i);
                }

                ++results.total_steps;
            }

            // If the terminal state was reached it means the experiment ended correctly
            results.reached_terminal_state = is_terminal_state;

            return results;
        }

        /// Default destructor
        virtual ~MDPExperiment() = default;

    protected:
        size_t m_max_steps;
    };

} // namespace rl::mdp

#endif //REINFORCEMENT_LEARNING_MDP_H
