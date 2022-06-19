#ifndef REINFORCEMENT_LEARNING_AGENTS_H
#define REINFORCEMENT_LEARNING_AGENTS_H

#include <mdp/gridworld.h>
#include <mdp/mdp.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include <random>
#include <vector>
#include <utility>
#include <map>
#include <numeric>
#include <algorithm>

namespace rl::mdp {

    /// Represents a basic agent with a random policy
    template<class TState, class TAction>
    class BasicRandomAgent: public MDPAgent<TState, TAction> {
    public:
        using RandomEngine = std::default_random_engine;
        using Reward = typename MDPAgent<TState, TAction>::Reward;
        using Actions = ActionTraits<TAction>;

        /// Default constructor
        /// \param seed Seed for generator, use 0 for random seed
        explicit BasicRandomAgent(RandomEngine::result_type seed = 0): m_random_engine(seed),
                                                                       m_distribution(0, Actions::available_actions().size()-1),
                                                                       m_total_reward{0}{
            // Check if seed should be changed
            if(seed == 0){
                m_random_engine.seed(std::random_device{}());
            }
        }

        /// Returns the first action the agent takes according to the given initial state.
        /// \param initial_state
        /// \return
        TAction start(const TState& initial_state) override{
            // Initialize reward
            m_total_reward = {};

            // Choose a random action
            return Actions::available_actions()[m_distribution(m_random_engine)];
        }

        /// Given the reward of the previous action and the following state compute the next action
        /// \param reward Reward of the previous action
        /// \param next_state State after the previous action
        /// \return Next action to perform
        TAction step(const Reward& reward, const TState& next_state) override{
            m_total_reward += reward;

            // Choose a random action
            return Actions::available_actions()[m_distribution(m_random_engine)];
        }

        /// Called when entering the final state, provides the reward of the last action taken
        /// \param reward Reward of the previous action
        void end(const Reward& reward) override{
            m_total_reward += reward;
        }

        /// Returns the total reward at the moment
        /// \return
        [[nodiscard]]
        Reward get_reward() const { return m_total_reward; }

    protected:
        RandomEngine m_random_engine;
        std::uniform_int_distribution<uint32_t> m_distribution;

        Reward m_total_reward;
    };

    /// Basic agent policy that
    /// \tparam TState
    /// \tparam TAction
    /// \tparam TValue
    template<class TState, class TAction, class TValue=double>
    class BasicAgentPolicy{
    public:
        using RandomEngine = std::default_random_engine;

        /// Initializes the Policy
        /// \param seed Seed for the random generator, 0 if random seed
        explicit BasicAgentPolicy(RandomEngine::result_type seed = 0): m_random_engine(seed),
        m_action_distribution(0, Actions::total_actions() - 1){
            if(seed == 0){
                m_random_engine.seed(std::random_device{}());
            }
        }

        /// Returns the value of a state action pair
        /// \param state
        /// \param action
        /// \return
        TValue& value(const TState& state, const TAction& action){
            return m_value_function[state][Actions::id(action)];
        }

        /// Returns the valie of a state action pair, sent as pair
        /// \param state_action
        /// \return
        TValue& value(const std::pair<TState, TAction>& state_action){
            const auto& [state, action] = state_action;
            return value(state, action);
        }

        /// Returns the best action for the state
        /// \param state
        /// \return
        TAction best_action(const TState& state){
            auto& actions = m_value_function[state];
            auto best_iter = std::max_element(actions.cbegin(), actions.cend());

            return Actions::from_id(std::distance(actions.cbegin(), best_iter));
        }

        /// Selects the best action using an e-soft policy
        /// \param state
        /// \return
        TAction best_action_e(const TState& state, double epsilon){
            std::bernoulli_distribution dist(epsilon);

            // Check if exploratory or greedy step will be done
            bool do_explore = dist(m_random_engine);
            if(do_explore){
                auto action = Actions::available_actions()[m_action_distribution(m_random_engine)];
                return action;
            } else {
                return best_action(state);
            }
        }

    private:
        using Actions = ActionTraits<TAction>;
        using ActionArray = std::array<double, ActionTraits<TAction>::total_actions()>;
        std::map<TState, ActionArray> m_value_function;

        RandomEngine m_random_engine;
        std::uniform_int_distribution<size_t> m_action_distribution;
    };

    /// Agent that implements the MonteCarlo approach to learning
    /// \tparam TState
    /// \tparam TAction
    template<class TState, class TAction>
    class MCAgent: public MDPAgent<TState, TAction>{
    public:
        using Reward = typename MDPAgent<TState, TAction>::Reward;
        using Policy = BasicAgentPolicy<TState, TAction>;
        using RandomEngine = typename Policy::RandomEngine;

        explicit MCAgent(double gamma = 1.0, double epsilon = 0.1, typename RandomEngine::result_type seed = 0):
        m_gamma(gamma), m_epsilon(epsilon), m_policy(seed) {}

        TAction start(const TState &initial_state) override {
            // Select initial action
            auto action = m_policy.best_action_e(initial_state, m_epsilon);

            // Restart episode information
            m_episode_run.clear();
            m_is_first_visit.clear();
            m_state_action_visited.clear();

            // Add information of this step
            m_episode_run.push_back({initial_state, action, 0.0});

            StateAction state_action{initial_state, action};
            m_is_first_visit.push_back(true);
            m_state_action_visited.insert(state_action);

            return action;
        }

        TAction step(const Reward &reward, const TState &next_state) override {
            // Select action and add episode information
            auto action = m_policy.best_action_e(next_state, m_epsilon);
            m_episode_run.push_back({next_state, action, reward});

            // Check if it is the first visit
            StateAction state_action{next_state, action};
            bool is_first_visit = m_state_action_visited.find(state_action) == m_state_action_visited.end();
            m_is_first_visit.push_back(is_first_visit);
            m_state_action_visited.insert(state_action);

            return action;
        }

        void end(const Reward &reward) override {
            m_episode_run.push_back({TState{}, TAction{}, reward});

            // Learn from episode information
            Reward total_return = 0.0;  // G

            // The last episode is the second to last, as the last only contains Reward info
            long last_episode = m_episode_run.size()-2;
            for(long idx = last_episode; idx >= 0; --idx){
                const size_t ID_STATE=0, ID_ACTION=1, ID_REWARD=2;
                auto& current_step = m_episode_run[idx];

                total_return = m_gamma * total_return + std::get<ID_REWARD>(m_episode_run[idx+1]);

                // Check if first visit
                if(m_is_first_visit[idx]){
                    StateAction state_action{std::get<ID_STATE>(current_step), std::get<ID_ACTION>(current_step)};
                    m_returns[state_action](total_return);

                    m_policy.value(state_action) = boost::accumulators::mean(m_returns[state_action]);
                }
            }
        }

    private:
        double m_gamma, m_epsilon;

        // Policy information
        Policy m_policy;

        // Episode memory
        using StateActionReward = std::tuple<TState, TAction, Reward>;
        using StateAction = std::pair<TState, TAction>;
        std::vector<StateActionReward> m_episode_run;

        // First visit
        // If the StateAction has been visited
        std::set<StateAction> m_state_action_visited;

        // Where has the StateAction was first visited
        std::vector<bool> m_is_first_visit;

        // Returns information
        using ReturnsInfo = boost::accumulators::accumulator_set<Reward,
            boost::accumulators::stats<boost::accumulators::tag::mean>>;
        std::map<StateAction, ReturnsInfo> m_returns;
    };

    /// Agent that implements the TD(0) approach to learning
    /// \tparam TState
    /// \tparam TAction
    template<class TState, class TAction>
    class TD0Agent: public MDPAgent<TState, TAction>{
    public:
        using typename MDPAgent<TState, TAction>::Reward;
        using RandomEngine = typename BasicAgentPolicy<TState, TAction>::RandomEngine;

        explicit TD0Agent(double alpha = 0.2,
                          double gamma = 1.0,
                          double epsilon = 0.1,
                          typename RandomEngine::result_type seed = 0)
        : m_gamma(gamma), m_epsilon(epsilon), m_policy(seed), m_alpha(alpha)
        {}

        TAction start(const TState &initial_state) override {
            m_last_state = initial_state;
            m_last_action = m_policy.best_action_e(initial_state, m_epsilon);

            return m_last_action;
        }

        TAction step(const Reward &reward, const TState &next_state) override {
            TAction next_action = m_policy.best_action_e(next_state, m_epsilon);

            Reward value = m_alpha * (reward + m_gamma * m_policy.value(next_state, next_action) - m_policy.value(m_last_state, m_last_action));
            m_policy.value(m_last_state, m_last_action) += value;

            m_last_state = next_state;
            m_last_action = next_action;

            return next_action;
        }

        void end(const Reward &reward) override {
            Reward value = m_alpha * (reward - m_policy.value(m_last_state, m_last_action));
            m_policy.value(m_last_state, m_last_action) += value;
        }

    private:
        double m_epsilon, m_gamma, m_alpha;
        BasicAgentPolicy<TState, TAction> m_policy;

        TState m_last_state;
        TAction m_last_action;
    };

} // namespace rl::mdp

#endif //REINFORCEMENT_LEARNING_AGENTS_H
