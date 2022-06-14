#include <mdp/gridworld.h>
#include <mdp/agents.h>

#include <fmt/core.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include <iostream>
#include <random>
#include <array>
#include <chrono>

using RandomEngine = std::default_random_engine;
using rl::mdp::Gridworld;
using Clock = std::chrono::system_clock;
using std::chrono::duration_cast;
using ms_duration = std::chrono::duration<double, std::milli>;

// Experiment parameters
const RandomEngine::result_type seed = 0;
constexpr size_t max_steps = 1000, total_episodes = 200;

/// Run a full experiment with a given agent
/// \tparam Agent
template<class Agent>
void run_basic_experiment(const std::shared_ptr<Gridworld>& gridworld){
    using Environment = rl::mdp::MDPEnvironment<Gridworld>;
    using Experiment = rl::mdp::MDPExperiment<Environment, Agent>;
    using Reward = Environment::Reward;

    // Create environment, agent and experiment
    auto environment = std::make_shared<Environment>(gridworld, seed);
    auto agent = std::make_shared<Agent>();
    Experiment experiment(max_steps);

    // Create accumulators
    namespace accum = boost::accumulators;
    accum::accumulator_set<size_t, accum::stats<accum::tag::mean, accum::tag::max, accum::tag::min>> steps;
    accum::accumulator_set<Reward, accum::stats<accum::tag::mean>> rewards;

    // Run each episode
    auto start_time = Clock::now();
    for(size_t current_episode=0; current_episode < total_episodes; ++current_episode){
        auto results = experiment.do_episode(environment, agent);

        steps(results.total_steps);
        rewards(results.total_reward);
    }
    auto total_time = Clock::now() - start_time;

    // Show statistics
    fmt::print("Steps -- min={}, max={}, avg={}\n",
               accum::min(steps), accum::max(steps), accum::mean(steps));
    fmt::print("Reward -- avg={}\n", accum::mean(rewards));
    fmt::print("\tRunning time -- {} ms\n", duration_cast<ms_duration>(total_time).count());
}

int main(){
    // Create Gridworld
    auto gridworld = std::make_shared<Gridworld>(4, 4);
    gridworld->set_initial_state({0, 0});
    gridworld->set_terminal_state({3, 3}, 0.0);

    rl::mdp::GridworldGreedyPolicy policy(gridworld, 1.0);
    while(policy.policy_evaluation() > 0.0001);
    fmt::print("Expected value from initial state: {:.2f}\n\n",
               policy.value_function({0, 0}));

    // Run experiment
    fmt::print("BasicGridworldAgent\n");
    run_basic_experiment<rl::mdp::BasicGridworldAgent>(gridworld);
}