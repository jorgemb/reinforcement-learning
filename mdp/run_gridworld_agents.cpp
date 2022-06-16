#include <mdp/gridworld.h>
#include <mdp/agents.h>

#include <fmt/core.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>

#include <sciplot/sciplot.hpp>

#include <iostream>
#include <random>
#include <array>
#include <chrono>

using RandomEngine = std::default_random_engine;
using rl::mdp::Gridworld;

using Clock = std::chrono::system_clock;
using std::chrono::duration_cast;
using ms_duration = std::chrono::duration<double, std::milli>;

namespace plt = sciplot;

// Experiment parameters
const RandomEngine::result_type seed = 321;
constexpr size_t max_steps = std::numeric_limits<size_t>::max(), total_episodes = 100;

// Agents
using rl::mdp::GridworldState;
using rl::mdp::GridworldAction;
using RandomAgent = rl::mdp::BasicRandomAgent<GridworldState, GridworldAction>;
using MCAgent = rl::mdp::MCAgent<GridworldState, GridworldAction>;

/// Run a full experiment with a given agent
/// \tparam Agent
template<class Agent>
void run_basic_experiment(const std::shared_ptr<Gridworld>& gridworld,
                          const std::shared_ptr<Agent>& agent,
                          const std::string& experiment_name,
                          plt::Plot& plot){
    using Environment = rl::mdp::MDPEnvironment<Gridworld>;
    using Experiment = rl::mdp::MDPExperiment<Environment, Agent>;
    using Reward = Environment::Reward;

    // Create environment, agent and experiment
    Experiment experiment(max_steps);

    // Create accumulators
    namespace accum = boost::accumulators;
    accum::accumulator_set<size_t,
        accum::stats<accum::tag::mean, accum::tag::max, accum::tag::min, accum::tag::rolling_mean>
        > steps(accum::tag::rolling_mean::window_size = 10);
    accum::accumulator_set<Reward,
        accum::stats<accum::tag::mean, accum::tag::rolling_mean, accum::tag::max>
        > rewards(accum::tag::rolling_mean::window_size = 10);

    // Plot data for optional plot
    std::array<double, total_episodes> plot_data{};

    // Run each episode
    auto start_time = Clock::now();
    for(size_t current_episode=0; current_episode < total_episodes; ++current_episode){
        auto environment = std::make_shared<Environment>(gridworld, seed);
        auto results = experiment.do_episode(environment, agent);

        steps(results.total_steps);
        rewards(results.total_reward);
        plot_data[current_episode] = results.total_reward;
//        plot_data[current_episode] = accum::rolling_mean(rewards);
    }
    auto total_time = Clock::now() - start_time;

    // Show statistics
    fmt::print("{}\n", experiment_name);
    fmt::print("\tSteps -- min={}, max={}, avg={:.2f}, r_avg={:.2f}\n",
               accum::min(steps), accum::max(steps), accum::mean(steps), accum::rolling_mean(steps));
    fmt::print("\tReward -- max={}, avg={}, r_avg={:.2f}\n", accum::max(rewards), accum::mean(rewards), accum::rolling_mean(rewards));
    fmt::print("\tRunning time={} ms\n", duration_cast<ms_duration>(total_time).count());

    // Add plot
    plot.drawCurve(plt::range(0, total_episodes),
                   plot_data).label(experiment_name);
}

int main(){
    // Create Gridworld
    auto gridworld = std::make_shared<Gridworld>(4, 4);
//    gridworld->cost_of_living(-1.0);
    gridworld->bounds_penalty(-1.0);
    gridworld->set_initial_state({0, 0});
    gridworld->set_terminal_state({3, 3}, 1.0);

    rl::mdp::GridworldGreedyPolicy policy(gridworld, 1.0);
    while(policy.policy_evaluation() > 0.0001);
    fmt::print("Expected value from initial state: {:.2f}\n\n",
               policy.value_function({0, 0}));

    // Create plot
    plt::Plot plot;
    plot.palette("set2");
    plot.size(1024, 768);
    plot.xlabel("Episode");
    plot.ylabel("Reward");

    plot.yrange(-200, 10);

    // Run experiment
    run_basic_experiment<>(gridworld, std::make_shared<RandomAgent>(seed), "Random agent", plot);
    run_basic_experiment<>(gridworld, std::make_shared<MCAgent>(1.0, 0.1, seed), "MCAgent", plot);

    // Show plot
    plot.show();
}