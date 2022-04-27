#include "kbandit/k-bandit.h"
#include "kbandit/k-bandit-agent.h"

#include <iostream>
#include <string>
#include <numeric>
#include <functional>
#include <algorithm>

#include <fmt/core.h>
#include <fmt/format.h>

#include <sciplot/sciplot.hpp>
namespace plt = sciplot;

std::vector<double> test_agent_episodes(const std::function<std::unique_ptr<KBandits>()>& bandits_generator,
                               const std::function<std::unique_ptr<KBanditsAgent>()>& agent_generator,
                               std::size_t runs_per_episode,
                               std::size_t total_episodes){
    std::vector<double> accumulator(runs_per_episode, 0.);

    // Perform each episode
    for(std::size_t episode=0; episode != total_episodes; ++episode){
        std::unique_ptr<KBandits> bandits = bandits_generator();
        std::unique_ptr<KBanditsAgent> agent = agent_generator();

        std::vector<double> results = test_agent(*bandits, *agent, runs_per_episode);
        std::transform(results.begin(), results.end(),
                       accumulator.begin(), accumulator.begin(),
                       [](double a, double b){
            return a + b;
        });
    }

    // Calculate average
    const auto total = static_cast<double>(total_episodes);
    std::transform(accumulator.begin(), accumulator.end(), accumulator.begin(), [total](auto val){
        return val / total;
    });

    return accumulator;
}

std::vector<double> running_average(const std::vector<double>& values, std::size_t window_size = 100){
//    const std::size_t steps = values.size() - window_size + 1;
//    std::vector<double> results(steps);
//
//    auto values_iter = values.cbegin();
//
//    for(auto results_iter = results.begin(); results_iter != results.end(); ++results_iter){
//        *results_iter = std::reduce(values_iter, values_iter + window_size) / static_cast<double>(window_size);
//        values_iter = std::next(values_iter);
//    }

    std::vector<double> results(values.size(), 0.);
    auto values_iter = values.cbegin();
    for(auto iter=results.begin(); iter != results.end(); ++iter){
        *iter = std::reduce(values.begin(), values_iter) / std::distance(values.begin(), values_iter);
        values_iter = std::next(values_iter);
    }

    return results;
}

int main() {
	std::cout << "Running K-Bandits tests \n";

    // Initial values
    const double reward_mean = 0.0, reward_variance = 1.0, bandit_variance = 1.0;
	const unsigned int tests = 2000, episodes = 200;
	const unsigned int n_bandits = 10;
    const double initial_agent_estimate = 0.0;

    // KBandits creator
    KBandits::Engine::result_type seed = 0;
    auto bandits_generator = [=, &seed](){
        return std::make_unique<KBandits>(reward_mean, reward_variance, bandit_variance, n_bandits, seed++);
    };

    plt::Plot plot;
    plot.palette("set2");
    plot.size(1024, 768);
    plot.xlabel("Time step");
    plot.ylabel(fmt::format("Average reward on {} episodes", episodes));

    auto time_x = plt::range(1, tests);

	// Greedy
    fmt::print("Greedy agent\n");
    auto greedy_agent_results = test_agent_episodes(bandits_generator, [=](){
        return std::unique_ptr<KBanditsAgent>( new BasicGreedyAgent(n_bandits, 0.0, initial_agent_estimate) );
    }, tests, episodes);
    plot.drawCurve(time_x, running_average(greedy_agent_results)).label("Greedy agent");

	// e-0.1
    fmt::print("e-0.1 agent\n");
    auto e01_agent_results = test_agent_episodes(bandits_generator, [=](){
        return std::unique_ptr<KBanditsAgent>( new BasicGreedyAgent(n_bandits, 0.1, initial_agent_estimate));
    }, tests, episodes);
    plot.drawCurve(time_x, running_average(e01_agent_results)).label("e0.1 agent");

	// e-0.01
    fmt::print("e-0.01 agent\n");
    auto e001_agent_results = test_agent_episodes(bandits_generator, [=](){
        return std::unique_ptr<KBanditsAgent>( new BasicGreedyAgent(n_bandits, 0.01, initial_agent_estimate));
    }, tests, episodes);
    plot.drawCurve(time_x, running_average(e001_agent_results)).label("e0.01 agent");

    plot.show();
}