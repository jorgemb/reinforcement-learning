#include <kbandit/k-bandit.h>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>

#include <fmt/core.h>

void do_test(const std::string& agent_name, KBanditsAgent* agent, unsigned int tests, KBandits& bandits) {
	std::cout << agent_name << " agent\n";

	std::vector<double> results(tests, 0.0);
	std::transform(results.begin(), results.end(), results.begin(), 
		[agent, &bandits](auto val) {
			std::size_t selection = agent->get_selection();
			double reward = bandits.get_reward(selection);
			agent->add_reward(selection, reward);

//            std::cout << reward << std::endl;
			
			return reward;
		});

	double total_reward = std::reduce(results.begin(), results.end(), 0.0);
	fmt::print("\tTotal reward: {}\n", total_reward);
	fmt::print("\tAverage reward: {}\n", total_reward / tests);
	fmt::print("\tBest bandit to agent: {}\n", agent->get_best_bandit());
}

int main() {
	std::cout << "Running K-Bandits tests \n";

	// Initialize
	const unsigned int tests = 1000;
	const unsigned int n_bandits = 10;
    const double initial_agent_estimate = 0.0;

	auto bandits = KBandits(0.0, 1.0, 1.0, n_bandits, 42);
	fmt::print("Best bandit: {}\n", bandits.best_bandit());
	for (size_t i = 0; i != n_bandits; i++) {
		Bandit& b = bandits.get_bandit(i);
		fmt::print("\t{0} :: {1:.3f} ({2:.3f})\n", i, b.mean_reward(), b.variance());
	}

	// Greedy
	BasicGreedyAgent greedy_agent(n_bandits, 0.0, initial_agent_estimate);
	do_test("Greedy", &greedy_agent, tests, bandits);

	// e-0.1
	BasicGreedyAgent e01_agent(n_bandits, 0.1, initial_agent_estimate);
	do_test("e-0.1", &e01_agent, tests, bandits);

	// e-0.01
	BasicGreedyAgent e001_agent(n_bandits, 0.01, initial_agent_estimate);
	do_test("e-0.01", &e001_agent, tests, bandits);
}