#include <kbandit/k-bandit.h>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>

void do_test(const std::string& agent_name, KBanditsAgent* agent, unsigned int tests, KBandits& bandits) {
	std::cout << agent_name << " agent\n";

	std::vector<double> results(tests, 0.0);
	std::transform(results.begin(), results.end(), results.begin(), 
		[agent, &bandits](auto val) {
			auto selection = agent->get_selection();
			auto reward = bandits.get_reward(selection);
			agent->add_reward(selection, reward);
			
			return reward;
		});

	double total_reward = std::reduce(results.begin(), results.end(), 0.0);
	std::cout << "\tTotal reward: " << total_reward << "\n";
	std::cout << "\tAverage reward: " << total_reward / tests << "\n";
	std::cout << "\tBest bandit to agent: " << agent->get_best_bandit() << std::endl;
}

int main() {
	std::cout << "Running K-Bandits tests \n";

	// Initialize
	const unsigned int tests = 10000;
	const unsigned int n_bandits = 10;

	auto bandits = KBandits(0.0, 10.0, 1.0, n_bandits);
	std::cout << "Best bandit: " << bandits.best_bandit() << std::endl;

	// Greedy
	BasicGreedyAgent greedy_agent(n_bandits, 0.0);
	do_test("Greedy", &greedy_agent, tests, bandits);

	// e-0.1
	BasicGreedyAgent e01_agent(n_bandits, 0.1);
	do_test("e-0.1", &e01_agent, tests, bandits);

	// e-0.01
	BasicGreedyAgent e001_agent(n_bandits, 0.01);
	do_test("e-0.01", &e001_agent, tests, bandits);
}