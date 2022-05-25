#include "kbandit/k-bandit.h"
#include "kbandit/k-bandit-agent.h"

#include <random>
#include <numeric>
#include <algorithm>

std::default_random_engine create_random_engine(std::default_random_engine::result_type seed) {
	// Determine seed
	using Engine = std::default_random_engine;
	Engine engine(seed);
	if (seed == 0) {
		// Create a new seed
		std::random_device dev;
		engine.seed(dev());
	}

	return engine;
}


Bandit::Bandit(double reward, double variance, typename Engine::result_type seed)
	: m_reward(reward), m_variance(variance), m_distribution(reward, variance), m_generator(create_random_engine(seed)) {
}

double Bandit::operator()() {
    return m_distribution(m_generator);
}

double Bandit::mean_reward() const noexcept {
    return m_reward;
}

double Bandit::variance() const noexcept {
    return m_variance;
}

KBandits::KBandits(double reward_mean, 
	double reward_variance, 
	double bandit_variance, 
	std::size_t bandits, 
	Engine::result_type seed): m_best_bandit(-1) {
		
	// Create distribution for rewards
	std::default_random_engine engine{create_random_engine(seed)};
	std::normal_distribution distribution(reward_mean, std::sqrt(reward_variance));
	double best_reward = -std::numeric_limits<double>::infinity();

	for (std::size_t i = 0; i < bandits; i++) {
		double reward = distribution(engine);
		m_bandits.emplace_back(reward, bandit_variance, seed);

		// Set best
		if (reward > best_reward) {
			m_best_bandit = i;
			best_reward = reward;
		}
	}
}

double KBandits::get_reward(std::size_t index) {
	return m_bandits[index]();
}

Bandit& KBandits::get_bandit(std::size_t index) {
	return m_bandits[index];
}

std::size_t KBandits::get_num_bandits() const {
	return m_bandits.size();
}

std::size_t KBandits::best_bandit() const {
	return m_best_bandit;
}

std::vector<double> test_agent(KBandits& bandits, KBanditsAgent& agent, std::size_t total_runs){
    std::vector<double> results(total_runs, 0.);

    // Do each run for the agent
    std::transform(results.begin(), results.end(), results.begin(), [&bandits, &agent](auto val){
        auto selection = agent.get_selection();
        auto reward = bandits.get_reward(selection);
        agent.add_reward(selection, reward);

        return reward;
    });

    return results;
}
