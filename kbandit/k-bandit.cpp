#include "kbandit/k-bandit.h"

#include <random>

KBandits::KBandits(double reward_mean, double reward_variance, double bandit_variance, std::size_t bandits) {
	// Create distribution for rewards
	std::default_random_engine engine;
	std::normal_distribution distribution(reward_mean, std::sqrt(reward_variance));
	double best_reward = -std::numeric_limits<double>::infinity();

	for (std::size_t i = 0; i < bandits; i++) {
		double reward = distribution(engine);
		m_bandits.emplace_back(reward, bandit_variance, engine.default_seed);

		// Set best
		if (reward > best_reward) {
			m_best_bandit = i;
			best_reward = reward;
		}
	}
}

double KBandits::get_reward(unsigned int k) {
	return m_bandits[k]();
}

Bandit<>& KBandits::get_bandit(unsigned int k) {
	return m_bandits[k];
}

std::size_t KBandits::get_num_bandits() const {
	return m_bandits.size();
}

std::size_t KBandits::best_bandit() const {
	return m_best_bandit;
}
