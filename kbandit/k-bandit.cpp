#include "kbandit/k-bandit.h"

#include <random>
#include <numeric>
#include <algorithm>

KBandits::KBandits(double reward_mean, double reward_variance, double bandit_variance, std::size_t bandits): m_best_bandit(-1) {
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

double KBandits::get_reward(std::size_t k) {
	return m_bandits[k]();
}

Bandit<>& KBandits::get_bandit(std::size_t k) {
	return m_bandits[k];
}

std::size_t KBandits::get_num_bandits() const {
	return m_bandits.size();
}

std::size_t KBandits::best_bandit() const {
	return m_best_bandit;
}

KBanditsAgent::KBanditsAgent(std::size_t total_bandits): m_total_bandits(total_bandits) {
}

std::size_t KBanditsAgent::total_bandits() const {
	return m_total_bandits;
}


BasicGreedyAgent::BasicGreedyAgent(std::size_t bandits, double epsilon): 
	KBanditsAgent(bandits), m_epsilon(epsilon), m_bandit_distribution(0, bandits-1), m_greedy_option_distribution(1-epsilon),
	m_rewards(bandits), m_expected_rewards(bandits, std::numeric_limits<double>::infinity())
{
}

std::size_t BasicGreedyAgent::get_selection() const{
	if (do_greedy()) {
		return get_best_bandit();
	}
	else {
		// Select a random option
		return m_bandit_distribution(m_engine);
	}
}

std::size_t BasicGreedyAgent::get_best_bandit() const {
	auto iter_element = std::max_element(m_expected_rewards.cbegin(), m_expected_rewards.cend());
	return std::distance(m_expected_rewards.cbegin(), iter_element);
}

void BasicGreedyAgent::add_reward(std::size_t selection, double reward) {
	// Push and update expected rewards
	m_rewards[selection].push_back(reward);
	m_expected_rewards[selection] = std::reduce(m_rewards[selection].cbegin(), m_rewards[selection].cend(), 0.0) / m_rewards[selection].size();
}

bool BasicGreedyAgent::do_greedy() const{
	return m_greedy_option_distribution(m_engine);
}
