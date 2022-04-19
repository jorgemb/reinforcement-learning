#include "kbandit/k-bandit.h"

#include <random>
#include <numeric>
#include <algorithm>

std::default_random_engine create_random_engine(std::default_random_engine::result_type seed) {
	// Determine seed
	using Engine = std::default_random_engine;
	Engine engine;
	if (seed == std::numeric_limits<typename Engine::result_type>::max()) {
		// Create a new seed
		std::random_device dev;
		engine.seed(dev());
	}
	else {
		// Use given seed
		engine.seed(seed);
	}

	return engine;
}


Bandit::Bandit(double reward, double variance, typename Engine::result_type seed)
	: m_reward(reward), m_variance(variance), m_distribution(reward, variance), m_generator(create_random_engine(seed)) {
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

Bandit& KBandits::get_bandit(std::size_t k) {
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


BasicGreedyAgent::BasicGreedyAgent(std::size_t bandits, double epsilon, double initial_estimate): 
	KBanditsAgent(bandits), m_epsilon(epsilon), m_bandit_distribution(0, bandits-1), m_greedy_option_distribution(1-epsilon),
	m_steps{0}, m_expected_rewards(bandits, initial_estimate)
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
	m_steps += 1;

	// Calculate the new expected reward
	double expected_reward = m_expected_rewards[selection];
	m_expected_rewards[selection] = expected_reward + step_value() * (reward - expected_reward);
}

double BasicGreedyAgent::step_value() const {
	return 1.0 / m_steps;
}


bool BasicGreedyAgent::do_greedy() const{
	return m_greedy_option_distribution(m_engine);
}
