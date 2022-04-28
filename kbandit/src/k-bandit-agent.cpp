//
// Created by jorge on 27/04/2022.
//

#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <execution>
#include "kbandit/k-bandit-agent.h"

KBanditsAgent::KBanditsAgent(size_t total_bandits): m_total_bandits(total_bandits) {
}

size_t KBanditsAgent::total_bandits() const {
	return m_total_bandits;
}

BasicGreedyAgent::BasicGreedyAgent(size_t bandits, double epsilon, double initial_estimate, RandomEngine::result_type seed):
	KBanditsAgent(bandits), m_bandit_distribution(0, bandits-1), m_greedy_option_distribution(1-epsilon),
    m_steps_per_bandit(bandits, 0), m_expected_rewards(bandits, initial_estimate)
{
    if(seed == std::numeric_limits<RandomEngine::result_type>::max()){
        std::random_device random_device;
        m_engine.seed( random_device());
    }
}

size_t BasicGreedyAgent::get_selection() const{
	if (do_greedy()) {
		return get_best_bandit();
	}
	else {
		// Select a random option
		return m_bandit_distribution(m_engine);
	}
}

size_t BasicGreedyAgent::get_best_bandit() const {
	auto iter_element = std::max_element(m_expected_rewards.cbegin(), m_expected_rewards.cend());
	return std::distance(m_expected_rewards.cbegin(), iter_element);
}

void BasicGreedyAgent::add_reward(size_t selection, double reward) {
    m_steps_per_bandit[selection] += 1;

	// Calculate the new expected reward
	double expected_reward = m_expected_rewards[selection];
	m_expected_rewards[selection] = expected_reward + step_value(m_steps_per_bandit[selection]) * (reward - expected_reward);
}

double BasicGreedyAgent::step_value(unsigned int steps_for_bandit) const {
	return 1.0 / steps_for_bandit;
}

bool BasicGreedyAgent::do_greedy() const{
	return m_greedy_option_distribution(m_engine);
}

UCBAgent::UCBAgent(std::size_t bandits, double confidence, double initial_estimate) : BasicGreedyAgent(bandits, 0., initial_estimate, 0),
m_confidence(confidence){

}

size_t UCBAgent::get_selection() const {
    std::vector<double> ucb(m_expected_rewards.begin(), m_expected_rewards.end());

    std::size_t total_steps = std::reduce( m_steps_per_bandit.begin(), m_steps_per_bandit.end(), 0u);
    std::transform(ucb.begin(), ucb.end(), m_steps_per_bandit.begin(), ucb.begin(),
                   [this, total_steps](auto expected, auto steps){
        return expected + m_confidence * std::sqrt( std::log(total_steps) / steps );
    });

    auto argmax = std::max_element(ucb.begin(), ucb.end());
    return std::distance(ucb.begin(), argmax);
}
