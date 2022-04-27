#ifndef REINFORCEMENT_LEARNING_K_BANDIT_AGENT_H
#define REINFORCEMENT_LEARNING_K_BANDIT_AGENT_H

#include <random>
#include <limits>
#include <array>
#include "k-bandit.h"


class KBanditsAgent {
public:
	/// <summary>
	/// Constructor with the amount of bandits
	/// </summary>
	/// <param name="total_bandits"></param>
	explicit KBanditsAgent(size_t total_bandits);

	/// <summary>
	/// Virtual destructor
	/// </summary>
	virtual ~KBanditsAgent() = default;

	/// <summary>
	/// Returns the next selection to be made
	/// </summary>
	/// <returns></returns>
	[[nodiscard]]
	virtual size_t get_selection() const = 0;

	/// <summary>
	/// Returns the best bandit according to the agent
	/// </summary>
	/// <returns></returns>
	[[nodiscard]]
	virtual size_t get_best_bandit() const = 0;

	/// <summary>
	/// Adds a reward to the agent
	/// </summary>
	/// <param name="selection"></param>
	/// <param name="reward"></param>
	virtual void add_reward(size_t selection, double reward) = 0;
protected:
    [[nodiscard]]
	size_t total_bandits() const;

private:
	const size_t m_total_bandits;
};

class BasicGreedyAgent : public KBanditsAgent{
public:
    using RandomEngine = std::default_random_engine;

	BasicGreedyAgent(size_t bandits, double epsilon, double initial_estimate = std::numeric_limits<double>::infinity(),
                     RandomEngine::result_type seed = std::numeric_limits<RandomEngine::result_type>::max());
	size_t get_selection() const override;
	size_t get_best_bandit() const override;
	void add_reward(size_t selection, double reward) override;
protected:
	/// <summary>
	/// Determine the step value for the next reward calculation
	/// </summary>
	/// <returns></returns>
	virtual double step_value(unsigned int steps_for_selection) const;
private:
	/// <summary>
	/// Returns true if the next selection should be the greedy one
	/// </summary>
	/// <returns></returns>
	bool do_greedy() const;

	mutable RandomEngine m_engine;
	mutable std::uniform_int_distribution<size_t> m_bandit_distribution;
	mutable std::bernoulli_distribution m_greedy_option_distribution;

    std::vector<unsigned int> m_steps_per_bandit;
	std::vector<double> m_expected_rewards;
};

#endif //REINFORCEMENT_LEARNING_K_BANDIT_AGENT_H
