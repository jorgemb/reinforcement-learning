#ifndef GUARD_K_BANDIT_H
#define GUARD_K_BANDIT_H

#include <random>
#include <limits>
#include <array>

/// <summary>
/// Creates a new random engine with the given seed or with a random one
/// </summary>
/// <param name="seed"></param>
/// <returns></returns>
std::default_random_engine create_random_engine(std::default_random_engine::result_type seed = std::numeric_limits<std::default_random_engine::result_type>::max());

class Bandit {
public:
	using Engine = std::default_random_engine;

	/// <summary>
	/// Initializes the Bandit with the given average and variance, and optinally a seed.
	/// </summary>
	Bandit(double reward, double variance, typename Engine::result_type seed = std::numeric_limits<typename Engine::result_type>::max());

	/// <summary>
	/// Returns a random value from the Bandit distribution
	/// </summary>
	/// <returns></returns>
	double operator()();


	/// <summary>
	/// Returns the average of the Bandit
	/// </summary>
	/// <returns></returns>
	double mean_reward() const noexcept;

	/// <summary>
	/// Returns the variance of the Bandit
	/// </summary>
	/// <returns></returns>
	double variance() const noexcept;

private:
	double m_reward, m_variance;
	Engine m_generator;
	std::normal_distribution<double> m_distribution;
};

class KBandits {
public:
	using Engine = std::default_random_engine;

	/// <summary>
	/// Creates the bandits with the given mean reward and variance
	/// </summary>
	/// <param name="reward_mean"></param>
	/// <param name="reward_variance"></param>
	KBandits(double reward_mean,
		double reward_variance, 
		double bandit_variance = 1.0,
		std::size_t bandits = 10,
		Engine::result_type seed = std::numeric_limits<Engine::result_type>::max());

	/// <summary>
	/// Gets a random reward from the given bandit
	/// </summary>
	/// <param name="k"></param>
	/// <returns></returns>
	double get_reward(std::size_t k);

	/// <summary>
	/// Returns a reference to the bandit
	/// </summary>
	/// <param name="k"></param>
	/// <returns></returns>
	Bandit& get_bandit(std::size_t k);

	/// <summary>
	/// Returns the number of bandits
	/// </summary>
	/// <returns></returns>
	std::size_t get_num_bandits() const;

	/// <summary>
	/// Returns the index of the best bandit
	/// </summary>
	/// <returns></returns>
	std::size_t best_bandit() const;
private:
	std::vector<Bandit> m_bandits;
	std::size_t m_best_bandit;
};

class KBanditsAgent {
public:
	/// <summary>
	/// Constructor with the amount of bandits
	/// </summary>
	/// <param name="total_bandits"></param>
	KBanditsAgent(std::size_t total_bandits);

	/// <summary>
	/// Virtual destructor
	/// </summary>
	virtual ~KBanditsAgent() {}

	/// <summary>
	/// Returns the next selection to be made
	/// </summary>
	/// <returns></returns>
	virtual std::size_t get_selection() const = 0;

	/// <summary>
	/// Returns the best bandit according to the agent
	/// </summary>
	/// <returns></returns>
	virtual std::size_t get_best_bandit() const = 0;

	/// <summary>
	/// Adds a reward to the agent
	/// </summary>
	/// <param name="selection"></param>
	/// <param name="reward"></param>
	virtual void add_reward(std::size_t selection, double reward) = 0;
protected:
	std::size_t total_bandits() const;

private:
	const std::size_t m_total_bandits;
};

class BasicGreedyAgent : public KBanditsAgent{
public:
	BasicGreedyAgent(std::size_t bandits, double epsilon, double initial_estimate = std::numeric_limits<double>::infinity());
	virtual std::size_t get_selection() const override;
	virtual std::size_t get_best_bandit() const override;
	virtual void add_reward(std::size_t selection, double reward) override;
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

	mutable std::default_random_engine m_engine;
	mutable std::uniform_int_distribution<std::size_t> m_bandit_distribution;
	mutable std::bernoulli_distribution m_greedy_option_distribution;

    std::vector<unsigned int> m_steps_per_bandit;
	std::vector<double> m_expected_rewards;
};

#endif // !GUARD_K_BANDIT_H

