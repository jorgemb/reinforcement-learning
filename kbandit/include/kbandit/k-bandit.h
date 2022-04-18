#ifndef GUARD_K_BANDIT_H
#define GUARD_K_BANDIT_H

#include <random>
#include <limits>
#include <array>

template<class Engine = std::default_random_engine>
class Bandit {
public:
	/// <summary>
	/// Initializes the Bandit with the given average and variance, and optinally a seed.
	/// </summary>
	Bandit(double reward, double variance, typename Engine::result_type seed = std::numeric_limits<typename Engine::result_type>::max())
		: m_reward(reward), m_variance(variance), m_distribution(reward, variance)
	{
		// Determine seed
		if (seed == std::numeric_limits<typename Engine::result_type>::max()) {
			// Create a new seed
			std::random_device dev;
			m_generator.seed(dev());
		}
		else {
			// Use given seed
			m_generator.seed(seed);
		}
	}

	/// <summary>
	/// Returns a random value from the Bandit distribution
	/// </summary>
	/// <returns></returns>
	double operator()() {
		return m_distribution(m_generator);
	}


	/// <summary>
	/// Returns the average of the Bandit
	/// </summary>
	/// <returns></returns>
	double mean_reward() const noexcept {
		return m_reward;
	}

	/// <summary>
	/// Returns the variance of the Bandit
	/// </summary>
	/// <returns></returns>
	double variance() const noexcept{
		return m_variance;
	}

private:
	double m_reward, m_variance;
	Engine m_generator;
	std::normal_distribution<double> m_distribution;
};

class KBandits {
public:
	/// <summary>
	/// Creates the bandits with the given mean reward and variance
	/// </summary>
	/// <param name="reward_mean"></param>
	/// <param name="reward_variance"></param>
	KBandits(double reward_mean, double reward_variance, double bandit_variance = 1.0, std::size_t bandits = 10);

	/// <summary>
	/// Gets a random reward from the given bandit
	/// </summary>
	/// <param name="k"></param>
	/// <returns></returns>
	double get_reward(unsigned int k);

	/// <summary>
	/// Returns a reference to the bandit
	/// </summary>
	/// <param name="k"></param>
	/// <returns></returns>
	Bandit<>& get_bandit(unsigned int k);

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
	std::vector<Bandit<>> m_bandits;
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
	BasicGreedyAgent(std::size_t bandits, double epsilon);
	virtual std::size_t get_selection() const override;
	virtual void add_reward(std::size_t selection, double reward) override;
private:
	/// <summary>
	/// Returns true if the next selection should be the greedy one
	/// </summary>
	/// <returns></returns>
	bool do_greedy() const;

	double m_epsilon;

	mutable std::default_random_engine m_engine;
	std::uniform_int_distribution<std::size_t> m_bandit_distribution;
	std::bernoulli_distribution m_greedy_option_distribution;

	std::vector<std::vector<double>> m_rewards;
	std::vector<double> m_expected_rewards;
};

#endif // !GUARD_K_BANDIT_H

