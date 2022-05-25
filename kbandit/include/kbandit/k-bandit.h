#ifndef GUARD_K_BANDIT_H
#define GUARD_K_BANDIT_H

#include <random>
#include <limits>
#include <array>

/// Creates a new random engine with the given seed or with a random one
/// \param seed
/// \return
std::default_random_engine create_random_engine(std::default_random_engine::result_type seed = 0);

class Bandit {
public:
	using Engine = std::default_random_engine;

    /// Initializes the Bandit with the given average and variance, and optinally a seed.
    /// \param reward
    /// \param variance
    /// \param seed
	Bandit(double reward, double variance, typename Engine::result_type seed = std::numeric_limits<typename Engine::result_type>::max());

    /// Returns a random value from the Bandit distribution
    /// \return
	double operator()();

    /// Returns the average of the Bandit
    /// \return
	[[nodiscard]]
	double mean_reward() const noexcept;

    /// Returns the variance of the Bandit
    /// \return
    [[nodiscard]]
	double variance() const noexcept;

private:
	double m_reward, m_variance;
	Engine m_generator;
	std::normal_distribution<double> m_distribution;
};

class KBandits {
public:
	using Engine = std::default_random_engine;

    /// Creates the bandits with the given mean reward and variance
    /// \param reward_mean
    /// \param reward_variance
    /// \param bandit_variance
    /// \param bandits
    /// \param seed
	KBandits(double reward_mean,
		double reward_variance, 
		double bandit_variance = 1.0,
		std::size_t bandits = 10,
		Engine::result_type seed = std::numeric_limits<Engine::result_type>::max());

    /// Gets a random reward from the given bandit
    /// \param index
    /// \return
    [[nodiscard]]
	double get_reward(std::size_t index);

    /// Returns a reference to the bandit
    /// \param index
    /// \return
	Bandit& get_bandit(std::size_t index);

    /// Returns the number of bandits
    /// \return
    [[nodiscard]]
	std::size_t get_num_bandits() const;

    /// Returns the index of the best bandit
    /// \return
    [[nodiscard]]
	std::size_t best_bandit() const;
private:
	std::vector<Bandit> m_bandits;
	std::size_t m_best_bandit;
};

// Forward declaration
class KBanditsAgent;

/// Tests a single agent using the provided bandits for a total_runs amount of times.
/// \param bandits K-Bandits implementation
/// \param agent Agent to test
/// \param total_runs Number of runs to test
/// \return Reward for each run
std::vector<double> test_agent(KBandits& bandits, KBanditsAgent& agent, std::size_t total_runs);

#endif // !GUARD_K_BANDIT_H

