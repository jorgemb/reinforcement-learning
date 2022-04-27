#include "kbandit/k-bandit.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

using namespace Catch::literals;

TEST_CASE("Single bandit", "[kbandit]") {
	int seed{ 42 };
	double mean_reward = 0.0, variance = 1.0;
	Bandit bandit(mean_reward, variance, seed);

	SECTION("Internal values") {
		REQUIRE(bandit.mean_reward() == Approx(mean_reward));
		REQUIRE(bandit.variance() == Approx(variance));
	}

	SECTION("Single value - Approximated") {
		double value = bandit();
		INFO("99% value should be mean +- 3sigma -- generated: " << value);
		REQUIRE(value == Approx(mean_reward).margin(3 * variance));
	}

	// Generate values for testing
	const unsigned int n = 100000;
	std::vector<double> values(n, 0.0);
	std::for_each(values.begin(), values.end(), [&bandit](double& val) { return val = bandit(); });
	SECTION("Approximate values") {
		// Calculate mean
		double total = std::reduce(values.begin(), values.end(), 0.0, std::plus<>{});
		double calc_mean = total / n;
		INFO("Total: " << total << ", calculated mean : " << calc_mean);
		REQUIRE(bandit.mean_reward() == Approx(calc_mean).margin(0.01));

		// Calculate variance
		double acum_variance = std::transform_reduce(values.begin(), values.end(), 0.0, std::plus<>{}, 
			[calc_mean](double val) { return std::pow(val - calc_mean, 2.0); });
		double variance = acum_variance / n;
		INFO("Acummulated variance: " << acum_variance << ", variance: " << variance);
		REQUIRE(bandit.variance() == Approx(variance).margin(0.01));
	}
}

TEST_CASE("K-Bandits", "[kbandit]") {
	unsigned int total_bandits = 10;
	double mean_reward = 0.0, variance_reward = 3.0, bandit_variance = 1.0;
	KBandits bandits(mean_reward, variance_reward, bandit_variance, total_bandits);

	SECTION("Basic validations") {
		REQUIRE(bandits.get_num_bandits() == total_bandits);
	}

	for (size_t i = 0; i < bandits.get_num_bandits(); i++) {
		DYNAMIC_SECTION("Rewards for i = " << i) {
			REQUIRE(bandits.get_bandit(i).mean_reward() == Approx(mean_reward).margin(3 * std::sqrt(variance_reward)));
			REQUIRE(bandits.get_bandit(i).variance() == Approx(bandit_variance));
		}
	}

	SECTION("Best bandit") {
		double best_reward = -std::numeric_limits<double>::infinity();
		std::size_t best = -1;

		for (size_t i = 0; i < bandits.get_num_bandits(); i++) {
			auto& b = bandits.get_bandit(i);

			if (b.mean_reward() > best_reward) {
				best = i;
				best_reward = b.mean_reward();
			}
		}

		REQUIRE(bandits.best_bandit() == best);
	}
}
