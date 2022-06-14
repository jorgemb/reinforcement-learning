#include <mdp/gridworld.h>
#include <mdp/agents.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <memory>
#include <vector>
#include <random>

using namespace Catch::literals;
using namespace rl::mdp;
using RandomEngine = std::default_random_engine;

TEMPLATE_TEST_CASE("Gridworld Agents", "[gridworld][agents]", BasicGridworldAgent) {
    RandomEngine::result_type seed(42);
    size_t max_steps(1000);

    // Create Gridworld
    size_t rows = 4, columns = 4;
    auto gridworld = std::make_shared<Gridworld>(rows, columns);
    GridworldState final_state{3, 3};
    gridworld->set_initial_state({0, 0});
    gridworld->set_terminal_state(final_state, 0.0);

    // Create environment
    using Environment = MDPEnvironment<Gridworld>;
    auto environment = std::make_shared<Environment>(gridworld, seed);

    // Create agent
    using Agent = TestType;
    auto agent = std::make_shared<Agent>();

    // Create experiment
    using Experiment = MDPExperiment<Environment, Agent>;
    Experiment experiment(max_steps);

    auto results = experiment.do_episode(environment, agent);
    INFO("Total steps: " << results.total_steps);
    REQUIRE(results.total_steps > 0);

    if (results.reached_terminal_state) {
        REQUIRE(results.last_state == final_state);
    }
}