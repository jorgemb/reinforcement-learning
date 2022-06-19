#include <draw/grid.h>

#include <mdp/gridworld.h>

#include <utility>
#include <memory>

using namespace rl::mdp;
using namespace rl::draw;

/// Updates the policy and the grid
/// \param gridworld
/// \param grid
/// \param policy
void update_policy(const std::shared_ptr<Gridworld> &gridworld, GridValue &grid,
                   MDPPolicy<GridworldState, GridworldAction> &policy) {// Set the values to the policy
    policy.policy_evaluation();
    for (size_t i = 0; i < gridworld->get_rows(); ++i) {
        for (size_t j = 0; j < gridworld->get_columns(); ++j) {
            grid.set_value(i, j, policy.value_function(GridworldState{i, j}));
        }
    }
}

/// Create a Gridworld MDP
/// \return
std::shared_ptr<Gridworld> create_gridworld() {
    auto gridworld = std::make_shared<Gridworld>(5, 5);
    gridworld->cost_of_living(-1.0);
    gridworld->set_initial_state({0, 0});
    gridworld->set_terminal_state({4, 4}, 0.0);
    gridworld->set_wall_state({0, 1}, -1.0);
    gridworld->set_wall_state({1, 1}, -1.0);
//    gridworld->set_wall_state({2, 1}, -1.0);
    gridworld->set_wall_state({3, 1}, -1.0);

//    gridworld->set_wall_state({4, 3}, -1.0);
    gridworld->set_wall_state({3, 3}, -1.0);
    gridworld->set_wall_state({2, 3}, -1.0);
    gridworld->set_wall_state({1, 3}, -1.0);


    gridworld->set_wall_state({3, 2}, -1.0);

    return gridworld;
}

int main(){
    auto gridworld = create_gridworld();
    auto grid = GridValue(gridworld);
    sf::Clock clock;

    // Create and solve a policy
    auto policy = GridworldGreedyPolicy(gridworld, 1.0);
    sf::Time time_between_updates = sf::seconds(0.1);

    // Create the window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Draw Gridworld");
//    sf::RenderWindow window(sf::VideoMode(1920, 1080), "Draw Gridworld", sf::Style::Fullscreen);
    window.setVerticalSyncEnabled(true);
    window.setView(grid.get_view());

    // Main loop
    while(window.isOpen()){
        // Events loop
        sf::Event event{};
        while(window.pollEvent(event)){
            // Close window event
            if(event.type == sf::Event::EventType::Closed){
                window.close();
            }

            // Quit key
            if(event.type == sf::Event::EventType::KeyPressed){
                if(event.key.code == sf::Keyboard::Escape){
                    window.close();
                }
            }
        }

        // Drawing loop
        window.clear(sf::Color::Black);

        // Update policy
        if(clock.getElapsedTime() > time_between_updates) {
            update_policy(gridworld, grid, policy);

            clock.restart();
        }

        grid.draw(window, sf::RenderStates::Default);

        window.display();
    }
}