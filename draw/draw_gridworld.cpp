#include <draw/grid.h>

#include <mdp/gridworld.h>

#include <utility>
#include <memory>

using namespace rl::mdp;
using namespace rl::draw;
std::shared_ptr<Gridworld> create_gridworld();

int main(){
    auto gridworld = create_gridworld();
    auto grid = GridValue(gridworld);

    // Create and solve a policy
    auto policy = GridworldGreedyPolicy(gridworld, 1.0);
    while(policy.policy_evaluation() > 0.1);

    // Set the values to the policy
    for (size_t i = 0; i < gridworld->get_rows(); ++i) {
        for (size_t j = 0; j < gridworld->get_columns(); ++j) {
            grid.set_value(i, j, policy.value_function(GridworldState{i, j}));
        }
    }

    sf::RenderWindow window(sf::VideoMode(800, 600), "Draw Gridworld");

    // Main loop
    while(window.isOpen()){
        // Events loop
        sf::Event event{};
        while(window.pollEvent(event)){
            // Close window event
            if(event.type == sf::Event::EventType::Closed){
                window.close();
            }
        }

        // Drawing loop
        window.clear(sf::Color::Black);

        grid.draw(window, sf::RenderStates::Default);

        window.display();
    }
}

std::shared_ptr<Gridworld> create_gridworld() {
    auto gridworld = std::make_shared<Gridworld>(4, 5);
    gridworld->cost_of_living(-1.0);
    gridworld->set_terminal_state({3, 4}, 1.0);
    gridworld->set_terminal_state({1, 1}, -100.0);
    gridworld->set_wall_state({0, 1}, -1.0);
    gridworld->set_wall_state({2, 1}, -1.0);
    return gridworld;
}
