#ifndef REINFORCEMENT_LEARNING_DRAW_GRID_H
#define REINFORCEMENT_LEARNING_DRAW_GRID_H

#include <mdp/gridworld.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <utility>
#include <algorithm>

namespace rl::draw {
    /**
     * interpolate 2 RGB colors
     * @param color1    integer containing color as 0x00RRGGBB
     * @param color2    integer containing color as 0x00RRGGBB
     * @param fraction  how much interpolation (0..1)
     * - 0: full color 1
     * - 1: full color 2
     * From answer in: https://stackoverflow.com/a/21010385
     * @return the new color after interpolation
     */
    sf::Color interpolate(sf::Color color1, sf::Color color2, float fraction) {
        sf::Color result = color1;
        result.r += (color2.r - color1.r) * fraction;
        result.g += (color2.g - color1.g) * fraction;
        result.b += (color2.b - color1.b) * fraction;

        return result;
    }

    /// Allows for drawing in a Window
    class GridValue: public sf::Drawable{
    public:
        explicit GridValue(std::shared_ptr<rl::mdp::Gridworld>  gridworld):
        m_gridworld(std::move(gridworld)){
            // Position initial rectangles
            for (size_t i = 0; i < m_gridworld->get_rows(); ++i) {
                for (size_t j = 0; j < m_gridworld->get_columns(); ++j) {
                    // Get current rectangle
                    sf::RectangleShape rectangle(sf::Vector2f(SIZE_MULTIPLIER, SIZE_MULTIPLIER));

                    // Adjust position and colors
                    sf::Vector2f position(static_cast<float>(j) * SIZE_MULTIPLIER, static_cast<float>(i) * SIZE_MULTIPLIER);
                    rectangle.setPosition(position);
                    rectangle.setOutlineColor(sf::Color::Black);
                    rectangle.setOutlineThickness(0.5f);

                    // Set color
                    mdp::GridworldState state{i, j};
                    if(m_gridworld->is_terminal_state(state)){
                        rectangle.setFillColor(COLOR_TERMINAL);
                    }else if(m_gridworld->is_wall_state(state)){
                        rectangle.setFillColor(COLOR_WALL);
                    } else {
                        rectangle.setFillColor(COLOR_BEST);
                    }

                    // Add rectangle to vector
                    m_rectangles.push_back(rectangle);
                    m_state_value.push_back(0.0f);
                }
            }
        }

        /// Returns a View that would contain the whole grid
        /// \return
        [[nodiscard]]
        sf::View get_view() const{
            return sf::View(sf::FloatRect(
                    0.0f,
                    0.0f,
                    SIZE_MULTIPLIER*static_cast<float>(m_gridworld->get_columns()),
                    SIZE_MULTIPLIER*static_cast<float>(m_gridworld->get_rows())));
        }

        /// Sets the value of a position
        /// \param row
        /// \param column
        /// \param value
        void set_value(size_t row, size_t column, float value){
            size_t idx = row * m_gridworld->get_columns() + column;
            m_state_value[idx] = value;

            // Create color interpolation in all rectangles
            auto [worst_value, best_value] = std::minmax_element(m_state_value.cbegin(), m_state_value.cend());
            float value_distance = *best_value - *worst_value;
            for (int i = 0; i < m_rectangles.size(); ++i) {
                // Ignore if it is a special state
                mdp::GridworldState state{i / m_gridworld->get_columns(), i % m_gridworld->get_columns()};
                if(m_gridworld->is_terminal_state(state) || m_gridworld->is_wall_state(state)){
                    continue;
                }

                // currentColour = (targetColour - currentColour) * rate + currentColour
                float rate = (m_state_value[i] - *worst_value) / value_distance;
                sf::Color target_color = interpolate(COLOR_WORST, COLOR_BEST, rate);

                m_rectangles[i].setFillColor(target_color);
            }
        }

        /// Draws the grid
        /// \param target
        /// \param states
        void draw(sf::RenderTarget &target, sf::RenderStates states) const override{
            std::for_each(m_rectangles.cbegin(), m_rectangles.cend(),
                          [&](const sf::RectangleShape& rectangle){
                target.draw(rectangle, states);
            });
        }

    private:
        // Size multiplier for view size
        const float SIZE_MULTIPLIER = 100.0f;
        const sf::Color
            COLOR_BEST = sf::Color::Green,
            COLOR_WORST = sf::Color::Black,
            COLOR_TERMINAL = sf::Color::White,
            COLOR_WALL = sf::Color(128, 128, 128, 255);

        std::shared_ptr<rl::mdp::Gridworld> m_gridworld;

        std::vector<sf::RectangleShape> m_rectangles;
        std::vector<float> m_state_value;
    };

} // rl::draw

#endif //REINFORCEMENT_LEARNING_DRAW_GRID_H
