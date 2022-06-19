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
    sf::Color interpolate(sf::Color color1, sf::Color color2, float fraction);

    /// Allows for drawing in a Window
    class GridValue: public sf::Drawable{
    public:
        explicit GridValue(std::shared_ptr<rl::mdp::Gridworld>  gridworld);

        /// Returns a View that would contain the whole grid
        /// \return
        [[nodiscard]]
        sf::View get_view() const;

        /// Sets the value of a position
        /// \param row
        /// \param column
        /// \param value
        void set_value(size_t row, size_t column, float value);

        /// Draws the grid
        /// \param target
        /// \param states
        void draw(sf::RenderTarget &target, sf::RenderStates states) const override;

    private:
        // Size multiplier for view size
        const float SIZE_MULTIPLIER = 100.0f;

        // Constants for color showing
        const sf::Color
            COLOR_BEST = sf::Color::Green,
            COLOR_WORST = sf::Color::Black,
            COLOR_TERMINAL = sf::Color::White,
            COLOR_WALL = sf::Color(128, 128, 128, 255);

        // Gridworld reference
        std::shared_ptr<rl::mdp::Gridworld> m_gridworld;

        // Internal drawables
        std::vector<sf::RectangleShape> m_rectangles;
        std::vector<float> m_state_value;
    };

} // rl::draw

#endif //REINFORCEMENT_LEARNING_DRAW_GRID_H
