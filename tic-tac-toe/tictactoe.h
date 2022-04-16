#ifndef GUARD_TICTACTOE_H
#define GUARD_TICTACTOE_H

#include <array>
#include <vector>
#include <pair>

namespace rl::tictactoe {
	class Game {
	public:
		class enum Player : char{
			none = ' ',
			p1 = 'O',
			p2 = 'X'
		};

		bool set_position(int row, int col, Player player);

		Player get_position(int row, int col) const;

		bool player_win(Player player);

		std::vector<std::pair<int, int>> get_available_positions();

	private:
		std::array<Player, 9> m_board;
	};
}

#endif // !GUARD_TICTACTOE_H
