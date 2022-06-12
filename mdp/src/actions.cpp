#include <mdp/actions.h>

namespace rl::mdp{
    // FourWayActions
    const std::array<FourWayAction, 4> ActionTraits<FourWayAction>::s_actions{
        FourWayAction::LEFT,
        FourWayAction::UP,
        FourWayAction::RIGHT,
        FourWayAction::DOWN,
    };

    // TwoWayActions
    const std::array<TwoWayAction, 2> ActionTraits<TwoWayAction>::s_actions{
        TwoWayAction::LEFT,
        TwoWayAction::RIGHT
    };

    std::ostream &operator<<(std::ostream &os, const FourWayAction &action) {
        os << ActionTraits<FourWayAction>::to_str(action);
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const TwoWayAction &action) {
        os << ActionTraits<TwoWayAction>::to_str(action);
        return os;
    }
}