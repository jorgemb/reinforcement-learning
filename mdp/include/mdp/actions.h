#ifndef REINFORCEMENT_LEARNING_ACTIONS_H
#define REINFORCEMENT_LEARNING_ACTIONS_H

#include <boost/core/span.hpp>
#include <vector>
#include <optional>
#include <array>

namespace rl::mdp {
    /// Template function to get a vector of the action space for a type.
    /// \tparam ActionType
    /// \return
    template<class ActionType>
    std::vector<ActionType> get_actions_list() = delete;

    template<class ActionType>
    class ActionTraits {
    public:
        /// Returns a read-only span with the available actions for the type
        /// \return
        static constexpr boost::span<const ActionType> available_actions() noexcept = delete;

        /// Returns a string representation given an action
        /// \param action
        /// \return
        static constexpr std::string to_str(const ActionType& action) noexcept = delete;

        /// Tries to parse a given string into an action
        /// \param action_str
        /// \return
//        static std::optional<ActionType> try_parse(const std::string& action_str) = delete;
    };


    /// VON NEUMANN NEIGHBORHOOD ACTIONS ///
    enum class FourWayAction{
        LEFT = 0,
        UP = 1,
        RIGHT = 2,
        DOWN = 3
    };

    template<>
    class ActionTraits<FourWayAction>{
    public:
        static constexpr boost::span<const FourWayAction> available_actions() noexcept;
        static constexpr std::string to_str(const FourWayAction& action) noexcept;
    private:
        static const std::array<FourWayAction, 4> s_actions;
    };

    /// STEERING ACTIONS ///
    enum class TwoWayAction{

    };
}


#endif //REINFORCEMENT_LEARNING_ACTIONS_H
