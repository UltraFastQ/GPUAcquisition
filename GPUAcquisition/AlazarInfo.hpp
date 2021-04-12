#ifndef __ALAZAR_INFO_HPP__
#define __ALAZAR_INFO_HPP__

#include <cstdint>

namespace alz {
    /* Displays information about all Alazar board plugged into the system, as well as information about the Alazar SDK. Throws std::runtime_error in case of failure. */
    void display_alazar_info();

    /* Displays information about the Alazar SDK. Throws std::runtime_error in case of failure. */
    void display_sdk_version();

    /* Displays information about a board system. Throws std::runtime_error in case of failure. */
    void display_board_system_info(uint32_t system_id);

    /* Displays information about a board. Throws std::runtime_error in case of failure. */
    void display_board_info(uint32_t system_id, uint32_t board_id);
}

#endif /* __ALAZAR_INFO_HPP__ */
