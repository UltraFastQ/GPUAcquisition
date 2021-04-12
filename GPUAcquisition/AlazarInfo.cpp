#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

#include "AlazarInfo.hpp"
#include "Utils.hpp"

void alz::display_alazar_info() {
    alz::display_sdk_version();

    uint32_t num_systems = AlazarNumOfSystems();
    if (num_systems == 0) {
        utils::log_warning("No Alazar systems found", __FILE__, __LINE__);
    }
    else {
        utils::log_message(std::string("Number of Alazar systems: ") + std::to_string(num_systems));

        // Display information about all systems
        for (uint32_t system_id = 1; system_id <= num_systems; ++system_id) {
            alz::display_board_system_info(system_id); // throws std::runtime_error in case of failure
        }
    }
}

void alz::display_sdk_version() {
    uint8_t sdk_major = 0;
    uint8_t sdk_minor = 0;
    uint8_t sdk_revision = 0;

    RETURN_CODE rc = AlazarGetSDKVersion(&sdk_major, &sdk_minor, &sdk_revision);
    utils::alazar_err_handle(rc, "AlazarGetSDKVersion failed", __FILE__, __LINE__); // throws std::runtime_error in case of failure
    utils::log_message(std::string("Alazar SDK Version: ") + std::to_string(sdk_major) + "." + std::to_string(sdk_minor) + "." + std::to_string(sdk_revision));
}

void alz::display_board_system_info(uint32_t system_id) {
    utils::log_message(std::string("System ID: ") + std::to_string(system_id));

    uint32_t num_boards = AlazarBoardsInSystemBySystemID(system_id);
    if (num_boards == 0) {
        utils::log_error(std::string("No Alazar boards found in system ID ") + std::to_string(system_id), __FILE__, __LINE__);
        throw std::runtime_error(std::string("No Alazar boards found in system ID ") + std::to_string(system_id));
    }
    utils::log_message(std::string("Number of Alazar boards: ") + std::to_string(num_boards));

    HANDLE handle = AlazarGetSystemHandle(system_id);
    if (handle == nullptr) {
        utils::log_error(std::string("AlazarGetSytemHandle failed for system ID ") + std::to_string(system_id), __FILE__, __LINE__);
        throw std::runtime_error(std::string("AlazarGetSytemHandle failed for system ID ") + std::to_string(system_id));
    }

    uint32_t board_type = AlazarGetBoardKind(handle);
    if (board_type == ATS_NONE || board_type >= ATS_LAST) {
        utils::log_error(std::string("Unknown Alazar board type ") + std::to_string(board_type), __FILE__, __LINE__);
        throw std::runtime_error(std::string("Unknown Alazar board type ") + std::to_string(board_type));
    }
    utils::log_message(std::string("Board type: ") + utils::get_board_type_string(board_type));

    uint8_t driver_major = 0;
    uint8_t driver_minor = 0;
    uint8_t driver_revision = 0;
    RETURN_CODE rc = AlazarGetDriverVersion(&driver_major, &driver_minor, &driver_revision);
    utils::alazar_err_handle(rc, "AlazarGetDriverVersion failed", __FILE__, __LINE__); // throws std::runtime_error in case of failure
    utils::log_message(std::string("Driver version: ") + std::to_string(driver_major) + "." + std::to_string(driver_minor) + "." + std::to_string(driver_revision));

    // Display information about all boards in this system
    for (uint32_t board_id = 1; board_id <= num_boards; ++board_id) {
        std::cout << "\n";
        alz::display_board_info(system_id, board_id); // throws std::runtime_error in case of failure
    }
}

void alz::display_board_info(uint32_t system_id, uint32_t board_id) {
    utils::log_message(std::string("Board ID: ") + std::to_string(board_id));

    HANDLE handle = AlazarGetBoardBySystemID(system_id, board_id);
    if (handle == nullptr) {
        utils::log_error(std::string("AlazarGetBoardSystemID failed for system ID ") + std::to_string(system_id) + " and board ID " + std::to_string(board_id), __FILE__, __LINE__);
        throw std::runtime_error(std::string("AlazarGetBoardSystemID failed for system ID ") + std::to_string(system_id) + " and board ID " + std::to_string(board_id));
    }

    unsigned long samples_per_channel = 0;
    uint8_t bits_per_sample = 0;
    RETURN_CODE rc = AlazarGetChannelInfo(handle, &samples_per_channel, &bits_per_sample);
    utils::alazar_err_handle(rc, "AlazarGetChannelInfo failed", __FILE__, __LINE__); // throws std::runtime_error in case of failure
    utils::log_message(std::string("Bits per sample: ") + std::to_string(bits_per_sample));
    utils::log_message(std::string("Max samples per channel: ") + std::to_string(samples_per_channel));

    unsigned long asopc_type = 0;
    rc = AlazarQueryCapability(handle, ASOPC_TYPE, 0, &asopc_type);
    utils::alazar_err_handle(rc, "AlazarQueryCapability failed", __FILE__, __LINE__); // throws std::runtime_error in case of failure
    utils::log_message(std::string("ASoPC signature: ") + utils::to_hex(asopc_type));

    unsigned char cpld_major = 0;
    unsigned char cpld_minor = 0;
    rc = AlazarGetCPLDVersion(handle, &cpld_major, &cpld_minor);
    utils::alazar_err_handle(rc, "AlazarGetCPLDVersion failed", __FILE__, __LINE__); // throws std::runtime_error in case of failure
    utils::log_message(std::string("CPLD version: ") + std::to_string(cpld_major) + "." + std::to_string(cpld_minor));

    unsigned long serial_num = 0;
    rc = AlazarQueryCapability(handle, GET_SERIAL_NUMBER, 0, &serial_num);
    utils::alazar_err_handle(rc, "AlazarQueryCapability failed", __FILE__, __LINE__); // throws std::runtime_error in case of failure
    utils::log_message(std::string("Serial number: ") + std::to_string(serial_num));

    uint32_t board_type = AlazarGetBoardKind(handle);
    if (board_type == ATS_NONE || board_type >= ATS_LAST) {
        utils::log_error(std::string("Uknown Alazar board type ") + std::to_string(board_type), __FILE__, __LINE__);
        throw std::runtime_error(std::string("Unknown Alazar board type ") + std::to_string(board_type));
    }
    if (board_type >= ATS9462) { // Ensure we have a PCIe board
        unsigned long pcie_lanes = 0;
        rc = AlazarQueryCapability(handle, GET_PCIE_LINK_WIDTH, 0, &pcie_lanes);
        utils::alazar_err_handle(rc, "AlazarQueryCapability failed", __FILE__, __LINE__); // throws std::runtime_error in case of failure
        utils::log_message(std::string("PCIe lanes available: " + std::to_string(pcie_lanes)));

        unsigned long pcie_link_speed = 0;
        rc = AlazarQueryCapability(handle, GET_PCIE_LINK_SPEED, 0, &pcie_link_speed);
        utils::alazar_err_handle(rc, "AlazarQueryCapability failed", __FILE__, __LINE__); // throws std::runtime_error in case of failure
        utils::log_message(std::string("PCIe link speed: ") + std::to_string(pcie_link_speed) + " Gb/s");

        float fpga_temp = 0.f;
        rc = AlazarGetParameterUL(handle, CHANNEL_ALL, GET_FPGA_TEMPERATURE, reinterpret_cast<unsigned long*>(&fpga_temp)); // TODO: There is no way this works as intended
        utils::alazar_err_handle(rc, "AlazarGetParameterUL failed", __FILE__, __LINE__);
        utils::log_message(std::string("FPGA temperature: ") + std::to_string(fpga_temp));
    }
}
