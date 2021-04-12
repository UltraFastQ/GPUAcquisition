#include <array>
#include <cstdlib>
#include <exception>

#include "Acquisition.hpp"
#include "AlazarInfo.hpp"
#include "GPUInfo.hpp"
#include "Utils.hpp"

constexpr const int gpu_device_id = 0;
constexpr const uint32_t alazar_system_id = 0;
constexpr const uint32_t alazar_board_id = 0;

int main(int argc, char* argv[]) {
	try {
		alz::display_alazar_info();
		utils::log_break();
		gpu::display_cuda_info();
		utils::log_break();

		Acquisition acq = Acquisition(alazar_system_id, alazar_board_id, gpu_device_id);

		// TODO: acq.set_...(); to change how the acquisition configuration will go

		acq.configure_devices();
		utils::log_message("Configured Alazar board and GPU successfully");

		utils::log_message("Starting data acquisition");
		acq.start();

		// acq's destructor cleans up all the memory it allocated
		utils::log_message("Cleaning up");
	}
	catch (const std::exception& e) {
		utils::log_error(std::string("Exception caught: ") + e.what(), __FILE__, __LINE__);
		return EXIT_FAILURE;
	}

    return EXIT_SUCCESS;
}