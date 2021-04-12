#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <typeinfo>
#include <type_traits>

#include <cuda_runtime.h>

#include <GLFW/glfw3.h>

#include <ATS_GPU.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Acquisition.hpp"
#include "UnpackOp.hpp"
#include "Utils.hpp"

namespace py = pybind11;

Acquisition::Acquisition(uint32_t alazar_system_id, uint32_t alazar_board_id, int gpu_device_id) :
	m_alazar_system_id(alazar_system_id),
	m_alazar_board_id(alazar_board_id),
	m_gpu_device_id(gpu_device_id)
{
	m_config = AcquisitionConfig();

	// Acquisition size parameters
	m_samples_per_buffer = 0; // This will get properly initialized in Acquisition::configure_devices()
	m_bytes_per_buffer = 0; // This will get properly initialized in Acquisition::configure_devices()
	m_num_channels = 0; // This will get properly initialized in Acquisition::configure_devices()

	m_did_start_capture = false;
	m_did_alloc_memory = false;

	m_ops_chan_a = std::vector<OpPtr>();
	m_ops_chan_b = std::vector<OpPtr>();

	try {
		// Get a board handle
		m_alazar_board = AlazarGetBoardBySystemID(m_alazar_system_id, m_alazar_board_id);
		if (m_alazar_board == nullptr) {
			utils::log_error(std::string("Could not retrieve Alazar board ") + std::to_string(alazar_board_id) + " in Alazar system " + std::to_string(alazar_system_id), __FILE__, __LINE__);
			throw std::runtime_error(std::string("Could not retrieve Alazar board ") + std::to_string(alazar_board_id) + " in Alazar system " + std::to_string(alazar_system_id));
		}
	}
	catch (const std::exception& e) {
		/* Note that it is necessary to call Acquisition::cleanup() here
		   because an exception thrown in a constructor does NOT lead to
		   the destructor being called (and thus calling cleanup()). */
		utils::log_error(std::string("Exception caught in Acquisition constructor: ") + e.what(), __FILE__, __LINE__);
		cleanup(); // clean up any memory we may have allocated in the constructor
		throw; // throw back the error
	}
}

void Acquisition::configure_devices(AcquisitionConfig* config) {
	if (config != nullptr) {
		m_config = *config;
	}

	// Clock
	RETURN_CODE rc = AlazarSetCaptureClock(m_alazar_board, m_config.capture_clock.source, m_config.capture_clock.sample_rate, m_config.capture_clock.edge, m_config.capture_clock.decimation);
	utils::alazar_err_handle(rc, "AlazarSetCaptureClock failed", __FILE__, __LINE__);

	// Inputs
	for (auto input_control : m_config.input_control) {
		rc = AlazarInputControl(m_alazar_board, input_control.channel, input_control.coupling, input_control.input_range, input_control.impedance);
		utils::alazar_err_handle(rc, "AlazarInputControl failed", __FILE__, __LINE__);
	}

	// Trigger operation
	rc = AlazarSetTriggerOperation(m_alazar_board, m_config.trigger_operation.trigger_operation, m_config.trigger_operation.trigger_engine1, m_config.trigger_operation.source1, m_config.trigger_operation.slope1, m_config.trigger_operation.level1, m_config.trigger_operation.trigger_engine2, m_config.trigger_operation.source2, m_config.trigger_operation.slope2, m_config.trigger_operation.level2);
	utils::alazar_err_handle(rc, "AlazarSetTriggerOperation failed", __FILE__, __LINE__);

	// External trigger
	rc = AlazarSetExternalTrigger(m_alazar_board, m_config.external_trigger.coupling, m_config.external_trigger.range);
	utils::alazar_err_handle(rc, "AlazarSetExternalTrigger failed", __FILE__, __LINE__);

	// Trigger delay
	rc = AlazarSetTriggerDelay(m_alazar_board, m_config.trigger_delay);
	utils::alazar_err_handle(rc, "AlazarSetTriggerDelay failed", __FILE__, __LINE__);

	// Trigger timeout
	rc = AlazarSetTriggerTimeOut(m_alazar_board, m_config.trigger_timeout_ticks);
	utils::alazar_err_handle(rc, "AlazarSetTriggerTimeOut failed", __FILE__, __LINE__);

	// AUX IO
	rc = AlazarConfigureAuxIO(m_alazar_board, m_config.aux_io.mode, m_config.aux_io.parameter);
	utils::alazar_err_handle(rc, "AlazarConfigureAuxIO failed", __FILE__, __LINE__);

	// Pack mode
	rc = AlazarSetParameter(m_alazar_board, CHANNEL_ALL, PACK_MODE, m_config.pack_mode);
	utils::alazar_err_handle(rc, "AlazarSetParameter failed", __FILE__, __LINE__);

	// Choose which GPU to use
	rc = ATS_GPU_SetCUDAComputeDevice(m_alazar_board, m_gpu_device_id);
	utils::alazar_err_handle(rc, "ATS_GPU_SetCUDADevice failed", __FILE__, __LINE__);

	// GPU Setup
	uint32_t samples_per_record_per_channel = m_config.acquisition_setup.pre_trigger_samples + m_config.acquisition_setup.post_trigger_samples;
	rc = ATS_GPU_Setup(m_alazar_board, m_config.acquisition_setup.channels, m_config.acquisition_setup.transfer_offset, samples_per_record_per_channel, m_config.acquisition_setup.records_per_buffer, m_config.acquisition_setup.records_per_acquisition, m_config.acquisition_setup.adma_flags, m_config.acquisition_setup.gpu_flags);
	utils::alazar_err_handle(rc, "ATS_GPU_Setup failed", __FILE__, __LINE__);

	// Number of channels enabled on the Alazar board
	m_num_channels = 0;
	long channels_per_board = 0;
	rc = AlazarGetParameter(m_alazar_board, CHANNEL_ALL, GET_CHANNELS_PER_BOARD, &channels_per_board);
	utils::alazar_err_handle(rc, "AlazarGetParameter failed", __FILE__, __LINE__);
	for (long i = 0; i < channels_per_board; ++i) {
		uint32_t channel_id = 1U << i;
		if (channel_id & m_config.acquisition_setup.channels) {
			++m_num_channels;
		}
	}

	m_samples_per_buffer = samples_per_record_per_channel * m_config.acquisition_setup.records_per_buffer;

	// Information about how the data is stored on the card
	unsigned long max_samples = 0;
	unsigned char bits_per_sample = 0;
	rc = AlazarGetChannelInfo(m_alazar_board, &max_samples, &bits_per_sample);
	utils::alazar_err_handle(rc, "AlazarGetChannelInfo failed", __FILE__, __LINE__);

	double bytes_per_sample = 0;
	if (!(m_config.acquisition_setup.gpu_flags & ATS_GPU_SETUP_FLAG::ATS_GPU_SETUP_FLAG_UNPACK)) {
		// If we have to unpack the data on our own, we have to figure out the number of bytes per sample
		switch (m_config.pack_mode) {
			case PACK_8_BITS_PER_SAMPLE:
				bytes_per_sample = 1.0;
				break;
			case PACK_12_BITS_PER_SAMPLE:
				bytes_per_sample = 1.5; // 12 bits = 1.5 bytes
				break;
			case PACK_DEFAULT:
				// FIXME: The default on all boards isn't 8 bits
			default:
				bytes_per_sample = 1.0;
				break;
		}
	}
	else {
		// If the card is to unpack the data for us, we let it determine the number of bytes per sample
		bytes_per_sample = static_cast<double>(((bits_per_sample + 7) / 8));
	}
	m_bytes_per_buffer = static_cast<uint32_t>(m_samples_per_buffer * bytes_per_sample);
}

void Acquisition::set_ops(std::vector<OpPtr>& ops_chan_a, std::vector<OpPtr>& ops_chan_b) {
	m_ops_chan_a = ops_chan_a;
	m_ops_chan_b = ops_chan_b;
}

void Acquisition::alloc_mem() {
	auto mini_cleanup = [&](const size_t i, const std::exception& e) {
		utils::log_error(std::string("Error caught: ") + e.what(), __FILE__, __LINE__);

		bool failed = false;
		// Free the previously allocated buffers
		for (size_t j = 0; j <= i; ++j) {
			// ATS allocated buffers
			try {
				auto rc = ATS_GPU_FreeBuffer(m_alazar_board, m_raw_buffers[j]);
				utils::alazar_err_handle(rc, "ATS_GPU_FreeBuffer", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				// Ignore the error for now.
				// We will report it later, but we need to free all the buffers we can before that
				failed = true;
			}

			// Raw data from channel A (GPU)
			try {
				auto cuda_rc = cudaFree(m_chan_a_real_buffers[j]);
				utils::cuda_err_handle(cuda_rc, "cudaFree failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}

			// Raw data from channel B (GPU)
			try {
				auto cuda_rc = cudaFree(m_chan_b_real_buffers[j]);
				utils::cuda_err_handle(cuda_rc, "cudaFree failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}

			// Processed data from channel A (GPU)
			try {
				auto cuda_rc = cudaFree(m_chan_a_complex_buffers[j]);
				utils::cuda_err_handle(cuda_rc, "cudaFree failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}

			// Processed data from channel B (GPU)
			try {
				auto cuda_rc = cudaFree(m_chan_b_complex_buffers[j]);
				utils::cuda_err_handle(cuda_rc, "cudaFree failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}

			// Processed data from channel A (CPU)
			try {
				auto cuda_rc = cudaFreeHost(m_cpu_chan_a_real_buffers[j]);
				utils::cuda_err_handle(cuda_rc, "cudaFreeHost failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}

			// Processed data from channel B (CPU)
			try {
				auto cuda_rc = cudaFreeHost(m_cpu_chan_b_real_buffers[j]);
				utils::cuda_err_handle(cuda_rc, "cudaFreeHost failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}

			// Processed data from channel A (CPU)
			try {
				auto cuda_rc = cudaFreeHost(m_cpu_chan_a_complex_buffers[j]);
				utils::cuda_err_handle(cuda_rc, "cudaFreeHost failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}

			// Processed data from channel B (CPU)
			try {
				auto cuda_rc = cudaFreeHost(m_cpu_chan_b_complex_buffers[j]);
				utils::cuda_err_handle(cuda_rc, "cudaFreeHost failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}

			// Destory CUDA streams
			try {
				auto cuda_rc = cudaStreamDestroy(m_streams[j]);
				utils::cuda_err_handle(cuda_rc, "cudaStreamDestroy failed", __FILE__, __LINE__);
			}
			catch (std::exception& _e) {
				failed = true;
			}
		}
		if (failed) {
			// Report the error that we saw earlier
			utils::log_error("Error handling code failed", __FILE__, __LINE__);
			throw std::runtime_error("Error handling code failed");
		}
	};

	// CPU/GPU buffers
	m_raw_buffers = std::vector<unsigned char*>(m_config.num_gpu_buffers, nullptr);
	m_chan_a_real_buffers = std::vector<cufftReal*>(m_config.num_gpu_buffers, nullptr);
	m_chan_b_real_buffers = std::vector<cufftReal*>(m_config.num_gpu_buffers, nullptr);
	m_chan_a_complex_buffers = std::vector<cufftComplex*>(m_config.num_gpu_buffers, nullptr);
	m_chan_b_complex_buffers = std::vector<cufftComplex*>(m_config.num_gpu_buffers, nullptr);
	m_cpu_chan_a_real_buffers = std::vector<cufftReal*>(m_config.num_gpu_buffers, nullptr);
	m_cpu_chan_b_real_buffers = std::vector<cufftReal*>(m_config.num_gpu_buffers, nullptr);
	m_cpu_chan_a_complex_buffers = std::vector<cufftComplex*>(m_config.num_gpu_buffers, nullptr);
	m_cpu_chan_b_complex_buffers = std::vector<cufftComplex*>(m_config.num_gpu_buffers, nullptr);
	m_streams = std::vector<cudaStream_t>(m_config.num_gpu_buffers, nullptr);

	// File I/O
	if (m_config.data_writing.fname != "" && m_config.data_writing.num_buffs_to_write > 0) {
		m_out_file = std::ofstream(m_config.data_writing.fname, std::ios::out | std::ios::app | std::ios::binary);
	}
	
	// We need to allocate space on the GPU to receive the data from the Alazar card
	for (size_t i = 0; i < m_config.num_gpu_buffers; ++i) {
		// Allocate a buffer on the GPU (this holds the output from the Alazar card)
		try {
			m_raw_buffers[i] = static_cast<unsigned char*>(ATS_GPU_AllocBuffer(m_alazar_board, m_bytes_per_buffer * m_num_channels, nullptr)); // Note: nullptr is &m_streams[i] in the example code for some reason
			if (m_raw_buffers[i] == nullptr) {
				utils::log_error("ATS_GPU_AllocBuffer returned a nullptr", __FILE__, __LINE__);
				throw std::runtime_error("ATS_GPU_AllocBuffer returned a nullptr");
			}
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Post this buffer to the Alazar board
		try {
			auto rc = ATS_GPU_PostBuffer(m_alazar_board, m_raw_buffers[i], m_bytes_per_buffer * m_num_channels);
			utils::alazar_err_handle(rc, "ATS_GPU_PostBuffer failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Allocate a buffer on the GPU for the unprocessed data from channel A (as a cufftReal type, i.e. a float)
		try {
			auto cuda_rc = cudaMalloc(&m_chan_a_real_buffers[i], m_samples_per_buffer * sizeof(*m_chan_a_real_buffers[i]));
			utils::cuda_err_handle(cuda_rc, "cudaMalloc failed", __FILE__, __LINE__);

			cuda_rc = cudaMemset(m_chan_a_real_buffers[i], 0, m_samples_per_buffer * sizeof(*m_chan_a_real_buffers[i]));
			utils::cuda_err_handle(cuda_rc, "cudaMemset failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Allocate a buffer on the GPU for the unprocessed data from channel B (as a cufftReal type, i.e. a float)
		try {
			auto cuda_rc = cudaMalloc(&m_chan_b_real_buffers[i], m_samples_per_buffer * sizeof(*m_chan_b_real_buffers[i]));
			utils::cuda_err_handle(cuda_rc, "cudaMalloc failed", __FILE__, __LINE__);

			cuda_rc = cudaMemset(m_chan_b_real_buffers[i], 0, m_samples_per_buffer * sizeof(*m_chan_b_real_buffers[i]));
			utils::cuda_err_handle(cuda_rc, "cudaMemset failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Allocates a buffer on the GPU for the processed data from channel A (as a cufftComplex type, i.e. a tuple of floats)
		try {
			auto cuda_rc = cudaMalloc(&m_chan_a_complex_buffers[i], ((static_cast<unsigned long long>(m_samples_per_buffer) / 2) + 1) * sizeof(*m_chan_a_complex_buffers[i]));
			utils::cuda_err_handle(cuda_rc, "cudaMalloc failed", __FILE__, __LINE__);

			cuda_rc = cudaMemset(m_chan_a_complex_buffers[i], 0, ((static_cast<unsigned long long>(m_samples_per_buffer) / 2) + 1) * sizeof(*m_chan_a_complex_buffers[i]));
			utils::cuda_err_handle(cuda_rc, "cudaMemset failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Allocates a buffer on the GPU for the processed data from channel B (as a cufftComplex type, i.e. a tuple of floats)
		try {
			auto cuda_rc = cudaMalloc(&m_chan_b_complex_buffers[i], ((static_cast<unsigned long long>(m_samples_per_buffer) / 2) + 1) * sizeof(*m_chan_b_complex_buffers[i]));
			utils::cuda_err_handle(cuda_rc, "cudaMalloc failed", __FILE__, __LINE__);

			cuda_rc = cudaMemset(m_chan_b_complex_buffers[i], 0, ((static_cast<unsigned long long>(m_samples_per_buffer) / 2) + 1) * sizeof(*m_chan_b_complex_buffers[i]));
			utils::cuda_err_handle(cuda_rc, "cudaMemset failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Allocates a buffer on the CPU for the processed from channel A, but on the CPU (as a cufftReal type, i.e. a float)
		try {
			auto cuda_rc = cudaHostAlloc(&m_cpu_chan_a_real_buffers[i], static_cast<unsigned long long>(m_samples_per_buffer) * sizeof(*m_cpu_chan_a_real_buffers[i]), cudaHostAllocDefault);
			utils::cuda_err_handle(cuda_rc, "cudaHostAlloc failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Allocates a buffer on the CPU for the processed from channel B, but on the CPU (as a cufftReal type, i.e. a float)
		try {
			auto cuda_rc = cudaHostAlloc(&m_cpu_chan_b_real_buffers[i], static_cast<unsigned long long>(m_samples_per_buffer) * sizeof(*m_cpu_chan_b_real_buffers[i]), cudaHostAllocDefault);
			utils::cuda_err_handle(cuda_rc, "cudaHostAlloc failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Allocates a buffer on the CPU for the processed from channel A, but on the CPU (as a cufftComplex type, i.e. a tuple of floats)
		try {
			auto cuda_rc = cudaHostAlloc(&m_cpu_chan_a_complex_buffers[i], ((static_cast<unsigned long long>(m_samples_per_buffer) / 2) + 1) * sizeof(*m_cpu_chan_a_complex_buffers[i]), cudaHostAllocDefault);
			utils::cuda_err_handle(cuda_rc, "cudaHostAlloc failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Allocates a buffer on the CPU for the processed from channel B, but on the CPU (as a cufftComplex type, i.e. a tuple of floats)
		try {
			auto cuda_rc = cudaHostAlloc(&m_cpu_chan_b_complex_buffers[i], ((static_cast<unsigned long long>(m_samples_per_buffer) / 2) + 1) * sizeof(*m_cpu_chan_a_complex_buffers[i]), cudaHostAllocDefault);
			utils::cuda_err_handle(cuda_rc, "cudaHostAlloc failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}

		// Create CUDA streams
		try {
			auto cuda_rc = cudaStreamCreate(&m_streams[i]);
			utils::cuda_err_handle(cuda_rc, "cudaStreamCreate failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			mini_cleanup(i, e);
		}
	}

	m_did_alloc_memory = true;
}

static void glfw_err_callback(int error, const char* desc) {
	utils::log_error(std::string("GLFW error  ") + std::to_string(error) + " " + desc, __FILE__, __LINE__);
}

static std::shared_mutex should_stop_acq_mutex;
static bool should_stop_acq;

static std::atomic<size_t> buff_idx = 0;

static std::atomic<bool> buffer_thread_done = false;

void Acquisition::buffer_handler() {
	std::vector<unsigned char*> raw_buffers_cpu(m_config.num_gpu_buffers, nullptr);
	for (size_t i = 0; i < m_config.num_gpu_buffers; ++i) {
		auto cuda_rc = cudaHostAlloc(&raw_buffers_cpu[i], m_bytes_per_buffer * m_num_channels, cudaHostAllocDefault);
		utils::cuda_err_handle(cuda_rc, "cudaHostAlloc failed", __FILE__, __LINE__);
	}

	buff_idx = 0;
	buffer_thread_done = false;

	static std::array<float, 7> times;
	std::array<std::chrono::steady_clock::time_point, times.size()> timers_start;
	std::array<std::chrono::steady_clock::time_point, times.size()> timers_stop;

	auto time_start = std::chrono::steady_clock::now();
	size_t buffers_full = 0; // The number of buffers this thread fills
	constexpr const uint32_t buffer_timeout_ms = 10000; // Timeout should be longer than the time required to capture all the data (TODO: Make this a prviate variable)

	utils::log_message(std::string("Capturing data from ") + std::to_string(m_config.num_gpu_buffers) + " buffers");

	// Tell the Alazar card to start capturing data
	RETURN_CODE rc = ATS_GPU_StartCapture(m_alazar_board);
	m_did_start_capture = true;
	utils::alazar_err_handle(rc, "ATS_GPU_StartCapture failed", __FILE__, __LINE__);

	while (true) {
		timers_start[6] = std::chrono::steady_clock::now();
		timers_start[5] = std::chrono::steady_clock::now();
		{
			std::shared_lock lock(should_stop_acq_mutex);
			if (should_stop_acq) break;
		}

		timers_stop[5] = std::chrono::steady_clock::now();
		times[5] = ((times[5] * buffers_full) + static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(timers_stop[5] - timers_start[5]).count())) / static_cast<float>(buffers_full + 1);

		buff_idx = buffers_full % m_config.num_gpu_buffers;

		timers_start[0] = std::chrono::steady_clock::now();
		// Grab the raw data from the acquisition and place it into m_raw_buffers[buff_idx]
		auto rc = ATS_GPU_GetBuffer(m_alazar_board, m_raw_buffers[buff_idx], buffer_timeout_ms, nullptr);
		// FIXME: ApiDmaPending? ApiDmaInProgress? ApiInvalidBuffer?
		try {
			utils::alazar_err_handle(rc, "ATS_GPU_GetBuffer failed", __FILE__, __LINE__);
		}
		catch (const std::exception& e) {
			utils::log_error(std::string("Caught exception: ") + e.what());
			break;
		}
		timers_stop[0] = std::chrono::steady_clock::now();
		times[0] = ((times[0] * buffers_full) + static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(timers_stop[0] - timers_start[0]).count())) / static_cast<float>(buffers_full + 1);

		// We have one more full buffer
		++buffers_full;

		// TODO: Use streams properly (ignore Alazar guide) to compute faster
		// TODO: Why are we bothering with streams if we only use the default stream?

		timers_start[1] = std::chrono::steady_clock::now();
		// Separate the raw data into their separate streams
		UnpackOp unpack(static_cast<size_t>(m_bytes_per_buffer) * static_cast<size_t>(m_num_channels));
		unpack.operate(m_raw_buffers[buff_idx], m_chan_a_real_buffers[buff_idx], m_chan_b_real_buffers[buff_idx]);

		auto cuda_rc = cudaStreamSynchronize(0); // default stream is stream 0
		utils::cuda_err_handle(cuda_rc, "cudaStreamSynchronize failed", __FILE__, __LINE__);
		timers_stop[1] = std::chrono::steady_clock::now();
		times[1] = ((times[1] * (buffers_full - 1)) + static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(timers_stop[1] - timers_start[1]).count())) / static_cast<float>(buffers_full);

		timers_start[2] = std::chrono::steady_clock::now();
		// Apply all the other operations
		for (auto operation : m_ops_chan_a) {
			std::visit([&](auto&& op) {
				using T = std::decay_t<decltype(op)>;
				if constexpr (std::is_same_v<T, FFTOp*>) {
					op->operate(m_chan_a_real_buffers[buff_idx], m_chan_a_complex_buffers[buff_idx]);
				}
				else if constexpr (std::is_same_v<T, BoxCarOp*>) {
					op->init(m_samples_per_buffer, 2000, 2000 * static_cast<size_t>(m_samples_per_buffer) * (buffers_full - 1));
					op->set_prev_buffer((buffers_full - 1) == 0 ? nullptr : m_chan_a_real_buffers[(buffers_full - 2) % m_config.num_gpu_buffers]);
					op->operate(m_chan_a_real_buffers[buff_idx], m_chan_a_real_buffers[buff_idx]);
				}
				else if constexpr (std::is_same_v<T, LIA*>) {
					op->operate(m_chan_a_real_buffers[buff_idx], m_chan_a_complex_buffers[buff_idx]);
				}
				else if constexpr (std::is_same_v<T, ExtractPulsesOp*>) {
					op->init(m_samples_per_buffer);
					op->operate(m_chan_a_real_buffers[buff_idx], m_chan_a_real_buffers[buff_idx]);
				}
				else {
					utils::log_error(std::string(typeid(op).name()) + " is not a valid Operation class", __FILE__, __LINE__);
					throw std::runtime_error(std::string(typeid(op).name()) + " is not a valid Operation class");
				}
				cuda_rc = cudaStreamSynchronize(0);
				utils::cuda_err_handle(cuda_rc, "cudaStreamSynchronize failed", __FILE__, __LINE__);
			}, operation);
		}
		for (auto operation : m_ops_chan_b) {
			std::visit([&](auto&& op) {
				using T = std::decay_t<decltype(op)>;
				if constexpr (std::is_same_v<T, FFTOp*>) {
					op->operate(m_chan_b_real_buffers[buff_idx], m_chan_b_complex_buffers[buff_idx]);
				}
				else if constexpr (std::is_same_v<T, BoxCarOp*>) {
					op->init(m_samples_per_buffer, 2000, 2000 * static_cast<size_t>(m_samples_per_buffer) * (buffers_full - 1));
					op->set_prev_buffer((buffers_full - 1) == 0 ? nullptr : m_chan_b_real_buffers[(buffers_full - 2) % m_config.num_gpu_buffers]);
					op->operate(m_chan_b_real_buffers[buff_idx], m_chan_b_real_buffers[buff_idx]);
				}
				else if constexpr (std::is_same_v<T, LIA*>) {
					op->operate(m_chan_b_real_buffers[buff_idx], m_chan_b_complex_buffers[buff_idx]);
				}
				else if constexpr (std::is_same_v<T, ExtractPulsesOp*>) {
					op->init(m_samples_per_buffer);
					op->operate(m_chan_b_real_buffers[buff_idx], m_chan_b_real_buffers[buff_idx]);
				}
				else {
					utils::log_error(std::string(typeid(op).name()) + " is not a valid Operation class", __FILE__, __LINE__);
					throw std::runtime_error(std::string(typeid(op).name()) + " is not a valid Operation class");
				}

				cuda_rc = cudaStreamSynchronize(0); // default stream is stream 0
				utils::cuda_err_handle(cuda_rc, "cudaStreamSynchronize failed", __FILE__, __LINE__);
			}, operation);
		}
		timers_stop[2] = std::chrono::steady_clock::now();
		times[2] = ((times[2] * (buffers_full - 1)) + static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(timers_stop[2] - timers_start[2]).count())) / static_cast<float>(buffers_full);

		if (buffers_full % 500 == 0) {
			utils::log_message(std::string("Processed buffer ") + std::to_string(buffers_full));
		}
		
		// Save data to disk if we need to
		//if (buffers_full <= m_config.data_writing.num_buffs_to_write && m_out_file.is_open()) {
		if (m_out_file.is_open()) {
			cuda_rc = cudaMemcpy(m_cpu_chan_a_real_buffers[buff_idx], m_chan_a_real_buffers[buff_idx], m_samples_per_buffer * sizeof(cufftReal), cudaMemcpyDeviceToHost);
			//cuda_rc = cudaMemcpy(raw_buffers_cpu[buff_idx], m_raw_buffers[buff_idx], m_bytes_per_buffer * m_num_channels, cudaMemcpyDeviceToHost);
			utils::cuda_err_handle(cuda_rc, "cudaMemcpy failed", __FILE__, __LINE__);
			// utils::log_message("Writing data to disk");
			m_out_file.write(reinterpret_cast<char*>(&m_cpu_chan_a_real_buffers[buff_idx][0]), m_samples_per_buffer * sizeof(cufftReal));
		}

		// Post this buffer back to the Alazar board so that it can be reused for a future data acquisition
		timers_start[4] = std::chrono::steady_clock::now();
		rc = ATS_GPU_PostBuffer(m_alazar_board, m_raw_buffers[buff_idx], m_bytes_per_buffer * m_num_channels);
		utils::alazar_err_handle(rc, "ATS_GPU_PostBuffer failed", __FILE__, __LINE__);
		timers_stop[4] = std::chrono::steady_clock::now();
		times[4] = ((times[4] * (buffers_full - 1)) + static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(timers_stop[4] - timers_start[4]).count())) / static_cast<float>(buffers_full);
		
		timers_stop[6] = std::chrono::steady_clock::now();
		times[6] = ((times[6] * (buffers_full - 1)) + static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(timers_stop[6] - timers_start[6]).count())) / static_cast<float>(buffers_full);

		auto time_end = std::chrono::steady_clock::now();
		auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
		if (time_ms >= 5000) break; // stop after 5 seconds
		// if (m_config.data_writing.num_buffs_to_write && buffers_full >= m_config.data_writing.num_buffs_to_write) break;
		// if (buffers_full == 1010) break;
		// break;
	}

	{
		std::unique_lock lock(should_stop_acq_mutex);
		should_stop_acq = true;
	}

	for (size_t i = 0; i < m_config.num_gpu_buffers; ++i) {
		auto cuda_rc = cudaFreeHost(raw_buffers_cpu[i]);
		utils::cuda_err_handle(cuda_rc, "cudaFreeHost failed", __FILE__, __LINE__);
	}

	auto time_end = std::chrono::steady_clock::now();
	auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
	float acq_size = static_cast<float>(buffers_full * static_cast<size_t>(m_bytes_per_buffer) * static_cast<size_t>(m_num_channels)) / (static_cast<float>(1 << 30));
	utils::log_message(std::string("Buffers filled: ") + std::to_string(buffers_full));
	utils::log_message(std::string("Acquisition time: ") + std::to_string(time_ms) + " ms");
	utils::log_message(std::string("Acquisition size: ") + std::to_string(acq_size) + " GiB");
	utils::log_message(std::string("Average acquisition rate: ") + std::to_string(acq_size / (static_cast<float>(time_ms) / 1e3f)) + " GiB/s");
	utils::log_break();
	utils::log_message(std::string("Average GetBuffer time: ") + std::to_string(times[0]) + " us");
	utils::log_message(std::string("Average unpack time: ") + std::to_string(times[1]) + " us");
	utils::log_message(std::string("Average operations time: ") + std::to_string(times[2]) + " us");
	// utils::log_message(std::string("Average memcpy time: ") + std::to_string(times[3]) + " us");
	utils::log_message(std::string("Average PostBuffer time: ") + std::to_string(times[4]) + " us");
	utils::log_message(std::string("Average Window/Python time: ") + std::to_string(times[5]) + " us");
	utils::log_message(std::string("Average loop time: ") + std::to_string(times[6]) + " us");

	buffer_thread_done = true;
}

static void window_handler() {
	using namespace std::chrono_literals;

	// GLFW prep
	glfwSetErrorCallback(glfw_err_callback);
	if (!glfwInit()) {
		utils::log_error("Could not initialize GLFW", __FILE__, __LINE__);
		throw std::runtime_error("Could not initialize GLFW");
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	GLFWwindow* window = glfwCreateWindow(400, 1, "Close window to stop data acquisition", nullptr, nullptr);
	if (window == nullptr) {
		utils::log_error("Could not create GLFW window", __FILE__, __LINE__);
		throw std::runtime_error("Could not create GLFW window");
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	while (true) {
		{
			std::shared_lock lock(should_stop_acq_mutex);
			if (should_stop_acq) break;
		}

		glfwPollEvents();
		if (glfwWindowShouldClose(window)) {
			std::unique_lock lock(should_stop_acq_mutex);
			should_stop_acq = true;
			break;
		}
		std::this_thread::sleep_for(16.67ms); // Check window events ~ 60 times per second
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}

void Acquisition::start() {
	// Allocate the necessary memory
	alloc_mem();

	{
		std::unique_lock lock(should_stop_acq_mutex);
		should_stop_acq = false;
	}

	// Creates a new window to tell us when to stop our program
	std::thread window_thread = std::thread(&window_handler);
	window_thread.detach();

	// The thread that will do the work of capturing and processing the data
	std::thread buffer_thread([&]() { buffer_handler(); });
	buffer_thread.detach();
}

py::memoryview Acquisition::get_chan_a(const size_t num_points) {
	const size_t idx = buff_idx == 0 ? m_config.num_gpu_buffers - 1 : (buff_idx - 1) % m_config.num_gpu_buffers;

	const size_t size = min(num_points * sizeof(*m_cpu_chan_a_real_buffers[idx]), static_cast<unsigned long long>(m_samples_per_buffer) * sizeof(*m_cpu_chan_a_real_buffers[idx]));
	auto cuda_rc = cudaMemcpy(m_cpu_chan_a_real_buffers[idx], m_chan_a_real_buffers[idx], size, cudaMemcpyDeviceToHost);

	utils::cuda_err_handle(cuda_rc, "cudaMemcpyAsync failed", __FILE__, __LINE__);

	return py::memoryview::from_memory(m_cpu_chan_a_real_buffers[idx], size, true);
	// return py::memoryview::from_buffer(m_cpu_chan_a_real_buffers[idx], { 1<<22 }, { sizeof(cufftReal) * (1<<14) });
	// return py::memoryview::from_buffer(m_cpu_chan_a_real_buffers[idx], sizeof(cufftReal), "f", { 1 << 12 }, { sizeof(cufftReal) * 1 << 10 });
}

py::memoryview Acquisition::get_chan_b(const size_t num_points) {
	const size_t idx = buff_idx == 0 ? m_config.num_gpu_buffers - 1 : (buff_idx - 1) % m_config.num_gpu_buffers;

	const size_t size = min(num_points * sizeof(*m_cpu_chan_b_real_buffers[idx]), static_cast<unsigned long long>(m_samples_per_buffer) * sizeof(*m_cpu_chan_b_real_buffers[idx]));
	auto cuda_rc = cudaMemcpy(m_cpu_chan_b_real_buffers[idx], m_chan_b_real_buffers[idx], size, cudaMemcpyDeviceToHost);
	utils::cuda_err_handle(cuda_rc, "cudaMemcpyAsync failed", __FILE__, __LINE__);

	return py::memoryview::from_memory(m_cpu_chan_b_real_buffers[idx], size, true);
}

void Acquisition::cleanup() {
	// Tell everyone to stop
	{
		std::unique_lock lock(should_stop_acq_mutex);
		should_stop_acq = true;
	}

	// Wait for all the threads to finish (or give up after 30 seconds)
	auto thread_time_start = std::chrono::steady_clock::now();
	while (true) {
		using namespace std::chrono_literals;

		auto time_end = std::chrono::steady_clock::now();
		auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - thread_time_start).count();

		// Stop after all the threads are done (or give up after 30 seconds)
		if (time_ms >= 30000 || buffer_thread_done) break;

		std::this_thread::sleep_for(100ms); // Wait a little bit for more threads to finish
	}
	utils::log_message(std::string("Thread finished (or timed out) in ") + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - thread_time_start).count()) + " ms\n");

	// Stop capturing data
	if (m_did_start_capture) {
		ATS_GPU_AbortCapture(m_alazar_board);
		m_did_start_capture = false;
	}

	// Close file if we need to
	if (m_out_file.is_open()) {
		m_out_file.close();
	}

	// If we did not allocate any memory, we don't need to free any memory
	if (!m_did_alloc_memory) return;

	// Free the previously allocated buffers
	bool failed = false;
	for (size_t i = 0; i < m_config.num_gpu_buffers; ++i) {
		// ATS allocated buffers
		try {
			auto rc = ATS_GPU_FreeBuffer(m_alazar_board, m_raw_buffers[i]);
			utils::alazar_err_handle(rc, "ATS_GPU_FreeBuffer", __FILE__, __LINE__);
		}
		catch (std::exception& _e) {
			// Ignore the error for now.
			// We will report it later, but we need to free all the buffers we can before that
			failed = true;
		}

		// Raw data from channel A (GPU)
		try {
			auto cuda_rc = cudaFree(m_chan_a_real_buffers[i]);
			utils::cuda_err_handle(cuda_rc, "cudaFree failed", __FILE__, __LINE__);
		}
		catch (std::exception& _e) {
			failed = true;
		}

		// Raw data from channel B (GPU)
		try {
			auto cuda_rc = cudaFree(m_chan_b_real_buffers[i]);
			utils::cuda_err_handle(cuda_rc, "cudaFree failed", __FILE__, __LINE__);
		}
		catch (std::exception& _e) {
			failed = true;
		}

		// Processed data from channel A (GPU)
		try {
			auto cuda_rc = cudaFree(m_chan_a_complex_buffers[i]);
			utils::cuda_err_handle(cuda_rc, "cudaFree failed", __FILE__, __LINE__);
		}
		catch (std::exception& _e) {
			failed = true;
		}

		// Processed data from channel B (GPU)
		try {
			auto cuda_rc = cudaFree(m_chan_b_complex_buffers[i]);
			utils::cuda_err_handle(cuda_rc, "cudaFree failed", __FILE__, __LINE__);
		}
		catch (std::exception& _e) {
			failed = true;
		}

		// Processed data from channel A (CPU)
		try {
			auto cuda_rc = cudaFreeHost(m_cpu_chan_a_complex_buffers[i]);
			utils::cuda_err_handle(cuda_rc, "cudaFreeHost failed", __FILE__, __LINE__);
		}
		catch (std::exception& _e) {
			failed = true;
		}

		// Processed data from channel B (CPU)
		try {
			auto cuda_rc = cudaFreeHost(m_cpu_chan_b_complex_buffers[i]);
			utils::cuda_err_handle(cuda_rc, "cudaFreeHost failed", __FILE__, __LINE__);
		}
		catch (std::exception& _e) {
			failed = true;
		}

		// Destory CUDA streams
		try {
			auto cuda_rc = cudaStreamDestroy(m_streams[i]);
			utils::cuda_err_handle(cuda_rc, "cudaStreamDestroy failed", __FILE__, __LINE__);
		}
		catch (std::exception& _e) {
			failed = true;
		}
	}
	m_did_alloc_memory = false;

	if (failed) {
		// Report the error that we saw earlier
		utils::log_error("Acquisition memory cleanup failed", __FILE__, __LINE__);
		throw std::runtime_error("Acquisition memory cleanup failed");
	}
}

bool Acquisition::is_finished() {
	return should_stop_acq;
}

Acquisition::~Acquisition() {
	try {
		cleanup();
	}
	catch (const std::exception& e) {
		// Never throw from a C++ destructor
		utils::log_error(std::string("Acquisition destructor caught error: ") + e.what());
	}
}