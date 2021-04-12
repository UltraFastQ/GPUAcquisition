#ifndef __ACQUISITION_HPP__
#define __ACQUISITION_HPP__

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <fstream>
#include <variant>
#include <vector>

#include "AcquisitionConfig.hpp"
#include "BoxCarOp.hpp"
#include "ExtractPulsesOp.hpp"
#include "FFTOp.hpp"
#include "LIA.hpp"
#include "Operation.hpp"
#include "Utils.hpp"

namespace py = pybind11;

class Acquisition {
	using OpPtr = std::variant<FFTOp*, BoxCarOp*, LIA*, ExtractPulsesOp*>;
public:
	Acquisition(uint32_t alazar_system_id=1, uint32_t alazar_board_id=1, int gpu_device_id=0);
	~Acquisition();

	/* Configures the Alazar board to work with the GPU */
	void configure_devices(AcquisitionConfig* config=nullptr);
	
	/* Configures the operations that will be apllied to the data */
	void set_ops(std::vector<OpPtr>& ops_chan_a, std::vector<OpPtr>& ops_chan_b);

	/* Starts the acquisition of data, and streams it to the GPU */
	void start();

	/* Cleans up all the memory allocations that were made. This function should only be called if you can guarantee the object will be destroyed after. */
	void cleanup();

	/* Returns the latest buffer from the A channel */
	py::memoryview get_chan_a(const size_t num_points);

	/* Returns the latest buffer from the B channel */
	py::memoryview get_chan_b(const size_t num_points);

	/* Returns true if the acquisition is complete */
	bool is_finished();

private:
	/* Allocates the necessary data on the CPU and GPU */
	void alloc_mem();

	/* Handles data acquisition */
	void buffer_handler();

	// Alazar board paramteres
	uint32_t m_alazar_system_id;
	uint32_t m_alazar_board_id;
	HANDLE m_alazar_board;

	// GPU parameters
	int m_gpu_device_id;

	// Config
	AcquisitionConfig m_config;

	// Acquisition size parameters
	uint32_t m_samples_per_buffer; // Samples per GPU buffer
	uint32_t m_bytes_per_buffer; // Bytes per buffer
	uint32_t m_num_channels; // Number of active channels

	// GPU/CPU buffers
	std::vector<unsigned char*> m_raw_buffers; // Holds the raw acquisition data (allocated on the GPU by ATS-GPU)
	std::vector<cufftReal*> m_chan_a_real_buffers; // Holds the acquisition data coming from channel A as a real number (allocated on the GPU by CUDA)
	std::vector<cufftReal*> m_chan_b_real_buffers; // Holds the acquisition data coming from channel B as a real number (allocated on the GPU by CUDA)
	std::vector<cufftComplex*> m_chan_a_complex_buffers; // Holds the data coming from channel A after processing it (allocated on the GPU by CUDA)
	std::vector<cufftComplex*> m_chan_b_complex_buffers; // Holds the data coming from channel A after processing it (allocated on the GPU by CUDA)
	std::vector<cufftReal*> m_cpu_chan_a_real_buffers; // Holds the data coming from channel A after processing it on the CPU (allocated on the CPU by CUDA)
	std::vector<cufftReal*> m_cpu_chan_b_real_buffers; // Holds the data coming from channel A after processing it on the CPU (allocated on the CPU by CUDA)
	std::vector<cufftComplex*> m_cpu_chan_a_complex_buffers; // Holds the data coming from channel A after processing it on the CPU (allocated on the CPU by CUDA)
	std::vector<cufftComplex*> m_cpu_chan_b_complex_buffers; // Holds the data coming from channel A after processing it on the CPU (allocated on the CPU by CUDA)
	std::vector<cudaStream_t> m_streams; // One stream per buffer TODO: Actually use these

	// Operations
	std::vector<OpPtr> m_ops_chan_a;
	std::vector<OpPtr> m_ops_chan_b;

	// For error handling
	bool m_did_alloc_memory;
	bool m_did_start_capture;

	// For writing to disk
	std::ofstream m_out_file;
};

#endif /* __ACQUISITION_HPP__ */
