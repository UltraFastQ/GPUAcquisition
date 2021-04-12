#include <stdexcept>

#include "UnpackOp.hpp"
#include "Utils.hpp"

void gpu_unpack_data(const unsigned char* const raw_data, cufftReal* const chan_a, cufftReal* const chan_b, const size_t num_bytes, const cudaStream_t stream);

UnpackOp::UnpackOp(size_t num_bytes) : m_num_bytes(num_bytes) {}

void UnpackOp::operate(unsigned char* input_buffer, cufftReal* chan_a, cufftReal* chan_b) {
	// TODO: Different stream?
	gpu_unpack_data(input_buffer, chan_a, chan_b, m_num_bytes, 0);
}

void UnpackOp::operate(void*, void*) {
	utils::log_error("UnpackOp::operate is a special Operation and requires 3 inputs(input_buffer, chan_a, chan_b)", __FILE__, __LINE__);
	throw std::runtime_error("UnpackOp::operate is a special Operation and requires 3 inputs(input_buffer, chan_a, chan_b)");
}
