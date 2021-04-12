#include "ExtractPulsesOp.hpp"

#include <stdexcept>

#include "Utils.hpp"

void gpu_extract_pulses(cufftReal* input_buffer, cufftReal* output_buffer, const size_t samples_per_buffer, const size_t left_tail, const size_t right_tail, size_t* data_size);

ExtractPulsesOp::ExtractPulsesOp(size_t left_tail, size_t right_tail) :
	m_left_tail(left_tail),
	m_right_tail(right_tail),
	m_samples_per_buffer(0ULL)
{}

void ExtractPulsesOp::init(size_t samples_per_buffer) {
	m_samples_per_buffer = samples_per_buffer;
}

void ExtractPulsesOp::operate(void* input_buffer, void* output_buffer) {
	operate(static_cast<cufftReal*>(input_buffer), static_cast<cufftReal*>(output_buffer));
}

void ExtractPulsesOp::operate(cufftReal* input_buffer, cufftReal* output_buffer) {
	if (m_samples_per_buffer == 0ULL) {
		utils::log_error("The number of samples per buffer must be given to extrat pulses. Please call ExtractPulsesOp::init before calling ExtractPulsesOp::operate", __FILE__, __LINE__);
		throw std::runtime_error("The number of samples per buffer must be given to extrat pulses. Please call ExtractPulsesOp::init before calling ExtractPulsesOp::operate");
	}
	gpu_extract_pulses(input_buffer, output_buffer, m_samples_per_buffer, m_left_tail, m_right_tail, &m_data_size);
}