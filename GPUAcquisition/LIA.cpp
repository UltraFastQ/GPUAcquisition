#include "LIA.hpp"
#include "Utils.hpp"

#include <stdexcept>

void gpu_lia(cufftReal* input_buffer, cufftComplex* output_buffer, size_t samples_per_buffer, size_t time_jump_ps, size_t sample_period_ps, float freq);

LIA::LIA(size_t samples_per_buffer, size_t time_jump_ps, size_t sample_period_ps, float freq) :
	m_samples_per_buffer(samples_per_buffer),
	m_time_jump_ps(time_jump_ps),
	m_sample_period_ps(sample_period_ps),
	m_freq(freq)
{}

void LIA::operate(void* input_buffer, void* output_buffer) {
	operate(static_cast<cufftReal*>(input_buffer), static_cast<cufftComplex*>(output_buffer));
}

void LIA::operate(cufftReal* input_buffer, cufftComplex* output_buffer) {
	gpu_lia(input_buffer, output_buffer, m_samples_per_buffer, m_time_jump_ps, m_sample_period_ps, m_freq);
}