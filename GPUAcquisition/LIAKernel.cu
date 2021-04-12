#include <cufft.h>

#include "Utils.hpp"

// TODO: This is not integrating anything
// Reference: https://www.zhinst.com/americas/resources/principles-of-lock-in-detection
__global__ void lia(cufftReal* input_buffer, cufftComplex* output_buffer, size_t time_jump_ps, size_t sample_period_ps, float freq) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	const size_t time_ps = time_jump_ps + sample_period_ps * static_cast<size_t>(idx);

	float s = 0.f;
	float c = 0.f;
	sincosf(static_cast<float>(time_ps) * 1e-12f, &s, &c); // sin(t) and cos(t) at the same time

	// TODO:
	// float x = low_pass_filter(c * input_buffer[idx]);
	// float y = low_pass_filter(s * input_buffer[idx]);

	float x = c * input_buffer[idx];
	float y = s * input_buffer[idx];

	float r = sqrtf(x * x + y * y); // TODO: only store r^2? (sqrt is slow)
	float theta = atan2(y, x);

	output_buffer[idx].x = r;
	output_buffer[idx].x = theta;
}

void gpu_lia(cufftReal* input_buffer, cufftComplex* output_buffer, size_t samples_per_buffer, size_t time_jump_ps, size_t sample_period_ps, float freq) {
	constexpr const unsigned threads_per_block = 1024; // Max size afforded by the GTX Quadro 4000
	const unsigned num_blocks = static_cast<unsigned>(ceil(static_cast<double>(samples_per_buffer) / static_cast<double>(threads_per_block)));
	lia<<<num_blocks, threads_per_block>>>(input_buffer, output_buffer, time_jump_ps, sample_period_ps, freq);
}