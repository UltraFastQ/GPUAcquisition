#include <cuda_runtime.h>
#include <cufft.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

#include "Utils.hpp"

#include <iostream>

__global__ void extract_pulses(cufftReal* input_buffer, cufftReal* output_buffer, const size_t samples_per_buffer, const size_t left_tail, const size_t right_tail) {
	cufftReal min_elem = input_buffer[0];
	cufftReal max_elem = input_buffer[0];
	for (size_t i = 1; i < samples_per_buffer; ++i) {
		if (input_buffer[i] < min_elem) min_elem = input_buffer[i];
		else if (input_buffer[i] > max_elem) max_elem = input_buffer[i];
	}

	const cufftReal thresh = (min_elem + max_elem) / 2.f;
	size_t i = 0;
	size_t output_idx = 0;
	while (i < samples_per_buffer) {
		cufftReal curr = input_buffer[i];

		// If we are not above the pulse peak threshold, keep searching
		if (curr < thresh) {
			++i;
			continue;
		}

		// curr is at or above the threshold now
		bool found_pulse = false;
		size_t j = i + 1;
		for (; j < samples_per_buffer; ++j) {
			cufftReal test = input_buffer[j];

			// Keep searching until we dip below the threshold again
			if (test < thresh) {
				--j;
				found_pulse = true;
				break;
			}
		}

		if (!found_pulse) break;

		// We now have a pulse of width (j-i)+1 in idx units
		const size_t mid_idx = (i + j) / 2;
		const size_t fwhm = j - i + 1;

		const size_t left_end = left_tail * fwhm > mid_idx ? 0 : mid_idx - left_tail * fwhm;
		const size_t right_end = right_tail * fwhm + mid_idx >= samples_per_buffer ? samples_per_buffer - 1 : mid_idx + right_tail * fwhm;
		const size_t pulse_size = right_end - left_end;

		// Copy the peak to the output buffer
		for (size_t k = 0; k < pulse_size; ++k) {
			if ((output_idx + k) < samples_per_buffer && (left_end + k) < samples_per_buffer) {
				output_buffer[output_idx + k] = input_buffer[left_end + k];
			}
		}

		// Add a NaN between pulses to delimit them
		output_idx += pulse_size + 1;
		if (output_idx < samples_per_buffer) {
			output_buffer[output_idx++] = nanf("");
		}

		// Skip to the end of the pulse
		i = j + 1;
	}
	// *data_size = output_idx;
}

// TODO: Split this up into many kernel launches?
// __global__ void extract_pulses(cufftReal* input_buffer, cufftReal* output_buffer, const size_t samples_per_buffer, const size_t left_tail, const size_t right_tail, size_t* data_size) {
__host__ void gpu_extract_pulses(cufftReal* input_buffer, cufftReal* output_buffer, const size_t samples_per_buffer, const size_t left_tail, const size_t right_tail, size_t* data_size) {
	extract_pulses<<<1, 1>>>(input_buffer, output_buffer, samples_per_buffer, left_tail, right_tail);
}