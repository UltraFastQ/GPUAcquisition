#include <cuda_runtime.h>
#include <cufft.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

#include "Utils.hpp"

#include <iostream>

__global__ void get(cufftReal* buff, size_t idx, cufftReal* result) {
	*result = buff[idx];
}

__global__ void set(cufftReal* buff, size_t idx, cufftReal value) {
	buff[idx] = value;
}

// TODO: Split this up into many kernel launches?
// __global__ void extract_pulses(cufftReal* input_buffer, cufftReal* output_buffer, const size_t samples_per_buffer, const size_t left_tail, const size_t right_tail, size_t* data_size) {
__host__ void gpu_extract_pulses(cufftReal* input_buffer, cufftReal* output_buffer, const size_t samples_per_buffer, const size_t left_tail, const size_t right_tail, size_t* data_size) {
	std::cout << "gpu_extract_pulses(" << input_buffer << ", " << output_buffer << ", " << samples_per_buffer << ", " << left_tail << ", " << right_tail << ")\n";
	
	cudaPointerAttributes attrs;
	cudaPointerGetAttributes(&attrs, input_buffer);
	std::cout << "attrs.device: " << attrs.device << ", attrs.devicePointer: " << attrs.devicePointer << ", attrs.hostPointer: " << attrs.hostPointer << ", attrs.type: " << attrs.type << ")\n";
	
	//std::cout << "thrust::minmax_element(" << input_buffer << ", " << input_buffer + samples_per_buffer-1 << ")\n";

	//// TODO: Search n/k of the search space? (Using some fancy iterators from thrust)
	//const auto minmax = thrust::minmax_element(input_buffer, input_buffer + samples_per_buffer-1);
	//std::cout << "minmax.first: " << minmax.first << ", minmax.second: " << minmax.second << "\n";
	//
	//const float min_elem = *minmax.first;
	//const float max_elem = *minmax.second;

	float min_elem = 0.f;
	float max_elem = 0.f;
	for (size_t i = 0; i < samples_per_buffer; ++i) {
		cufftReal curr = 41.1234f;
		get<<<1, 1 >>>(input_buffer, i, &curr);
		cudaDeviceSynchronize();
		//std::cout << curr << "\n";
		if (curr < min_elem) {
			min_elem = curr;
		}
		else if (curr > max_elem) {
			max_elem = curr;
		}
	}

	const float thresh = (max_elem + min_elem) / 2.f;

	std::cout << "min_elem: " << min_elem << ", max_elem: " << max_elem << ", thresh: " << thresh << "\n";

	size_t i = 0;
	size_t output_idx = 0;
	while (i < samples_per_buffer) {
		cufftReal curr = 60.1234f;
		get<<<1, 1>>>(input_buffer, i, &curr);
		cudaDeviceSynchronize();

		// If we are not above the pulse peak threshold, keep searching
		if (curr < thresh) {
			++i;
			continue;
		}

		// Curr is at or above the threshold now
		bool found_pulse = false;
		size_t j = i + 1;
		for (; j < samples_per_buffer; ++j) {
			cufftReal test = 73.1234f;
			get<<<1, 1>>>(input_buffer, j, &test);
			cudaDeviceSynchronize();

			// Keep searching until we dip below the threshold again
			if (test < thresh) {
				--j;
				found_pulse = true;
				break;
			}
		}
		if (!found_pulse) break;

		// We now have a pulse with width (j-i)+1 in idx units
		const size_t mid_idx = (i + j) / 2;
		const size_t fwhm = j - i + 1;

		const size_t left_end = left_tail * fwhm > mid_idx ? 0 : mid_idx - left_tail * fwhm;
		const size_t right_end = right_tail * fwhm + mid_idx >= samples_per_buffer ? samples_per_buffer - 1 : mid_idx + right_tail * fwhm;
		const size_t pulse_size = right_end - left_end;

		std::cout << "i: " << i << ", j: " << j << ", output_idx: " << output_idx << ", mid_idx: " << mid_idx << ", fwhm: " << fwhm << ", left_end: " << left_end << ", right_end: " << right_end << ", pulse_size: " << pulse_size << "\n";
		std::cout << "cudaMemcpy(" << output_buffer + output_idx << ", " << input_buffer + left_end << ", " << pulse_size * sizeof(*output_buffer) << ")\n";
		std::cout << "\n";

		// Copy the peak to the output buffer
		//for (size_t k = 0; k < pulse_size; ++k) {
		//	output_buffer[output_idx + k] = input_buffer[left_end + k];
		//}
		 auto rc = cudaMemcpy(output_buffer + output_idx, input_buffer + left_end, pulse_size * sizeof(*output_buffer), cudaMemcpyDeviceToDevice);
		 utils::cuda_err_handle(rc, "cudaMemcpy failed", __FILE__, __LINE__);

		// Add a NaN between pulses to delimit them
		output_idx += pulse_size + 1;
		set<<<1, 1 >>>(output_buffer, output_idx++, nanf(""));
		cudaDeviceSynchronize();

		// output_buffer[output_idx++] = nanf("");

		i = j + 1;
	}
	// Save how much data we wrote
	*data_size = output_idx;
}

//__host__ void gpu_extract_pulses(cufftReal* input_buffer, cufftReal* output_buffer, const size_t samples_per_buffer, const size_t left_tail, const size_t right_tail, size_t* data_size) {
//	extract_pulses << <1, 1 >> > (input_buffer, output_buffer, samples_per_buffer, left_tail, right_tail, data_size);
//}