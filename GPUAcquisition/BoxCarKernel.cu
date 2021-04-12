#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include "Utils.hpp"

static constexpr const unsigned block_size = 1024; // Max size afforded by the GTX Quadro 4000

struct IntegParam {
	size_t start_idx;
	size_t stop_idx;
	bool use_prev_buffer;
	size_t prev_buffer_idxs;
};

// NEW CODE
// __global__ void integrate_window(cufftReal* input_buffer, cufftReal* output_buffer, const size_t out_idx, const size_t start_idx, const size_t stop_idx, const size_t samples_per_buffer, const bool use_prev_buffer, const size_t prev_buffer_idxs, cufftReal const* __restrict__ prev_buffer) {
__global__ void integrate_window(cufftReal* input_buffer, cufftReal* output_buffer, const size_t num_windows, const size_t samples_per_buffer, cufftReal const* __restrict__ prev_buffer, IntegParam* params_arr) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_windows) return;

	IntegParam params = params_arr[idx];
	float integral = 0.f;

	// Use the information from the previous buffer if necessary
	if (params.use_prev_buffer) {
		integral += prev_buffer[samples_per_buffer - 1] * params.prev_buffer_idxs;
	}

	// Sum from start_idx to stop_idx
	for (size_t i = params.start_idx; i <= params.stop_idx; ++i) {
		integral += input_buffer[i];
	}

	// Average the integral over the time the window is open
	integral /= params.use_prev_buffer ? (params.stop_idx - params.start_idx + params.prev_buffer_idxs) : (params.stop_idx - params.start_idx);

	// Set the output buffer value to the integral
	output_buffer[idx] = integral;
}

__host__ void gpu_boxcar_averager(cufftReal* input_buffer, cufftReal* output_buffer, cufftReal const* __restrict__ prev_buffer, const size_t samples_per_buffer, const size_t time_jump, const size_t sample_duration, const size_t phase_offset, const size_t on_time, const size_t off_time) {
	size_t t = time_jump;

	// Are we still in the phase_offset zone?
	if (t < phase_offset) {
		// Skip ahead to when the window first opens
		t = phase_offset;
		if (t / sample_duration >= samples_per_buffer) return;
	}

	// TODO: Is this problematic?
	t -= phase_offset;

	size_t window_t = t % (on_time + off_time);
	if (window_t > on_time) { // Is the window closed?
		// Skip ahead to when the window opens
		t += on_time + off_time - window_t;
	}

	size_t num_windows = (samples_per_buffer * sample_duration - (t - time_jump)) / (on_time + off_time) + 1;
	
	std::vector<IntegParam> param_arr;
	param_arr.reserve(num_windows);
	IntegParam* param_arr_gpu;
	auto rc = cudaMalloc(&param_arr_gpu, num_windows * sizeof(*param_arr_gpu));
	utils::cuda_err_handle(rc, "cudaMalloc failed", __FILE__, __LINE__);

	size_t i = 0ULL;
	while (1) {
		// The window is open at this point in time

		size_t buffer_t = t - time_jump;
		const size_t start_idx = buffer_t / sample_duration;
		window_t = t % (on_time + off_time);

		// We can go until the window closes, or until we've hit the end of the buffer
		const size_t stop_time = min(buffer_t + on_time - window_t, (samples_per_buffer - 1) * sample_duration);
		const size_t stop_idx = stop_time / sample_duration;

		// Stop when we are outside of the buffer
		if (start_idx >= samples_per_buffer) break;

		bool use_prev_buffer = false;
		size_t prev_buffer_idxs = 0ULL;
		if (window_t != 0) { // Is the window only partially open?
			// The window is only partially open if it is the first window of any given buffer
			// We will have to use the information from the previous buffer to perform our averaging for this window
			use_prev_buffer = true;
			prev_buffer_idxs = window_t / sample_duration;
		}

		// Perform the boxcar over this window
		// TODO: Make this better
		// std::cout << "i: " << i << ", t: " << t << ", tj: " << time_jump << ", buff_t: " << buffer_t << ", start_i: " << start_idx << ", stop_i: " << stop_idx << ", spb: " << samples_per_buffer << ", use_prev: " << use_prev_buffer << ", prev_i: " << prev_buffer_idxs << ", prev_b: " << prev_buffer << "\n";
		param_arr.push_back({ start_idx, stop_idx, use_prev_buffer, prev_buffer_idxs });
		// integrate_window<<<1, 1>>>(input_buffer, output_buffer, out_idx, start_idx, stop_idx, samples_per_buffer, use_prev_buffer, prev_buffer_idxs, prev_buffer);
		++i;

		// Step forward in time, just past where we stopped calculating
		buffer_t = stop_time; // +(1 * sample_duration);

		// We are now right at the start of the window being closed.
		// We can skip ahead until the window opens again
		buffer_t += off_time;
		t = buffer_t + time_jump;
	}

	// Copy the data to the GPU
	rc = cudaMemcpy(param_arr_gpu, param_arr.data(), param_arr.size() * sizeof(param_arr[0]), cudaMemcpyHostToDevice);
	utils::cuda_err_handle(rc, "cudaMemcpy failed", __FILE__, __LINE__);

	// Perform the averaging
	size_t num_blocks = (num_windows + block_size - 1) / block_size;
	//std::cout << "num_windows: " << num_windows << ", block_size: " << block_size << ", num_blocks: " << num_blocks << "\n";
	integrate_window<<<block_size, num_blocks>>>(input_buffer, output_buffer, num_windows, samples_per_buffer, prev_buffer, param_arr_gpu);
	
	rc = cudaStreamSynchronize(0);
	utils::cuda_err_handle(rc, "cudaStreamSynchronize failed", __FILE__, __LINE__);

	rc = cudaFree(param_arr_gpu);
	utils::cuda_err_handle(rc, "cudaFree failed", __FILE__, __LINE__);
}

// OLD CODE
//__global__ void set_buffer(cufftReal* buffer, const size_t idx_offset, cufftReal value) {
//	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x + idx_offset;
//	buffer[idx] = value;
//}
//
//// Note: Launch this kernel with only 1 block
//__global__ void integrate_window(cufftReal* input_buffer, const size_t idx_offset, const size_t num_elements, cufftReal* sum) {
//	const size_t idx = threadIdx.x; // Assume blockIdx.x = 0 since only one block is used when launching this kernel
//
//	// Fill up an array of partial sums
//	__shared__ cufftReal partial_sums[1024];
//	cufftReal partial_sum = 0.f;
//	for (size_t i = idx; i < num_elements; i += block_size) {
//		partial_sum += input_buffer[i + idx_offset];
//	}
//	partial_sums[idx] = partial_sum;
//	__syncthreads(); // Wait for all the partial sums to be done
//
//	// Sum up the partial sums (serial operation in log(block_size) time)
//	for (size_t size = block_size / 2; size > 0; size /= 2) {
//		if (idx < size) {
//			partial_sums[idx] += partial_sums[idx + size];
//		}
//		__syncthreads();
//	}
//
//	// Place the computed sum in the sum pointer
//	if (idx == 0) {
//		*sum = partial_sums[0];
//	}
//}
//
//__host__ void gpu_boxcar_averager(cufftReal* input_buffer, cufftReal* output_buffer, cufftReal const* __restrict__ prev_buffer, const size_t samples_per_buffer, const size_t time_jump_ps, const size_t sample_duration_ps, const size_t phase_offset_ps, const size_t on_time_ps, const size_t off_time_ps) {
//	// Assume the jump has passed the phase_offset
//	size_t idx_offset = 0;
//
//	// Is the jump behind the phase_offset?
//	if (time_jump_ps <= phase_offset_ps) {
//		// How much time do we have until we pass the phase_offset?
//		const size_t time_rem_ps = phase_offset_ps - time_jump_ps;
//		const size_t idxs_rem = time_rem_ps / sample_duration_ps;
//		// Do we have enough indices left in our buffer to pass the phase_offset?
//		if (idxs_rem < samples_per_buffer) {
//			// Start off when we pass the phase_offset
//			idx_offset = idxs_rem;
//		}
//	}
//
//	// Current time
//	const size_t curr_time_ps = time_jump_ps + idx_offset * sample_duration_ps;
//
//	// This represents how far along the open/close cycle of the window we are
//	const size_t t = curr_time_ps % (on_time_ps + off_time_ps);
//
//	// Did we start in a partially open window?
//	if (t <= on_time_ps && t != 0) {
//		// How many indices left until the window closes?
//		const size_t window_size = utils::clamp((on_time_ps - t) / sample_duration_ps, 0ULL, samples_per_buffer - idx_offset);
//
//		// Integrate over the window
//		cufftReal* integral;
//		cudaHostAlloc(&integral, sizeof(*integral), cudaHostAllocDefault);
//		integrate_window<<<1, block_size>>>(input_buffer, idx_offset, utils::clamp((on_time_ps - t) / sample_duration_ps, 0ULL, samples_per_buffer - idx_offset), integral);
//
//		cudaStreamSynchronize(0);
//
//		// Was the integral started in the previous buffer?
//		if (prev_buffer != nullptr) {
//			// Grab previous integral
//			cufftReal prev_integral;
//			cudaMemcpy(&prev_integral, prev_buffer + samples_per_buffer - 1, sizeof(prev_integral), cudaMemcpyDeviceToHost);
//
//			// Remove averaging of previous integral
//			prev_integral *= static_cast<float>(t / sample_duration_ps);
//
//			// Average total integral over all time
//			*integral = (*integral + prev_integral) / static_cast<float>(on_time_ps / sample_duration_ps);
//		}
//		else {
//			// Average the integral over the time the window was open for
//			*integral /= static_cast<float>((on_time_ps - t) / sample_duration_ps);
//		}
//
//		// Fill out the output_buffer with the value calculated from the integral
//		set_buffer<<<1, window_size>>>(output_buffer, idx_offset, *integral); // The window is open
//
//		const size_t window_off_size = utils::clamp(off_time_ps / sample_duration_ps, 0ULL, samples_per_buffer - (idx_offset + window_size));
//		set_buffer<<<1, window_off_size>>>(output_buffer, idx_offset + window_size, 0.f); // The window is closed
//
//		cudaFreeHost(integral);
//
//		// Move on to the next window
//		// Increase our idx_offset until the window is open
//		idx_offset += ((on_time_ps + off_time_ps) - t) / sample_duration_ps;
//	}
//	else if (t != 0) {
//		// How many indices left until the window opens back up?
//		const size_t window_off_size = (on_time_ps + off_time_ps - t) / sample_duration_ps;
//
//		set_buffer<<<1, window_off_size>>>(output_buffer, idx_offset, 0.f); // The window is closed
//
//		idx_offset += ((on_time_ps + off_time_ps) - t) / sample_duration_ps;
//	}
//
//	// Now we know that the next window will be open for its full time (or until the buffer runs out)
//	
//	const size_t stride = (on_time_ps + off_time_ps) / sample_duration_ps;
//
//	cufftReal* integrals;
//	cudaHostAlloc(&integrals, ((samples_per_buffer - idx_offset + stride - 1) / stride) * sizeof(*integrals), cudaHostAllocDefault);
//	
//	for (size_t i = idx_offset, j = 0; i < samples_per_buffer; i += stride, ++j) {
//		// Integrate over the window
//		integrate_window<<<1, block_size>>>(input_buffer, i, utils::clamp(on_time_ps / sample_duration_ps, 0ULL, samples_per_buffer - i), &integrals[j]);		
//	}
//
//	cudaStreamSynchronize(0);
//
//	for (size_t i = idx_offset, j = 0; i < samples_per_buffer; i += stride, ++j) {
//		const size_t window_size = utils::clamp(on_time_ps / sample_duration_ps, 0ULL, samples_per_buffer - i);
//
//		// Now we average the intergral over the window size
//		integrals[j] /= static_cast<float>(window_size);
//
//		// Fill out the output_buffer with the value calculated from the integral
//		set_buffer<<<1, window_size>>>(output_buffer, i, integrals[j]); // The window is open
//
//		if (i + window_size < samples_per_buffer) { // Avoid integer underflow
//			const size_t window_off_size = utils::clamp(off_time_ps / sample_duration_ps, 0ULL, samples_per_buffer - (i + window_size));
//			set_buffer<<<1, window_off_size>>>(output_buffer, i + window_size, 0.f); // The window is closed
//		}
//	}
//
//	cudaFreeHost(integrals);
//}