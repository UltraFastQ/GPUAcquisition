#include <cuda_runtime.h>
#include <cufft.h>

__global__ void unpack_data_dual_channel(const unsigned char* const raw_data, cufftReal* const chan_a, cufftReal* const chan_b, const size_t num_bytes) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Make sure we are at the start of a new sample (every 3 bytes is a sample
	// from each channel) and that we are not at the end of our data
	if (idx % 3 != 0 || idx + 2 >= num_bytes) {
		return;
	}

	// The data is stored in three bytes on the card
	const unsigned char b0 = raw_data[idx + 0];
	const unsigned char b1 = raw_data[idx + 1];
	const unsigned char b2 = raw_data[idx + 2];

	// The data from the card is stored in little endian:
	// Data on card: AB CD EF
	// Var name:     b0 b1 b2
	// Channel A:    0D AB
	// Channel B:    0E FC
	const unsigned int a_data = ((b1 & 0x0F) << 8) | b0;
	const unsigned int b_data = ((b2 << 8) | (b1 & 0xF0)) >> 4;

	// We also want to convert it to a float (a cufftReal type)
	// TODO: Pass the value of 400 mV to this function
	chan_a[(idx / 3)] = ((static_cast<float>(a_data) / 4096.f) - 0.5f) * 2.f * 0.400f; // 400 mV
	chan_b[(idx / 3)] = ((static_cast<float>(b_data) / 4096.f) - 0.5f) * 2.f * 0.400f; // 400 mV
}


__global__ void unpack_data_single_channel(const unsigned char* const raw_data, cufftReal* const chan, const size_t num_bytes) {
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Make sure we are at the start of a new sample (every 3 bytes is a sample
	// from each channel) and that we are not at the end of our data
	if (idx % 3 != 0 || idx + 2 >= num_bytes) {
		return;
	}

	// The data is stored in three bytes on the card
	const unsigned char b0 = raw_data[idx + 0];
	const unsigned char b1 = raw_data[idx + 1];
	const unsigned char b2 = raw_data[idx + 2];

	// The data from the card is stored in little endian:
	// Data on card: AB CD EF
	// Var name:     b0 b1 b2
	// Channel A:    0D AB
	// Channel B:    0E FC
	const unsigned int a_data = ((b1 & 0x0F) << 8) | b0;
	const unsigned int b_data = ((b2 << 8) | (b1 & 0xF0)) >> 4;

	// We also want to convert it to a float (a cufftReal type)
	// TODO: Pass the value of 400 mV to this function
	chan[(idx / 3)] = ((static_cast<float>(a_data) / 4096.f) - 0.5f) * 2.f * 0.400f; // 400 mV
	chan[(idx / 3) + 1] = ((static_cast<float>(b_data) / 4096.f) - 0.5f) * 2.f * 0.400f; // 400 mV
}

void gpu_unpack_data_dual_channel(const unsigned char* const raw_data, cufftReal* const chan_a, cufftReal* const chan_b, const size_t num_bytes, const cudaStream_t stream) {
	constexpr const unsigned threads_per_block = 1024; // Max size afforded by the GTX Quadro 4000
	const unsigned num_blocks = static_cast<unsigned>(ceil(static_cast<double>(num_bytes) / static_cast<double>(threads_per_block)));

	unpack_data_dual_channel<<<num_blocks, threads_per_block, 0, stream>>>(raw_data, chan_a, chan_b, num_bytes);
}

void gpu_unpack_data_single_channel(const unsigned char* const raw_data, cufftReal* const chan, const size_t num_bytes, const cudaStream_t stream) {
	constexpr const unsigned threads_per_block = 1024; // Max size afforded by the GTX Quadro 4000
	const unsigned num_blocks = static_cast<unsigned>(ceil(static_cast<double>(num_bytes) / static_cast<double>(threads_per_block)));

	unpack_data_single_channel<<<num_blocks, threads_per_block, 0, stream>>>(raw_data, chan, num_bytes);
}