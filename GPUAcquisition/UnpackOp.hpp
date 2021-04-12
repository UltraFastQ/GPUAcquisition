#ifndef __GPUACQUISITION_UNPACKOP_HPP__
#define __GPUACQUISITION_UNPACKOP_HPP__

#include <cufft.h>

#include "Operation.hpp"

class UnpackOp : public Operation {
public:
	UnpackOp(size_t num_bytes);
	void operate(unsigned char* input_buffer, cufftReal* chan_a_output_buffer, cufftReal* chan_b_output_buffer);
	void operate(void*, void*) override;
private:
	size_t m_num_bytes;
};

#endif /* __GPUACQUISITION_UNPACKOP_HPP__ */