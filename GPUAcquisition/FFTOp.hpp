#ifndef __OPERATION_HPP__
#define __OPERATION_HPP__

#include <cufft.h>

#include "Operation.hpp"

class FFTOp : public Operation {
public:
	FFTOp(size_t fft_size);
	~FFTOp();

	void operate(void* input_buffer, void* output_buffer) override;
	void operate(cufftReal* input_buffer, cufftComplex* output_buffer);

private:
	cufftHandle m_plan;
};

#endif /*__OPERATION_HPP__ */