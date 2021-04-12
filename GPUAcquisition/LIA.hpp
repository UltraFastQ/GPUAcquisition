#ifndef __GPUACQUISITION_LIA_HPP__
#define __GPUACQUISITION_LIA_HPP__

#include <cufft.h>

#include "Operation.hpp"

class LIA : public Operation {
public:
	LIA(size_t samples_per_buffer, size_t time_jump_ps, size_t sample_period_ps, float freq);

	void operate(void* input_buffer, void* ouputput_buffer) override;
	void operate(cufftReal* input_buffer, cufftComplex* ouputput_buffer);

private:
	size_t m_samples_per_buffer;
	size_t m_time_jump_ps;
	size_t m_sample_period_ps;
	float m_freq;
};

#endif /* __GPUACQUISITION_LIA_HPP__ */