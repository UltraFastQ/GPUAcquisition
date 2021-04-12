#ifndef __GPUACQUISITION_EXTRACTPULSESOP_HPP__
#define __GPUACQUISITION_EXTRACTPULSESOP_HPP__

#include <cufft.h>

#include "Operation.hpp"

class ExtractPulsesOp : public Operation {
public:
	ExtractPulsesOp(size_t left_tail, size_t right_tail);

	void init(size_t samples_per_buffer);

	void operate(void* input_buffer, void* output_buffer) override;
	void operate(cufftReal* input_buffer, cufftReal* output_buffer);

	size_t get_data_size();

private:
	size_t m_samples_per_buffer;
	size_t m_left_tail; // How much data to extract before the peak of the pulse in units of FWHM
	size_t m_right_tail; // How much data to extract after the peak of the pulse in units of FWHM
	size_t m_data_size; // HOw many samples are there in the extracted data
};

#endif /* __GPUACQUISITION_EXTRACT_PULSESOP_HPP__ */