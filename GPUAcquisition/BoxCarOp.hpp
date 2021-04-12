#ifndef __GPUACQUISITION_BOXCAROP_HPP__
#define __GPUACQUISITION_BOXCAROP_HPP__

#include <cufft.h>

#include "Operation.hpp"

class BoxCarOp : public Operation {
public:
	BoxCarOp(size_t on_time_ps, size_t off_time_ps, size_t phase_offset_ps);

	void init(size_t samples_per_buffer, size_t sample_duration_ps, size_t time_jump_ps);
	void set_prev_buffer(cufftReal* prev_buffer);

	void operate(void* input_buffer, void* ouputput_buffer) override;
	void operate(cufftReal* input_buffer, cufftReal* ouputput_buffer);

	size_t get_window_time_ps() const;

private:
	size_t m_samples_per_buffer;
	size_t m_sample_duration_ps;
	size_t m_time_jump_ps;
	cufftReal* m_prev_buffer;
	size_t m_on_time_ps; // The gate width of the box car (i.e. how long the window is ong for) in picoseconds
	size_t m_off_time_ps; // The period of time thw window is off for until the next rising edge of the boxcar (in picoseconds)
	size_t m_phase_offset_ps; // The delay of the very first rising edge of the first window (in picoseconds)
};

#endif /* __GPUACQUISITION_BOXCAROP_HPP__ */