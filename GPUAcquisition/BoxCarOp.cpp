#include "BoxCarOp.hpp"
#include "Utils.hpp"

#include <stdexcept>

void gpu_boxcar_averager(cufftReal* input_buffer, cufftReal* output_buffer, cufftReal const* __restrict__ prev_buffer, const size_t samples_per_buffer, const size_t jump_ps, const size_t sample_duration_ps, const size_t phase_offset_ps, const size_t on_time_ps, const size_t off_time_ps);

BoxCarOp::BoxCarOp(size_t on_time_ps, size_t off_time_ps, size_t phase_offset_ps) :
	m_on_time_ps(on_time_ps),
	m_off_time_ps(off_time_ps),
	m_phase_offset_ps(phase_offset_ps)
{
	m_samples_per_buffer = 0;
	m_sample_duration_ps = 0;
	m_time_jump_ps = 0;
	m_prev_buffer = nullptr;
}

void BoxCarOp::init(size_t samples_per_buffer, size_t sample_duration_ps, size_t time_jump_ps) {
	m_samples_per_buffer = samples_per_buffer;
	m_sample_duration_ps = sample_duration_ps;
	m_time_jump_ps = time_jump_ps;
}

void BoxCarOp::set_prev_buffer(cufftReal* prev_buffer) {
	m_prev_buffer = prev_buffer;
}

void BoxCarOp::operate(void* input_buffer, void* output_buffer) {
	operate(static_cast<cufftReal*>(input_buffer), static_cast<cufftReal*>(output_buffer));
}

void BoxCarOp::operate(cufftReal* input_buffer, cufftReal* output_buffer) {
	if (m_sample_duration_ps == 0ULL) {
		utils::log_error("A sample duration must be given to the box car filter. Please call BoxCar::init before calling BoxCar::operate", __FILE__, __LINE__);
		throw std::runtime_error("A sample rate must be given to the box car filter. Please call BoxCar::init before calling BoxCar::operate");
	}
	gpu_boxcar_averager(input_buffer, output_buffer, m_prev_buffer, m_samples_per_buffer, m_time_jump_ps, m_sample_duration_ps, m_phase_offset_ps, m_on_time_ps, m_off_time_ps);
}

size_t BoxCarOp::get_window_time_ps() const {
	return m_on_time_ps + m_off_time_ps;
}