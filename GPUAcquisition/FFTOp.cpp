#include <stdexcept>

#include "FFTOp.hpp"
#include "Utils.hpp"

FFTOp::FFTOp(size_t fft_size) {
	auto cufft_rc = cufftCreate(&m_plan);
	utils::cufft_err_handle(cufft_rc, "cufftCreate failed", __FILE__, __LINE__);

	cufft_rc = cufftPlan1d(&m_plan, static_cast<int>(fft_size), CUFFT_R2C, 1);
	utils::cufft_err_handle(cufft_rc, "cufftPlan1d failed", __FILE__, __LINE__);

	// TODO: Change the stream?
	cufft_rc = cufftSetStream(m_plan, 0);
	utils::cufft_err_handle(cufft_rc, "cufftSetStream failed", __FILE__, __LINE__);
}

void FFTOp::operate(void* input_buffer, void* output_buffer) {
	operate(static_cast<cufftReal*>(input_buffer), static_cast<cufftComplex*>(output_buffer));
}

void FFTOp::operate(cufftReal* input_buffer, cufftComplex* output_buffer) {
	auto cufft_rc = cufftExecR2C(m_plan, input_buffer, output_buffer);
	utils::cufft_err_handle(cufft_rc, "cufftExecR2C failed", __FILE__, __LINE__);
}

FFTOp::~FFTOp() {
	try {
		auto cufft_rc = cufftDestroy(m_plan);
		utils::cufft_err_handle(cufft_rc, "cufftDestroy failed", __FILE__, __LINE__);
	}
	catch (const std::exception& e) {
		// Never throw from a C++ destructor
		utils::log_error(std::string("FFTOp destructor caught error: ") + e.what());
	}
}