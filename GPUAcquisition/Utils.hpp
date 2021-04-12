#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <cufft.h>

#include "AlazarApi.h"
#include "AlazarCmd.h"
#include "AlazarError.h"

namespace utils {
    /* Logs a message to the console. */
    void log_message(const char* msg, const char* const file="", const unsigned line=0);
    void log_message(const std::string& msg, const char* file = "", const unsigned line=0);

    /* Logs a warning to the console. */
    void log_warning(const char* msg, const char* const file="", const unsigned line=0);
    void log_warning(const std::string& msg, const char* file="", const unsigned line=0);

    /* Logs an error to the console. */
    void log_error(const char* msg, const char* const file="", const unsigned line=0);
    void log_error(const std::string& msg, const char* file="", const unsigned line=0);

	/* Logs a new line. */
	void log_break();

    /* If an Alazar function does not return ApiSuccess, log that information on to the console. Throws std::runtime_error in case of failure. */
    void alazar_err_handle(RETURN_CODE rc, const char* const msg, const char* const file, const unsigned line);

    /* Converts a BoardType (an enum) to a human readable string representation */
    const char* get_board_type_string(uint32_t board_type);

    /* If a CUDA function does not return cudaSuccess, log that information on to the console. Throws std::runtime_error in case of failure. */
    void cuda_err_handle(cudaError_t code, const char* const msg, const char* const file, const unsigned line);

	/* If a cuFFT function does not return CUFFT_SUCCESS, log that nformation on to the console. Throws std::runtime_error in case of failue.*/
	void cufft_err_handle(cufftResult_t code, const char* const msg, const char* const file, const unsigned line);

    /* Restricts x to the interval: [lower, upper] */
    template <typename T>
    T clamp(T x, T lower, T upper) {
        if (x < lower) {
            return lower;
        }
        else if (x > upper) {
            return upper;
        }
        return x;
    }

    template <typename T>
    std::string to_hex(T num) {
        static_assert(std::is_integral<T>::value, "num must be a primitive integer type");
    
        std::stringstream stream;
        stream << "0x" << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex << num;
        return stream.str();
    }
}

#endif /* __UTILS_HPP__ */
