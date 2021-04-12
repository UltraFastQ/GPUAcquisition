#include <iostream>

#include <pybind11/pybind11.h>

#include "Utils.hpp"

namespace py = pybind11;

static const char* ANSI_RESET = "\033[0m";
static const char* ANSI_BOLD = "\033[1m";
static const char* ANSI_RED = "\033[31m";
static const char* ANSI_GREEN = "\033[32m";
static const char* ANSI_ORANGE = "\033[38;2;255;165;0m";

void utils::log_message(const char* msg, const char* const file, const unsigned line) {
    //py::print("[", ANSI_BOLD, ANSI_GREEN, "MESSAGE", ANSI_RESET, "] ", msg, py::arg("sep") = "", py::arg("end") = "");
    //if (file != "" && line != 0) {
    //    py::print(" (", file, ":", line, ")", py::arg("sep") = "", py::arg("flush") = true);
    //}
    //else {
    //    py::print(py::arg("flush") = true);
    //}
    std::cout << "[" << ANSI_BOLD << ANSI_GREEN << "MESSAGE" << ANSI_RESET << "] " << msg;
    if (file != "" && line != 0) {
        std::cout << " (" << file << ":" << line << ")";
    }
    std::cout << "\n";
}

void utils::log_message(const std::string& msg, const char* const file, const unsigned line) {
    //py::print("[", ANSI_BOLD, ANSI_GREEN, "MESSAGE", ANSI_RESET, "] ", msg, py::arg("sep") = "", py::arg("end") = "");
    //if (file != "" && line != 0) {
    //    py::print(" (", file, ":", line, ")", py::arg("sep") = "", py::arg("flush") = true);
    //}
    //else {
    //    py::print(py::arg("flush") = true);
    //}
    std::cout << "[" << ANSI_BOLD << ANSI_GREEN << "MESSAGE" << ANSI_RESET << "] " << msg;
    if (file != "" && line != 0) {
        std::cout << " (" << file << ":" << line << ")";
    }
    std::cout << "\n";
}

void utils::log_warning(const char* msg, const char* const file, const unsigned line) {
    //py::print("[", ANSI_BOLD, ANSI_ORANGE, "WARNING", ANSI_RESET, "] ", msg, py::arg("sep") = "", py::arg("end") = "");
    //if (file != "" && line != 0) {
    //    py::print(" (", file, ":", line, ")", py::arg("sep") = "", py::arg("flush") = true);
    //}
    //else {
    //    py::print(py::arg("flush") = true);
    //}
    std::cout << "[" << ANSI_BOLD << ANSI_ORANGE << "WARNING" << ANSI_RESET << "] " << msg;
    if (file != "" && line != 0) {
        std::cout << " (" << file << ":" << line << ")";
    }
    std::cout << "\n";
}

void utils::log_warning(const std::string& msg, const char* const file, const unsigned line) {
    //py::print("[", ANSI_BOLD, ANSI_ORANGE, "WARNING", ANSI_RESET, "] ", msg, py::arg("sep") = "", py::arg("end") = "");
    //if (file != "" && line != 0) {
    //    py::print(" (", file, ":", line, ")", py::arg("sep") = "", py::arg("flush") = true);
    //}
    //else {
    //    py::print(py::arg("flush") = true);
    //}
    std::cout << "[" << ANSI_BOLD << ANSI_ORANGE << "WARNING" << ANSI_RESET << "] " << msg;
    if (file != "" && line != 0) {
        std::cout << " (" << file << ":" << line << ")";
    }
    std::cout << "\n";
}

void utils::log_error(const char* msg, const char* const file, const unsigned line) {
    //py::print("[", ANSI_BOLD, ANSI_RED, " ERROR ", ANSI_RESET, "] ", msg, py::arg("sep") = "", py::arg("end") = "");
    //if (file != "" && line != 0) {
    //    py::print(" (", file, ":", line, ")", py::arg("sep") = "", py::arg("flush") = true);
    //}
    //else {
    //    py::print(py::arg("flush") = true);
    //}
    std::cout << "[" << ANSI_BOLD << ANSI_RED << " ERROR " << ANSI_RESET << "] " << msg;
    if (file != "" && line != 0) {
        std::cout << " (" << file << ":" << line << ")";
    }
    std::cout << "\n";
}

void utils::log_error(const std::string& msg, const char* const file, const unsigned line) {
    //py::print("[", ANSI_BOLD, ANSI_RED, " ERROR ", ANSI_RESET, "] ", msg, py::arg("sep") = "", py::arg("end") = "");
    //if (file != "" && line != 0) {
    //    py::print(" (", file, ":", line, ")", py::arg("sep") = "", py::arg("flush") = true);
    //}
    //else {
    //    py::print(py::arg("flush") = true);
    //}
    std::cout << "[" << ANSI_BOLD << ANSI_RED << " ERROR " << ANSI_RESET << "] " << msg;
    if (file != "" && line != 0) {
        std::cout << " (" << file << ":" << line << ")";
    }
    std::cout << "\n";
}

void utils::log_break() {
    //py::print();
	std::cout << "\n";
}

void utils::alazar_err_handle(RETURN_CODE rc, const char* const msg, const char* const file, const unsigned line) {
    if (rc != ApiSuccess) {
        std::string error_msg = std::string(msg) + " -- Error code: " + AlazarErrorToText(rc) + " (" + file + ":" + std::to_string(line) + ")";
        utils::log_error(error_msg.c_str());
        throw std::runtime_error(error_msg);
    }
}

const char* utils::get_board_type_string(uint32_t board_type) {
    switch (board_type) {
        case ATS9373:
            return "ATS9373";
        case ATS850:
            return "ATS850";
        case ATS310:
            return "ATS310";
        case ATS330:
            return "ATS330";
        case ATS855:
            return "ATS855";
        case ATS315:
            return "ATS315";
        case ATS335:
            return "ATS335";
        case ATS460:
            return "ATS460";
        case ATS860:
            return "ATS860";
        case ATS660:
            return "ATS660";
        case ATS9461:
            return "ATS9461";
        case ATS9462:
            return "ATS9462";
        case ATS9850:
            return "ATS9850";
        case ATS9870:
            return "ATS9870";
        case ATS9310:
            return "ATS9310";
        case ATS9325:
            return "ATS9325";
        case ATS9350:
            return "ATS9350";
        case ATS9351:
            return "ATS9351";
        case ATS9360:
            return "ATS9360";
        case ATS9410:
            return "ATS9410";
        case ATS9440:
            return "ATS9440";
        case ATS9625:
            return "ATS9625";
        case ATS9626:
            return "ATS9626";
        case ATS9370:
            return "ATS9370";
        case ATS9416:
            return "ATS9416";
        default:
            return "?";
    }
}

void utils::cuda_err_handle(cudaError_t code, const char* const msg, const char* const file, const unsigned line) {
    if (code != cudaSuccess) {
        std::string error_msg = std::string(msg) + " -- Error code: " + cudaGetErrorString(code) + " (" + file + ":" + std::to_string(line) + ")";
        utils::log_error(error_msg.c_str());
        throw std::runtime_error(error_msg);
    }
}

static const char* cufft_get_err_string(cufftResult_t code) {
	switch (code) {
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";
	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";
	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";
	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";
	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";
	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";
	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";
	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";
	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";
	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	default:
		return "?";
	}
}

void utils::cufft_err_handle(cufftResult_t code, const char* const msg, const char* const file, const unsigned line) {
	if (code != CUFFT_SUCCESS) {
		std::string error_msg = std::string(msg) + " -- Error code: " + cufft_get_err_string(code) + " (" + file + ":" + std::to_string(line) + ")";
		utils::log_error(error_msg.c_str());
		throw std::runtime_error(error_msg);
	}
}