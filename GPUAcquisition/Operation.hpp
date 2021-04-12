#ifndef __GPUACQUISITION_OPERATION_HPP__
#define __GPUACQUISITION_OPERATION_HPP__

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class Operation {
public:
	virtual void operate(void* input_buffer, void* output_buffer) = 0;
};

class PyOperation : public Operation {
public:
	using Operation::Operation;

	void operate(void* input_buffer, void* output_buffer) override {
		PYBIND11_OVERLOAD_PURE(
			void,         /* Return type */
			Operation,    /* Parent class */
			operate,      /* Name of C++ function */
			input_buffer, /* First argument */
			output_buffer /* Second argument */
		);
	}
};

#endif /* __GPUACQUISITION_OPERATION_HPP__ */