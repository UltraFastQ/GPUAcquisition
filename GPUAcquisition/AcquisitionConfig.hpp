#ifndef __ACQUISITIONCONFIG_HPP__
#define __ACQUISITIONCONFIG_HPP__

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AlazarApi.h"
#include "AlazarCmd.h"
#include "AlazarError.h"

namespace py = pybind11;

struct AcquisitionConfig {
	AcquisitionConfig();
	AcquisitionConfig(py::dict config);

	std::string as_str();

	// Parameters for AlazarSetCaptureClock
	struct CaptureClock {
		ALAZAR_CLOCK_SOURCES source;
		ALAZAR_SAMPLE_RATES sample_rate;
		ALAZAR_CLOCK_EDGES edge;
		uint32_t decimation;
	};
	CaptureClock capture_clock;

	// Parameters for AlazarInputControl
	struct InputControl {
		ALAZAR_CHANNELS channel;
		ALAZAR_INPUT_RANGES input_range;
		ALAZAR_COUPLINGS coupling;
		ALAZAR_IMPEDANCES impedance;
	};
	std::vector<InputControl> input_control;

	// Parameters for AlazarSetTriggerOperation
	struct TriggerOperation {
		ALAZAR_TRIGGER_OPERATIONS trigger_operation;
		ALAZAR_TRIGGER_ENGINES trigger_engine1;
		ALAZAR_TRIGGER_SOURCES source1;
		ALAZAR_TRIGGER_SLOPES slope1;
		uint32_t level1;
		ALAZAR_TRIGGER_ENGINES trigger_engine2;
		ALAZAR_TRIGGER_SOURCES source2;
		ALAZAR_TRIGGER_SLOPES slope2;
		uint32_t level2;
	};
	TriggerOperation trigger_operation;

	// Parameters for AlazarSetExternalTrigger
	struct ExternalTrigger {
		ALAZAR_COUPLINGS coupling;
		ALAZAR_EXTERNAL_TRIGGER_RANGES range;
	};
	ExternalTrigger external_trigger;

	// Paramters for AlazarSetTriggerDelay
	uint32_t trigger_delay;

	// Parameters for AlazarSetTriggerTimeOut
	uint32_t trigger_timeout_ticks;

	// Parameters for AlazarConfigureAuxIO
	struct AuxIO {
		ALAZAR_AUX_IO_MODES mode;
		uint32_t parameter;
	};
	AuxIO aux_io;

	// Parameters for AlazarSetParameter(..., ..., PACK_MODE, ...)
	ALAZAR_PACK_MODES pack_mode;

	// Parameters for ATS_GPU_Setup
	struct AcquisitionSetup {
		uint32_t channels;
		long transfer_offset;
		uint32_t pre_trigger_samples;
		uint32_t post_trigger_samples;
		uint32_t records_per_buffer;
		uint32_t records_per_acquisition;
		uint32_t adma_flags;
		uint32_t gpu_flags;
	};
	AcquisitionSetup acquisition_setup;

	uint32_t num_gpu_buffers;

	// Parameters for saving data to disk
	struct DataWriting {
		std::string fname;
		uint32_t num_buffs_to_write; // Will write the first n buffers
	};
	DataWriting data_writing;
};

#endif /* __ACQUISITION_HPP__ */