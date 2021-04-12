#include <stdexcept>
#include <tuple>

#include <ATS_GPU.h>

#include "AcquisitionConfig.hpp"
#include "Utils.hpp"

namespace py = pybind11;

AcquisitionConfig::AcquisitionConfig() {
	// Parameters for AlazarSeCaptureClock
	capture_clock.source = INTERNAL_CLOCK; // EXTERNAL_CLOCK_10MHZ_REF; // INTERNAL_CLOCK;
	capture_clock.sample_rate = SAMPLE_RATE_2000MSPS;
	capture_clock.edge = CLOCK_EDGE_RISING;
	capture_clock.decimation = 0;

	// Parameters for AlazarInputControl
	input_control = std::vector<AcquisitionConfig::InputControl>(2, AcquisitionConfig::InputControl());

	input_control[0].channel = CHANNEL_A;
	input_control[0].coupling = DC_COUPLING;
	input_control[0].input_range = INPUT_RANGE_PM_400_MV;
	input_control[0].impedance = IMPEDANCE_50_OHM;

	input_control[1].channel = CHANNEL_B;
	input_control[1].coupling = DC_COUPLING;
	input_control[1].input_range = INPUT_RANGE_PM_400_MV;
	input_control[1].impedance = IMPEDANCE_50_OHM;

	// Parameters for AlazarSetTriggerOperation
	trigger_operation.trigger_operation = TRIG_ENGINE_OP_J;
	trigger_operation.trigger_engine1 = TRIG_ENGINE_J;
	trigger_operation.source1 = TRIG_EXTERNAL;
	trigger_operation.slope1 = TRIGGER_SLOPE_POSITIVE;
	trigger_operation.level1 = 128;
	trigger_operation.trigger_engine2 = TRIG_ENGINE_K;
	trigger_operation.source2 = TRIG_DISABLE;
	trigger_operation.slope2 = TRIGGER_SLOPE_POSITIVE;
	trigger_operation.level2 = 128;

	// Parameters for AlazarSetExternalTrigger
	external_trigger.coupling = DC_COUPLING;
	external_trigger.range = ETR_TTL;

	// Parameters for AlazarSetTriggerDelay
	trigger_delay = 0;

	// Parameters for AlazarSetTriggerTimeOut
	trigger_timeout_ticks = 0;

	// Parameters for AlazarConfigureAuxIO
	aux_io.mode = AUX_OUT_TRIGGER;
	aux_io.parameter = 0;

	// Parameters for AlazarSetParameter(..., ..., PACK_MODE, ...)
	pack_mode = PACK_12_BITS_PER_SAMPLE;

	// Parameters for ATS_GPU_Setup
	acquisition_setup.channels = CHANNEL_A | CHANNEL_B;
	acquisition_setup.transfer_offset = 0;
	acquisition_setup.pre_trigger_samples = 0;
	acquisition_setup.post_trigger_samples = 0; // 8192000
	acquisition_setup.records_per_buffer = 1; // Keep at 1
	acquisition_setup.records_per_acquisition = 0x7FFFFFFF; //1; // 100, 0x7FFFFFFFFF...
	acquisition_setup.adma_flags = ADMA_TRIGGERED_STREAMING | ADMA_EXTERNAL_STARTCAPTURE;
	acquisition_setup.gpu_flags = ATS_GPU_SETUP_FLAG_DEINTERLEAVE; // | ATS_GPU_SETUP_FLAG_UNPACK;

	num_gpu_buffers = 0; // 12

	data_writing.fname = "";
	data_writing.num_buffs_to_write = 0;
}

AcquisitionConfig::AcquisitionConfig(py::dict config) : AcquisitionConfig::AcquisitionConfig() {
	auto set = [&]<typename T>(const char* key1, const char* key2, T& what) {
		if (key2 == "") {
			// if (py::isinstance<py::int_>(static_cast<py::object>(config[key1]))) {
				what = static_cast<T>(static_cast<py::object>(config[key1]).cast<uint32_t>());
			// }
			/*else {
				utils::log_error("Configuration parameter " + static_cast<std::string>(key1) + " is not an int or an instance of the correct Alazar enum", __FILE__, __LINE__);
				throw py::key_error(key1);
			}*/
		}
		else if (config[key1].contains(key2)) {
			// if (py::isinstance<py::int_>(static_cast<py::object>(config[key1][key2]))) {
				what = static_cast<T>(static_cast<py::object>(config[key1][key2]).cast<uint32_t>());
			// }
			/*else {
				utils::log_error("Configuration parameter \"" + static_cast<std::string>(key1) + "." + static_cast<std::string>(key2) + "\" is not an int or an instance of the correct Alazar enum", __FILE__, __LINE__);
				throw py::key_error(key1 + static_cast<std::string>(".") + key2);
			}*/
		}
	};

	auto set_from_list = [&]<typename T>(py::detail::list_iterator iterator, const char* key, T& what) {
		if (py::isinstance<py::dict>(*iterator)) {
			if (iterator->contains(key)) {
				// if (py::isinstance<py::int_>(static_cast<py::object>((*iterator)[key]))) {
					what = static_cast<T>(static_cast<py::object>((*iterator)[key]).cast<uint32_t>());
				// }
				/*else {
					utils::log_error("Configuration parameter " + static_cast<std::string>(key) + " is not an int or an instance of the correect Alazar enum", __FILE__, __LINE__);
					throw py::key_error(key);
				}*/
			}
		}
		else {
			utils::log_error("Configuration parameter is not a dict", __FILE__, __LINE__);
			throw std::runtime_error("Configuration parameter is not a dict");
		}
	};

	if (config.contains("capture_clock")) {
		set("capture_clock", "source", capture_clock.source);
		set("capture_clock", "sample_rate", capture_clock.sample_rate);
		set("capture_clock", "edge", capture_clock.edge);
		set("capture_clock", "decimation", capture_clock.decimation);
	}

	if (config.contains("input_control")) {
		if (py::isinstance<py::list>(static_cast<py::object>(config["input_control"]))) {
			input_control = std::vector<AcquisitionConfig::InputControl>(static_cast<py::list>(config["input_control"]).size(), input_control[0]);
			for (auto [i, it] = std::tuple{ 0, static_cast<py::list>(config["input_control"]).begin() }; it != static_cast<py::list>(config["input_control"]).end(); ++it, ++i) {
				set_from_list(it, "channel", input_control[i].channel);
				set_from_list(it, "input_range", input_control[i].input_range);
				set_from_list(it, "coupling", input_control[i].coupling);
				set_from_list(it, "impedance", input_control[i].impedance);
			}
		}
		else {
			input_control = std::vector<AcquisitionConfig::InputControl>(1, input_control[0]);
			set("input_control", "channel", input_control[0].channel);
			set("input_control", "input_range", input_control[0].input_range);
			set("input_control", "coupling", input_control[0].coupling);
			set("input_control", "impedance", input_control[0].impedance);
		}
	}

	if (config.contains("trigger_operation")) {
		set("trigger_operation", "trigger_operation", trigger_operation.trigger_operation);
		set("trigger_operation", "trigger_engine1", trigger_operation.trigger_engine1);
		set("trigger_operation", "source1", trigger_operation.source1);
		set("trigger_operation", "slope1", trigger_operation.slope1);
		set("trigger_operation", "level1", trigger_operation.level1);
		set("trigger_operation", "trigger_engine2", trigger_operation.trigger_engine2);
		set("trigger_operation", "source2", trigger_operation.source2);
		set("trigger_operation", "slope2", trigger_operation.slope2);
		set("trigger_operation", "level2", trigger_operation.level2);
	}

	if (config.contains("external_triger")) {
		set("external_trigger", "coupling", external_trigger.coupling);
		set("external_trigger", "range", external_trigger.range);
	}

	if (config.contains("trigger_delay")) {
		set("trigger_delay", "", trigger_delay);
	}

	if (config.contains("trigger_timeout_ticks")) {
		set("trigger_timeout_ticks", "", trigger_delay);
	}

	if (config.contains("aux_io")) {
		set("aux_io", "mode", aux_io.mode);
		set("aux_io", "parameter", aux_io.parameter);
	}

	if (config.contains("pack_mode")) {
		set("pack_mode", "", pack_mode);
	}

	if (config.contains("acquisition_setup")) {
		set("acquisition_setup", "channels", acquisition_setup.channels);
		set("acquisition_setup", "transfer_offset", acquisition_setup.transfer_offset);
		set("acquisition_setup", "pre_trigger_samples", acquisition_setup.pre_trigger_samples);
		set("acquisition_setup", "post_trigger_samples", acquisition_setup.post_trigger_samples);
		set("acquisition_setup", "records_per_buffer", acquisition_setup.records_per_buffer);
		set("acquisition_setup", "records_per_acquisition", acquisition_setup.records_per_acquisition);
		set("acquisition_setup", "adma_flags", acquisition_setup.adma_flags);
		set("acquisition_setup", "gpu_flags", acquisition_setup.gpu_flags);
	}

	if (config.contains("num_gpu_buffers")) {
		set("num_gpu_buffers", "", num_gpu_buffers);
	}

	if (config.contains("data_writing")) {
		data_writing.fname = static_cast<std::string>(static_cast<py::object>(config["data_writing"]["fname"]).cast<std::string>());
		set("data_writing", "num_buffs_to_write", data_writing.num_buffs_to_write);
	}
}

std::string AcquisitionConfig::as_str() {
	std::string str = "{\n";

	str += "    \"capture_clock\":\n";
	str += "    {\n";
	str += "        \"source\": " + py::cast<std::string>(py::cast(capture_clock.source).attr("__str__")()) + ",\n";
	str += "        \"sample_rate\": " + py::cast<std::string>(py::cast(capture_clock.sample_rate).attr("__str__")()) + ",\n";
	str += "        \"edge\": " + py::cast<std::string>(py::cast(capture_clock.edge).attr("__str__")()) + ",\n";
	str += "        \"decimation\": " + py::cast<std::string>(py::cast(capture_clock.decimation).attr("__str__")()) + "\n";
	str += "    },\n";

	str += "    \"input_control\":\n";
	str += "    [\n";
	for (auto it = input_control.begin(); it != input_control.end(); ++it) {
		str += "        {\n";
		str += "            \"channel\": " + py::cast<std::string>(py::cast(it->channel).attr("__str__")()) + ",\n";
		str += "            \"input_range\": " + py::cast<std::string>(py::cast(it->input_range).attr("__str__")()) + ",\n";
		str += "            \"coupling\": " + py::cast<std::string>(py::cast(it->coupling).attr("__str__")()) + ",\n";
		str += "            \"impedance\": " + py::cast<std::string>(py::cast(it->impedance).attr("__str__")()) + "\n";
		str += "        }" + static_cast<std::string>((std::distance(it, input_control.end()) != 1) ? ",\n" : "\n");
	}
	str += "    ],\n";

	str += "    \"trigger_operation\":\n";
	str += "    {\n";
	str += "        \"trigger_operation\": " + py::cast<std::string>(py::cast(trigger_operation.trigger_operation).attr("__str__")()) + ",\n";
	str += "        \"trigger_engine1\": " + py::cast<std::string>(py::cast(trigger_operation.trigger_engine1).attr("__str__")()) + ",\n";
	str += "        \"source1\": " + py::cast<std::string>(py::cast(trigger_operation.source1).attr("__str__")()) + ",\n";
	str += "        \"slope1\": " + py::cast<std::string>(py::cast(trigger_operation.slope1).attr("__str__")()) + ",\n";
	str += "        \"level1\": " + py::cast<std::string>(py::cast(trigger_operation.level1).attr("__str__")()) + ",\n";
	str += "        \"trigger_engine2\": " + py::cast<std::string>(py::cast(trigger_operation.trigger_engine2).attr("__str__")()) + ",\n";
	str += "        \"source2\": " + py::cast<std::string>(py::cast(trigger_operation.source2).attr("__str__")()) + ",\n";
	str += "        \"slope2\": " + py::cast<std::string>(py::cast(trigger_operation.slope2).attr("__str__")()) + ",\n";
	str += "        \"level2\": " + py::cast<std::string>(py::cast(trigger_operation.level2).attr("__str__")()) + "\n";
	str += "    },\n";

	str += "    \"external_trigger\":\n";
	str += "    {\n";
	str += "        \"coupling\": " + py::cast<std::string>(py::cast(external_trigger.coupling).attr("__str__")()) + ",\n";
	str += "        \"range\": " + py::cast<std::string>(py::cast(external_trigger.range).attr("__str__")()) + "\n";
	str += "    },\n";

	str += "    \"trigger_delay\": " + py::cast<std::string>(py::cast(trigger_delay).attr("__str__")()) + ",\n";

	str += "    \"trigger_timeout_ticks\": " + py::cast<std::string>(py::cast(trigger_timeout_ticks).attr("__str__")()) + ",\n";

	str += "    \"aux_io\":\n";
	str += "    {\n";
	str += "        \"mode\": " + py::cast<std::string>(py::cast(aux_io.mode).attr("__str__")()) + ",\n";
	str += "        \"parameter\": " + py::cast<std::string>(py::cast(aux_io.parameter).attr("__str__")()) + "\n";
	str += "    },\n";

	str += "    \"pack_mode\": " + py::cast<std::string>(py::cast(pack_mode).attr("__str__")()) + ",\n";

	str += "    \"acquisition_setup\":\n";
	str += "    {\n";
	str += "        \"channels\": " + py::cast<std::string>(py::cast(acquisition_setup.channels).attr("__str__")()) + ",\n";
	str += "        \"transfer_offset\": " + py::cast<std::string>(py::cast(acquisition_setup.transfer_offset).attr("__str__")()) + ",\n";
	str += "        \"pre_trigger_samples\": " + py::cast<std::string>(py::cast(acquisition_setup.pre_trigger_samples).attr("__str__")()) + ",\n";
	str += "        \"post_trigger_samples\": " + py::cast<std::string>(py::cast(acquisition_setup.post_trigger_samples).attr("__str__")()) + ",\n";
	str += "        \"records_per_buffer\": " + py::cast<std::string>(py::cast(acquisition_setup.records_per_buffer).attr("__str__")()) + ",\n";
	str += "        \"records_per_acquisition\": " + py::cast<std::string>(py::cast(acquisition_setup.records_per_acquisition).attr("__str__")()) + ",\n";
	str += "        \"adma_flags\": " + py::cast<std::string>(py::cast(acquisition_setup.adma_flags).attr("__str__")()) + ",\n";
	str += "        \"gpu_flags\": " + py::cast<std::string>(py::cast(acquisition_setup.gpu_flags).attr("__str__")()) + "\n";
	str += "    },\n";

	str += "    \"num_gpu_buffers\": " + py::cast<std::string>(py::cast(num_gpu_buffers).attr("__str__")()) + ",\n";

	str += "    \"data_writing\":\n";
	str += "    {\n";
	str += "        \"fname\": " + py::cast<std::string>(py::cast(data_writing.fname).attr("__repr__")()) + ",\n";
	str += "        \"num_buffs_to_write\": " + py::cast<std::string>(py::cast(data_writing.num_buffs_to_write).attr("__str__")()) + "\n";
	str += "    }\n";

	str += "}";

	return str;
}