#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATS_GPU.h>

#include "Acquisition.hpp"
#include "AcquisitionConfig.hpp"
#include "AlazarInfo.hpp"
#include "BoxCarOp.hpp"
#include "ExtractPulsesOp.hpp"
#include "GPUInfo.hpp"
#include "LIA.hpp"
#include "Operation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(GPUAcquisition, m) {
    m.doc() = ""
        "GPU accelerated acquisition module designed to be used with AlazarTech waveform\n"
        "digitizers.\n\n"
        ""
        "This module contains an Acquisition class which is the main interface used to\n"
        "perform a data acquisition. There are other classes, such as the Filter class\n"
        "(and its derivatives), who's use are all explained within the Acquisition class\n"
        "documentation. This module contains two submodules: ATS and info. ATS is to be\n"
        "used with the Acquisition class (see the Acquisition class documentation for\n"
        "more information). The info module, on the other hand, should be used in an\n"
        "interactive prompt, as follows:\n\n"
        ""
        ">>> import GPUAcquisition\n"
        ">>> GPUAcquisition.info.display_alazar_info()\n"
        ">>> GPUAcquisition.info.display_gpu_info()\n\n"
        ""
        "These two functions will inform you of the hardware installed on your system\n"
        "and some of the parameters you may wish to use when configuring your instance of\n"
        "the Acquisition class. You may wish to consult the documentation of the info\n"
        "module itself to better understand the data that will be presented by above two\n"
        "functions.";

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif /* VERSION_INFO */

    py::module info_mod = m.def_submodule("info", ""
        "Contains functions that display information about the system hardware.\n\n"
        ""
        "The display_alazar_info() function prints (to stdout) a summary of the Alazar\n"
        "hardware installd on the system. The display_gpu_info() funcntion prints (to\n"
        "stdout) a summary of the GPU hardware installed on the system. You should use\n"
        "run these functions in an interactive shell to get a better idea of the\n"
        "parameters to pass into the rest of the Acquisition module functions as they\n"
        "will inform you of the limitations of the hardware availabble on the system."
    );
    info_mod.def("display_alazar_info", &alz::display_alazar_info, ""
        "Shows all AlazarTech devices connected to the machine.\n\n"
        ""
        "This is typically used in an interactive shell to know the hardware installed on\n"
        "the system, which should reveal the limitations of the system hardware. You may\n"
        "wish to consult the ATS SDK guide, available at the followinng link\n"
        "https://www.alazartech.com/Support/Download%20Files/ATS-SDK-Guide.pdf\n"
        "to better understand the hardware limitations of the system."
    );
    info_mod.def("display_gpu_info", &gpu::display_cuda_info, ""
        "Shows information about the GPUs connected to the machine.\n\n"
        ""
        "This is typically used in an interactive shell to know the hardware installed on\n"
        "the system, which should reveal the limitations of the system hardware. You way\n"
        "wish to consult NVIDIA's website, available at the following link\n"
        "https://www.nvidia.com/\n to better understand the hardware limitations of the\n"
        "system."
    );

    py::class_<Operation, PyOperation>(m, "Operation", ""
        "TODO: Documentation"
        )
        .def(py::init<>())
        .def("operate", &Operation::operate, ""
            "TODO: Documentation",
            py::arg("input_buffer"), py::arg("output_buffer")
        );

    py::class_<FFTOp, Operation>(m, "FFTOp", ""
        "TODO:Documentation"
        )
        .def(py::init<size_t>(), ""
            "TODO: Documentation"
            , py::arg("fft_size")
        );

    py::class_<BoxCarOp, Operation>(m, "BoxCarOp", ""
        "Box car averager operation to be applied upon data acquisition.\n\n"
        ""
        "A boxcar averager is a rectangular window function (in the time domain) that\n"
        "multiplies through your signal. This has the effect of letting your signal pass\n"
        "while the window is open (i.e. while the window is high), and letting no signal\n"
        "through while the window is closed (i.e. while the window is closed). This is\n"
        "useful for a signal which is periodic in time, since you can have the window\n"
        "while the signal is active, and close the window while the signal if inactive.\n"
        "This will ensure that much less noise passes through, increasing your SNR. For\n"
        "more information about how boxcar filters work, check out the Wikipedia article:\n"
        "https://en.wikipedia.org/wiki/Boxcar_averager"
        )
        .def(py::init<size_t, size_t, size_t>(), ""
            "Constructor for a BoxCar filter.\n\n"
            ""
            "Below is an illustration of all the parameters of this BoxCar filter, where the\n"
            "x-axis is time, and the y-axis is dimensionless. Note that the peak of the\n"
            "boxcar filter is at 1, such the signal is not perturbed in any way.\n\n"
            ""
            "                     |< on_time_ps >|                                           \n"
            "                      _______________                   ________________        \n"
            "                     |               |                 |                |       \n"
            "                     |               |                 |                |       \n"
            "                     |               |                 |                |       \n"
            "                     |               |                 |                |       \n"
            "|< phase_offset_ps >||               ||< off_time_ps >||                |       \n"
            "_____________________|               |_________________|                |_______\n\n"
            ""
            "Args:\n"
            "    on_time_ps (int):\n"
            "        The width of the filter in picoseconds. This is also referred to as the\n"
            "        gate width of the filter. See above for an illustration.\n"
            "    off_time_ps (int):\n"
            "        The time between two windows open windows in picoseconds. See above for\n"
            "        an illustration.\n"
            "    phase_offset_ps (int):\n"
            "        How long to wait (in picoseconds) for the very first window to open. A\n"
            "        value of 0 means that the filter should begin immediately with an open\n"
            "        window. This parameter is never used after the very first window is\n"
            "         opened. See above for an illustration.",
            py::arg("on_time_ps"), py::arg("off_time_ps"), py::arg("phase_offset_ps"));

    py::class_<ExtractPulsesOp, Operation>(m, "ExtractPulsesOp", ""
        "TODO:Documentation"
        )
        .def(py::init<size_t, size_t>(), ""
            "TODO: Documentation",
            py::arg("left_tail "), py::arg("right_tail"));

    //py::class_<LIA, Filter>(m, "LIA", ""
    //    "TODO: Documentation"
    //    )
    //    .def(py::init<float>(), ""
    //        "Constructor for a LIA (lock-in amplifier).\n\n"
    //        ""
    //        "TODO: Documentation (and finish the implementation)\n\n"
    //        ""
    //        "Args:\n"
    //        "    freq (float):\n"
    //        "        The frequency of the LIA.",
    //        py::arg("freq")
    //    );

    py::class_<Acquisition>(m, "Acquisition", ""
        "Acquisition class used for data acquisition with the ATS9373.\n\n"
        ""
        "This class should only be used as a singleton. Creating multiple Acqiusition\n"
        "objects may lead to too much memory being allocated on the GPU which can cause\n"
        "bugs that are very hard to track down.\n\n"
        ""
        "This class has 2 important methods you need to call prior to calling start():\n"
        "configure_devices() and set_filter() (this one is actually optional). If both\n"
        "are to be called, set_filter() should be called last. These are important\n"
        "functions as they allocate the necessary resources both on the ATS9373 as well\n"
        "as on the GPU to ensure fast and proper data acquisition.\n\n"
        ""
        "Below is a typical usage of the Acquisition class:\n"
        ">>> import GPUAcquisition\n"
        ">>> acq = GPUAcquisition.Acquisition()\n"
        ">>> acq.configure_devices() # TODO: configure_devices API\n"
        ">>> filter = GPUAcquisition.BoxCar(1, 100, 0)\n"
        ">>> acq.set_filter(filter)\n"
        ">>> acq.start()\n\n"
        ""
        "TODO: Ensure the destructor for this class is called before the end of the\n"
        "program (or before an exception is raised). Do we need to do\n"
        ">>> del acq\n"
        "Or will the program call the destructor? TODO"
        "TODO: Ensure the destructor for this class is called before the end of the\n"
        "program (or before an exception is raised)."
        )
        .def(py::init<uint32_t, uint32_t, int>(), ""
            "Constructor for the Acquisition class.\n\n"
            ""
            "Args:\n"
            "    alazar_system_id (int, default=1):\n"
            "        The system ID for the Alazar system the acquisition card finds itself\n"
            "        in. Please refer to the Alazar SDK guide for more information. To find\n"
            "        out which system ID your Alazar acquisition card is in, please call\n"
            "        GPUAcquisition.info.display_alazar_info() from within an interactive\n"
            "        shell. Note that if you have only one Alazar card plugged in to your\n"
            "        system, the default value of 1 will assuredly work.\n"
            "    alazar_board_id (int, default=1):\n"
            "        The board ID for the Alazar card within the above specified Alazar\n"
            "        system. Please refer to the Alazar SDK guide for more information. To\n"
            "        find out which board ID your Alazar acquisition card corresponds to,\n"
            "        please call GPUAcquisition.info.display_alazar_info() from within an\n"
            "        interactive shell. Note that if you have only one Alazar card plugged in\n"
            "        to your system, the default value of 1 will assuredly work.\n"
            "    gpu_device_id (int, default=0):\n"
            "        The GPU device to be used to enable GPU acceleration. You may wish to\n"
            "        refer to NVIDIA's CUDA documentation (specifically the cudaSetDevice()\n"
            "        function). To know which GPU device you can use, please call\n"
            "        GPUAcquisition.info.display_gpu_info() from within an interactive shell.\n"
            "        Note that if you have only one GPU plugged in to your system, the\n"
            "        default value of 0 will assuredly work.",
            py::arg("alazar_system_id") = 1, py::arg("alazar_board_id") = 1, py::arg("gpu_device_id") = 0)
        .def("configure_devices", &Acquisition::configure_devices, ""
            "Configures the ATS9373 for data acquisition onto the GPU.\n\n"
            ""
            "TODO: Interface for Acquisition::configure_devices\n"
            "TODO: Actual documentation",
            py::arg("conf") = nullptr
        )
        .def("set_ops", &Acquisition::set_ops, ""
            "TODO: Documentation",
            py::arg("operations_chan_a"), py::arg("operations_chan_b")
        )
        .def("start", &Acquisition::start, ""
            "Starts the data acquisition.\n\n"
            ""
            "You should have called Acquisition.configure_devices() earlier to configure\n"
            "how this acquisition will go, and possibly have called\n"
            "Acquisition.set_filter() in order to use a filter (if desired). The data\n"
            "gets acquired by the Alazar card and processed by the GPU (i.e. a Fourier\n"
            "transform is applied to the data).\n"
            "TODO: Return the data back to the user (it is already back on the CPU)."
        )
        .def("cleanup", &Acquisition::cleanup, ""
            "TODO: Remove this function from the python API"
        )
        .def("get_chan_a", &Acquisition::get_chan_a, ""
            "TODO: Documentation",
            py::arg("num_points")
        )
        .def("get_chan_b", &Acquisition::get_chan_b, ""
            "TODO: Documentation",
            py::arg("num_points")
        )
        .def("is_finished", &Acquisition::is_finished, ""
            "TODO: Documentation"
        );
    // TODO: __enter__ and __exit__ for context manager

    py::module conf_module = m.def_submodule("config", ""
        "TODO: Documentation"
    );

    py::class_<AcquisitionConfig::CaptureClock>(conf_module, "CaptureClock", ""
        "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::CaptureClock::source, ""
            "TODO: Documentation"
        )
        .def_readwrite("sample_rate", &AcquisitionConfig::CaptureClock::sample_rate, ""
            "TODO: Documentation"
        )
        .def_readwrite("edge", &AcquisitionConfig::CaptureClock::edge, ""
            "TODO: Documentation"
        )
        .def_readwrite("decimation", &AcquisitionConfig::CaptureClock::decimation, ""
            "TODO: Documentation"
        );

    py::class_<AcquisitionConfig::InputControl>(conf_module, "InputControl", ""
        "TODO: Documentation"
        )
        .def_readwrite("channel", &AcquisitionConfig::InputControl::channel, ""
            "TODO: Documentation"
        )
        .def_readwrite("input_range", &AcquisitionConfig::InputControl::input_range, ""
            "TODO: Documentation"
        )
        .def_readwrite("coupling", &AcquisitionConfig::InputControl::coupling, ""
            "TODO: Documentation"
        )
        .def_readwrite("impedance", &AcquisitionConfig::InputControl::impedance, ""
            "TODO: Documentation"
        );

    py::class_<AcquisitionConfig::TriggerOperation>(conf_module, "TriggerOperation", ""
        "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::trigger_operation, ""
            "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::trigger_engine1, ""
            "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::source1, ""
            "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::slope1, ""
            "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::level1, ""
            "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::trigger_engine2, ""
            "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::source2, ""
            "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::slope2, ""
            "TODO: Documentation"
        )
        .def_readwrite("source", &AcquisitionConfig::TriggerOperation::level2, ""
            "TODO: Documentation"
        );

    py::class_<AcquisitionConfig::ExternalTrigger>(conf_module, "ExternalTrigger", ""
        "TODO: Documentation"
        )
        .def_readwrite("coupling", &AcquisitionConfig::ExternalTrigger::coupling, ""
            "TODO: Documentation"
        )
        .def_readwrite("range", &AcquisitionConfig::ExternalTrigger::range, ""
            "TODO: Documentation"
        );

    py::class_<AcquisitionConfig::AuxIO>(conf_module, "AuxIO", ""
        "TODO: Documentation"
        )
        .def_readwrite("mode", &AcquisitionConfig::AuxIO::mode, ""
            "TODO: Documentation"
        )
        .def_readwrite("parameter", &AcquisitionConfig::AuxIO::parameter, ""
            "TODO: Documentation"
        );

    py::class_<AcquisitionConfig::AcquisitionSetup>(conf_module, "AcquisitionSetup", ""
        "TODO: Documentation"
        )
        .def_readwrite("channels", &AcquisitionConfig::AcquisitionSetup::channels, ""
            "TODO: Documentation"
        )
        .def_readwrite("transfer_offset", &AcquisitionConfig::AcquisitionSetup::transfer_offset, ""
            "TODO: Documentation"
        )
        .def_readwrite("pre_trigger_samples", &AcquisitionConfig::AcquisitionSetup::pre_trigger_samples, ""
            "TODO: Documentation"
        )
        .def_readwrite("post_trigger_samples", &AcquisitionConfig::AcquisitionSetup::post_trigger_samples, ""
            "TODO: Documentation"
        )
        .def_readwrite("records_per_buffer", &AcquisitionConfig::AcquisitionSetup::records_per_buffer, ""
            "TODO: Documentation"
        )
        .def_readwrite("records_per_acquisition", &AcquisitionConfig::AcquisitionSetup::records_per_acquisition, ""
            "TODO: Documentation"
        )
        .def_readwrite("adma_flags", &AcquisitionConfig::AcquisitionSetup::adma_flags, ""
            "TODO: Documentation"
        )
        .def_readwrite("gpu_flags", &AcquisitionConfig::AcquisitionSetup::gpu_flags, ""
            "TODO: Documentation"
        );

    py::class_<AcquisitionConfig::DataWriting>(conf_module, "DataWriting", ""
        "TODO: Documentation"
        )
        .def_readwrite("fname", &AcquisitionConfig::DataWriting::fname, ""
            "TODO: Documentation"
        )
        .def_readwrite("num_buffs_to_write", &AcquisitionConfig::DataWriting::num_buffs_to_write, ""
            "TODO: Documentation"
        );

    py::class_<AcquisitionConfig>(conf_module, "AcquisitionConfig", ""
            "TODO: Documentation"
            )
        .def(py::init<>(), ""
            "TODO: Documentation"
        )
        .def(py::init<py::dict>(), ""
            "TODO: Documentation",
            py::arg("config")
        )
        .def("__str__", &AcquisitionConfig::as_str, ""
            "TODO: Documentation"
        )
        .def_readwrite("capture_clock", &AcquisitionConfig::capture_clock)
        .def_readwrite("input_control", &AcquisitionConfig::input_control)
        .def_readwrite("trigger_operation", &AcquisitionConfig::trigger_operation)
        .def_readwrite("external_trigger", &AcquisitionConfig::external_trigger)
        .def_readwrite("trigger_delay", &AcquisitionConfig::trigger_delay)
        .def_readwrite("trigger_timeout_ticks", &AcquisitionConfig::trigger_timeout_ticks)
        .def_readwrite("aux_io", &AcquisitionConfig::aux_io)
        .def_readwrite("pack_mode", &AcquisitionConfig::pack_mode)
        .def_readwrite("acquisition_setup", &AcquisitionConfig::acquisition_setup)
        .def_readwrite("num_gpu_buffers", &AcquisitionConfig::num_gpu_buffers)
        .def_readwrite("data_writing", &AcquisitionConfig::data_writing);

    py::module ats_module = m.def_submodule("ATS", ""
        "Alazar enums.\n\n"
        ""
        "These are to be used with Acquisition.configure_devices() to specify the\n"
        "acquisition type, clock speed, sample rate, etc. Refer to the Alazar SDK guide\n"
        "as well as the Acquisition.configure_devices() documentation for more\n"
        "information."
    );

    py::enum_<ALAZAR_ADMA_MODES>(ats_module, "ALAZAR_ADMA_MODES", py::arithmetic())
        .value("ADMA_TRADITIONAL_MODE", ADMA_TRADITIONAL_MODE)
        .value("ADMA_CONTINUOUS_MODE", ADMA_CONTINUOUS_MODE)
        .value("ADMA_NPT", ADMA_NPT)
        .value("ADMA_TRIGGERED_STREAMING", ADMA_TRIGGERED_STREAMING)
        .export_values();

    py::enum_<ALAZAR_ADMA_FLAGS>(ats_module, "ALAZAR_ADMA_FLAGS", py::arithmetic())
        .value("ADMA_EXTERNAL_STARTCAPTURE", ADMA_EXTERNAL_STARTCAPTURE)
        .value("ADMA_ENABLE_RECORD_HEADERS", ADMA_ENABLE_RECORD_HEADERS)
        .value("ADMA_ALLOC_BUFFERS", ADMA_ALLOC_BUFFERS)
        .value("ADMA_FIFO_ONLY_STREAMING", ADMA_FIFO_ONLY_STREAMING)
        .value("ADMA_INTERLEAVE_SAMPLES", ADMA_INTERLEAVE_SAMPLES)
        .value("ADMA_GET_PROCESSED_DATA", ADMA_GET_PROCESSED_DATA)
        .value("ADMA_DSP", ADMA_DSP)
        .value("ADMA_ENABLE_RECORD_FOOTERS", ADMA_ENABLE_RECORD_FOOTERS)
        .export_values();

    py::enum_<ALAZAR_AUX_IO_MODES>(ats_module, "ALAZAR_AUX_IO_MODES")
        .value("AUX_OUT_TRIGGER", AUX_OUT_TRIGGER)
        .value("AUX_IN_TRIGGER_ENABLE", AUX_IN_TRIGGER_ENABLE)
        .value("AUX_OUT_PACER", AUX_OUT_PACER)
        .value("AUX_OUT_SERIAL_DATA", AUX_OUT_SERIAL_DATA)
        .value("AUX_IN_AUXILIARY", AUX_IN_AUXILIARY)
        .export_values();

    py::enum_<ALAZAR_LSB>(ats_module, "ALAZAR_LSB")
        .value("LSB_DEFAULT", LSB_DEFAULT)
        .value("LSB_EXT_TRIG", LSB_EXT_TRIG)
        .value("LSB_AUX_IN_1", LSB_AUX_IN_1)
        .value("LSB_AUX_IN_2", LSB_AUX_IN_2)
        .export_values();

    py::enum_<ALAZAR_CRA_MODES>(ats_module, "ALAZAR_CRA_MODES")
        .value("CRA_MODE_DISABLE", CRA_MODE_DISABLE)
        .value("CRA_MODE_ENABLE_FPGA_AVE", CRA_MODE_ENABLE_FPGA_AVE)
        .export_values();

    py::enum_<ALAZAR_CRA_OPTIONS>(ats_module, "ALAZAR_CRA_OPTIONS")
        .value("CRA_OPTION_UNSIGNED", CRA_OPTION_UNSIGNED)
        .value("CRA_OPTION_SIGNED", CRA_OPTION_SIGNED)
        .export_values();

    py::enum_<ALAZAR_SAMPLE_SKIPPING_MODES>(ats_module, "ALAZAR_SAMPLE_SKIPPING_MODES")
        .value("SSM_DISABLE", SSM_DISABLE)
        .value("SSM_ENABLE", SSM_ENABLE)
        .export_values();

    py::enum_<ALAZAR_COPROCESSOR_DOWNLOAD_OPTIONS>(ats_module, "ALAZAR_COPROCESSOR_DOWNLOAD_OPTIONS")
        .value("CPF_OPTION_DMA_DOWNLOAD", CPF_OPTION_DMA_DOWNLOAD)
        .export_values();

    py::enum_<ALAZAR_ECC_MODES>(ats_module, "ALAZAR_ECC_MODES")
        .value("ECC_DISABLE", ECC_DISABLE)
        .value("ECC_ENABLE", ECC_ENABLE)
        .export_values();

    py::enum_<ALAZAR_AUX_INPUT_LEVELS>(ats_module, "ALAZAR_AUX_INPUT_LEVELS")
        .value("AUX_INPUT_LOW", AUX_INPUT_LOW)
        .value("AUX_INPUT_HIGH", AUX_INPUT_HIGH)
        .export_values();

    py::enum_<ALAZAR_PACK_MODES>(ats_module, "ALAZAR_PACK_MODES")
        .value("PACK_DEFAULT", PACK_DEFAULT)
        .value("PACK_8_BITS_PER_SAMPLE", PACK_8_BITS_PER_SAMPLE)
        .value("PACK_12_BITS_PER_SAMPLE", PACK_12_BITS_PER_SAMPLE)
        .export_values();

    py::enum_<ALAZAR_API_TRACE_STATES>(ats_module, "ALAZAR_API_TRACE_STATES")
        .value("API_ENABLE_TRACE", API_ENABLE_TRACE)
        .value("API_DISABLE_TRACE", API_DISABLE_TRACE)
        .export_values();

    py::enum_<ALAZAR_ADC_MODES>(ats_module, "ALAZAR_ADC_MODES")
        .value("ADC_MODE_DEFAULT", ADC_MODE_DEFAULT)
        .value("ADC_MODE_DES", ADC_MODE_DES)
        .export_values();

    py::enum_<ALAZAR_CHANNELS>(ats_module, "ALAZAR_CHANNELS", py::arithmetic())
        .value("CHANNEL_ALL", CHANNEL_ALL)
        .value("CHANNEL_A", CHANNEL_A)
        .value("CHANNEL_B", CHANNEL_B)
        .value("CHANNEL_C", CHANNEL_C)
        .value("CHANNEL_D", CHANNEL_D)
        .value("CHANNEL_E", CHANNEL_E)
        .value("CHANNEL_F", CHANNEL_F)
        .value("CHANNEL_G", CHANNEL_G)
        .value("CHANNEL_H", CHANNEL_H)
        .value("CHANNEL_I", CHANNEL_I)
        .value("CHANNEL_J", CHANNEL_J)
        .value("CHANNEL_K", CHANNEL_K)
        .value("CHANNEL_L", CHANNEL_L)
        .value("CHANNEL_M", CHANNEL_M)
        .value("CHANNEL_N", CHANNEL_N)
        .value("CHANNEL_O", CHANNEL_O)
        .value("CHANNEL_P", CHANNEL_P)
        .export_values();

    py::enum_<ALAZAR_INPUT_RANGES>(ats_module, "ALAZAR_INPUT_RANGES")
        .value("INPUT_RANGE_PM_20_MV", INPUT_RANGE_PM_20_MV)
        .value("INPUT_RANGE_PM_40_MV", INPUT_RANGE_PM_40_MV)
        .value("INPUT_RANGE_PM_50_MV", INPUT_RANGE_PM_50_MV)
        .value("INPUT_RANGE_PM_80_MV", INPUT_RANGE_PM_80_MV)
        .value("INPUT_RANGE_PM_100_MV", INPUT_RANGE_PM_100_MV)
        .value("INPUT_RANGE_PM_200_MV", INPUT_RANGE_PM_200_MV)
        .value("INPUT_RANGE_PM_400_MV", INPUT_RANGE_PM_400_MV)
        .value("INPUT_RANGE_PM_500_MV", INPUT_RANGE_PM_500_MV)
        .value("INPUT_RANGE_PM_800_MV", INPUT_RANGE_PM_800_MV)
        .value("INPUT_RANGE_PM_1_V", INPUT_RANGE_PM_1_V)
        .value("INPUT_RANGE_PM_2_V", INPUT_RANGE_PM_2_V)
        .value("INPUT_RANGE_PM_4_V", INPUT_RANGE_PM_4_V)
        .value("INPUT_RANGE_PM_5_V", INPUT_RANGE_PM_5_V)
        .value("INPUT_RANGE_PM_8_V", INPUT_RANGE_PM_8_V)
        .value("INPUT_RANGE_PM_10_V", INPUT_RANGE_PM_10_V)
        .value("INPUT_RANGE_PM_20_V", INPUT_RANGE_PM_20_V)
        .value("INPUT_RANGE_PM_40_V", INPUT_RANGE_PM_40_V)
        .value("INPUT_RANGE_PM_16_V", INPUT_RANGE_PM_16_V)
        .value("INPUT_RANGE_HIFI", INPUT_RANGE_HIFI)
        .value("INPUT_RANGE_PM_1_V_25", INPUT_RANGE_PM_1_V_25)
        .value("INPUT_RANGE_PM_2_V_5", INPUT_RANGE_PM_2_V_5)
        .value("INPUT_RANGE_PM_125_MV", INPUT_RANGE_PM_125_MV)
        .value("INPUT_RANGE_PM_250_MV", INPUT_RANGE_PM_250_MV)
        .value("INPUT_RANGE_0_TO_40_MV", INPUT_RANGE_0_TO_40_MV)
        .value("INPUT_RANGE_0_TO_80_MV", INPUT_RANGE_0_TO_80_MV)
        .value("INPUT_RANGE_0_TO_100_MV", INPUT_RANGE_0_TO_100_MV)
        .value("INPUT_RANGE_0_TO_160_MV", INPUT_RANGE_0_TO_160_MV)
        .value("INPUT_RANGE_0_TO_200_MV", INPUT_RANGE_0_TO_200_MV)
        .value("INPUT_RANGE_0_TO_250_MV", INPUT_RANGE_0_TO_250_MV)
        .value("INPUT_RANGE_0_TO_400_MV", INPUT_RANGE_0_TO_400_MV)
        .value("INPUT_RANGE_0_TO_500_MV", INPUT_RANGE_0_TO_500_MV)
        .value("INPUT_RANGE_0_TO_800_MV", INPUT_RANGE_0_TO_800_MV)
        .value("INPUT_RANGE_0_TO_1_V", INPUT_RANGE_0_TO_1_V)
        .value("INPUT_RANGE_0_TO_1600_MV", INPUT_RANGE_0_TO_1600_MV)
        .value("INPUT_RANGE_0_TO_2_V", INPUT_RANGE_0_TO_2_V)
        .value("INPUT_RANGE_0_TO_2_V_5", INPUT_RANGE_0_TO_2_V_5)
        .value("INPUT_RANGE_0_TO_4_V", INPUT_RANGE_0_TO_4_V)
        .value("INPUT_RANGE_0_TO_5_V", INPUT_RANGE_0_TO_5_V)
        .value("INPUT_RANGE_0_TO_8_V", INPUT_RANGE_0_TO_8_V)
        .value("INPUT_RANGE_0_TO_10_V", INPUT_RANGE_0_TO_10_V)
        .value("INPUT_RANGE_0_TO_16_V", INPUT_RANGE_0_TO_16_V)
        .value("INPUT_RANGE_0_TO_20_V", INPUT_RANGE_0_TO_20_V)
        .value("INPUT_RANGE_0_TO_80_V", INPUT_RANGE_0_TO_80_V)
        .value("INPUT_RANGE_0_TO_32_V", INPUT_RANGE_0_TO_32_V)
        .value("INPUT_RANGE_0_TO_MINUS_40_MV", INPUT_RANGE_0_TO_MINUS_40_MV)
        .value("INPUT_RANGE_0_TO_MINUS_80_MV", INPUT_RANGE_0_TO_MINUS_80_MV)
        .value("INPUT_RANGE_0_TO_MINUS_100_MV", INPUT_RANGE_0_TO_MINUS_100_MV)
        .value("INPUT_RANGE_0_TO_MINUS_160_MV", INPUT_RANGE_0_TO_MINUS_160_MV)
        .value("INPUT_RANGE_0_TO_MINUS_200_MV", INPUT_RANGE_0_TO_MINUS_200_MV)
        .value("INPUT_RANGE_0_TO_MINUS_250_MV", INPUT_RANGE_0_TO_MINUS_250_MV)
        .value("INPUT_RANGE_0_TO_MINUS_400_MV", INPUT_RANGE_0_TO_MINUS_400_MV)
        .value("INPUT_RANGE_0_TO_MINUS_500_MV", INPUT_RANGE_0_TO_MINUS_500_MV)
        .value("INPUT_RANGE_0_TO_MINUS_800_MV", INPUT_RANGE_0_TO_MINUS_800_MV)
        .value("INPUT_RANGE_0_TO_MINUS_1_V", INPUT_RANGE_0_TO_MINUS_1_V)
        .value("INPUT_RANGE_0_TO_MINUS_1600_MV", INPUT_RANGE_0_TO_MINUS_1600_MV)
        .value("INPUT_RANGE_0_TO_MINUS_2_V", INPUT_RANGE_0_TO_MINUS_2_V)
        .value("INPUT_RANGE_0_TO_MINUS_2_V_5", INPUT_RANGE_0_TO_MINUS_2_V_5)
        .value("INPUT_RANGE_0_TO_MINUS_4_V", INPUT_RANGE_0_TO_MINUS_4_V)
        .value("INPUT_RANGE_0_TO_MINUS_5_V", INPUT_RANGE_0_TO_MINUS_5_V)
        .value("INPUT_RANGE_0_TO_MINUS_8_V", INPUT_RANGE_0_TO_MINUS_8_V)
        .value("INPUT_RANGE_0_TO_MINUS_10_V", INPUT_RANGE_0_TO_MINUS_10_V)
        .value("INPUT_RANGE_0_TO_MINUS_16_V", INPUT_RANGE_0_TO_MINUS_16_V)
        .value("INPUT_RANGE_0_TO_MINUS_20_V", INPUT_RANGE_0_TO_MINUS_20_V)
        .value("INPUT_RANGE_0_TO_MINUS_80_V", INPUT_RANGE_0_TO_MINUS_80_V)
        .value("INPUT_RANGE_0_TO_MINUS_32_V", INPUT_RANGE_0_TO_MINUS_32_V)
        .export_values();

    py::enum_<ALAZAR_COUPLINGS>(ats_module, "ALAZAR_COUPLINGS")
        .value("AC_COUPLING", AC_COUPLING)
        .value("DC_COUPLING", DC_COUPLING)
        .export_values();

    py::enum_<ALAZAR_IMPEDANCES>(ats_module, "ALAZAR_IMPEDANCES")
        .value("IMPEDANCE_1M_OHM", IMPEDANCE_1M_OHM)
        .value("IMPEDANCE_50_OHM", IMPEDANCE_50_OHM)
        .value("IMPEDANCE_75_OHM", IMPEDANCE_75_OHM)
        .value("IMPEDANCE_300_OHM", IMPEDANCE_300_OHM)
        .export_values();

    py::enum_<ALAZAR_CLOCK_SOURCES>(ats_module, "ALAZAR_CLOCK_SOURCES")
        .value("INTERNAL_CLOCK", INTERNAL_CLOCK)
        .value("EXTERNAL_CLOCK", EXTERNAL_CLOCK)
        .value("FAST_EXTERNAL_CLOCK", FAST_EXTERNAL_CLOCK)
        .value("MEDIUM_EXTERNAL_CLOCK", MEDIUM_EXTERNAL_CLOCK)
        .value("SLOW_EXTERNAL_CLOCK", SLOW_EXTERNAL_CLOCK)
        .value("EXTERNAL_CLOCK_AC", EXTERNAL_CLOCK_AC)
        .value("EXTERNAL_CLOCK_DC", EXTERNAL_CLOCK_DC)
        .value("EXTERNAL_CLOCK_10MHZ_REF", EXTERNAL_CLOCK_10MHZ_REF)
        .value("INTERNAL_CLOCK_10MHZ_REF", INTERNAL_CLOCK_10MHZ_REF)
        .value("EXTERNAL_CLOCK_10MHZ_PXI", EXTERNAL_CLOCK_10MHZ_PXI)
        .export_values();

    py::enum_<ALAZAR_SAMPLE_RATES>(ats_module, "ALAZAR_SAMPLE_RATES")
        .value("SAMPLE_RATE_1KSPS", SAMPLE_RATE_1KSPS)
        .value("SAMPLE_RATE_2KSPS", SAMPLE_RATE_2KSPS)
        .value("SAMPLE_RATE_5KSPS", SAMPLE_RATE_5KSPS)
        .value("SAMPLE_RATE_10KSPS", SAMPLE_RATE_10KSPS)
        .value("SAMPLE_RATE_20KSPS", SAMPLE_RATE_20KSPS)
        .value("SAMPLE_RATE_50KSPS", SAMPLE_RATE_50KSPS)
        .value("SAMPLE_RATE_100KSPS", SAMPLE_RATE_100KSPS)
        .value("SAMPLE_RATE_200KSPS", SAMPLE_RATE_200KSPS)
        .value("SAMPLE_RATE_500KSPS", SAMPLE_RATE_500KSPS)
        .value("SAMPLE_RATE_1MSPS", SAMPLE_RATE_1MSPS)
        .value("SAMPLE_RATE_2MSPS", SAMPLE_RATE_2MSPS)
        .value("SAMPLE_RATE_5MSPS", SAMPLE_RATE_5MSPS)
        .value("SAMPLE_RATE_10MSPS", SAMPLE_RATE_10MSPS)
        .value("SAMPLE_RATE_20MSPS", SAMPLE_RATE_20MSPS)
        .value("SAMPLE_RATE_25MSPS", SAMPLE_RATE_25MSPS)
        .value("SAMPLE_RATE_50MSPS", SAMPLE_RATE_50MSPS)
        .value("SAMPLE_RATE_100MSPS", SAMPLE_RATE_100MSPS)
        .value("SAMPLE_RATE_125MSPS", SAMPLE_RATE_125MSPS)
        .value("SAMPLE_RATE_160MSPS", SAMPLE_RATE_160MSPS)
        .value("SAMPLE_RATE_180MSPS", SAMPLE_RATE_180MSPS)
        .value("SAMPLE_RATE_200MSPS", SAMPLE_RATE_200MSPS)
        .value("SAMPLE_RATE_250MSPS", SAMPLE_RATE_250MSPS)
        .value("SAMPLE_RATE_400MSPS", SAMPLE_RATE_400MSPS)
        .value("SAMPLE_RATE_500MSPS", SAMPLE_RATE_500MSPS)
        .value("SAMPLE_RATE_800MSPS", SAMPLE_RATE_800MSPS)
        .value("SAMPLE_RATE_1000MSPS", SAMPLE_RATE_1000MSPS)
        .value("SAMPLE_RATE_1GSPS", SAMPLE_RATE_1GSPS)
        .value("SAMPLE_RATE_1200MSPS", SAMPLE_RATE_1200MSPS)
        .value("SAMPLE_RATE_1500MSPS", SAMPLE_RATE_1500MSPS)
        .value("SAMPLE_RATE_1600MSPS", SAMPLE_RATE_1600MSPS)
        .value("SAMPLE_RATE_1800MSPS", SAMPLE_RATE_1800MSPS)
        .value("SAMPLE_RATE_2000MSPS", SAMPLE_RATE_2000MSPS)
        .value("SAMPLE_RATE_2GSPS", SAMPLE_RATE_2GSPS)
        .value("SAMPLE_RATE_2400MSPS", SAMPLE_RATE_2400MSPS)
        .value("SAMPLE_RATE_3000MSPS", SAMPLE_RATE_3000MSPS)
        .value("SAMPLE_RATE_3GSPS", SAMPLE_RATE_3GSPS)
        .value("SAMPLE_RATE_3600MSPS", SAMPLE_RATE_3600MSPS)
        .value("SAMPLE_RATE_4000MSPS", SAMPLE_RATE_4000MSPS)
        .value("SAMPLE_RATE_4GSPS", SAMPLE_RATE_4GSPS)
        .value("SAMPLE_RATE_300MSPS", SAMPLE_RATE_300MSPS)
        .value("SAMPLE_RATE_350MSPS", SAMPLE_RATE_350MSPS)
        .value("SAMPLE_RATE_370MSPS", SAMPLE_RATE_370MSPS)
        .value("SAMPLE_RATE_USER_DEF", SAMPLE_RATE_USER_DEF)
        .export_values();

    py::enum_<ALAZAR_CLOCK_EDGES>(ats_module, "ALAZAR_CLOCK_EDGES")
        .value("CLOCK_EDGE_RISING", CLOCK_EDGE_RISING)
        .value("CLOCK_EDGE_FALLING", CLOCK_EDGE_FALLING)
        .export_values();

    py::enum_<ALAZAR_EXTERNAL_TRIGGER_RANGES>(ats_module, "ALAZAR_EXTERNAL_TRIGGER_RANGES")
        .value("ETR_1V", ETR_1V)
        .value("ETR_2V5", ETR_2V5)
        .value("ETR_5V", ETR_5V)
        .value("ETR_TTL", ETR_TTL)
        .export_values();

    py::enum_<ALAZAR_TRIGGER_OPERATIONS>(ats_module, "ALAZAR_TRIGGER_OPERATIONS")
        .value("TRIG_ENGINE_OP_J", TRIG_ENGINE_OP_J)
        .value("TRIG_ENGINE_OP_K", TRIG_ENGINE_OP_K)
        .value("TRIG_ENGINE_OP_J_OR_K", TRIG_ENGINE_OP_J_OR_K)
        .value("TRIG_ENGINE_OP_J_AND_K", TRIG_ENGINE_OP_J_AND_K)
        .value("TRIG_ENGINE_OP_J_XOR_K", TRIG_ENGINE_OP_J_XOR_K)
        .value("TRIG_ENGINE_OP_J_AND_NOT_K", TRIG_ENGINE_OP_J_AND_NOT_K)
        .value("TRIG_ENGINE_OP_NOT_J_AND_K", TRIG_ENGINE_OP_NOT_J_AND_K)
        .export_values();

    py::enum_<ALAZAR_TRIGGER_ENGINES>(ats_module, "ALAZAR_TRIGGER_ENGINES")
        .value("TRIG_ENGINE_J", TRIG_ENGINE_J)
        .value("TRIG_ENGINE_K", TRIG_ENGINE_K)
        .export_values();

    py::enum_<ALAZAR_TRIGGER_SOURCES>(ats_module, "ALAZAR_TRIGGER_SOURCES")
        .value("TRIG_CHAN_A", TRIG_CHAN_A)
        .value("TRIG_CHAN_B", TRIG_CHAN_B)
        .value("TRIG_EXTERNAL", TRIG_EXTERNAL)
        .value("TRIG_DISABLE", TRIG_DISABLE)
        .value("TRIG_CHAN_C", TRIG_CHAN_C)
        .value("TRIG_CHAN_D", TRIG_CHAN_D)
        .value("TRIG_CHAN_E", TRIG_CHAN_E)
        .value("TRIG_CHAN_F", TRIG_CHAN_F)
        .value("TRIG_CHAN_G", TRIG_CHAN_G)
        .value("TRIG_CHAN_H", TRIG_CHAN_H)
        .value("TRIG_CHAN_I", TRIG_CHAN_I)
        .value("TRIG_CHAN_J", TRIG_CHAN_J)
        .value("TRIG_CHAN_K", TRIG_CHAN_K)
        .value("TRIG_CHAN_L", TRIG_CHAN_L)
        .value("TRIG_CHAN_M", TRIG_CHAN_M)
        .value("TRIG_CHAN_N", TRIG_CHAN_N)
        .value("TRIG_CHAN_O", TRIG_CHAN_O)
        .value("TRIG_CHAN_P", TRIG_CHAN_P)
        .value("TRIG_PXI_STAR", TRIG_PXI_STAR)
        .export_values();

    py::enum_<ALAZAR_TRIGGER_SLOPES>(ats_module, "ALAZAR_TRIGGER_SLOPES")
        .value("TRIGGER_SLOPE_POSITIVE", TRIGGER_SLOPE_POSITIVE)
        .value("TRIGGER_SLOPE_NEGATIVE", TRIGGER_SLOPE_NEGATIVE)
        .export_values();

    py::enum_<ATS_GPU_SETUP_FLAG>(ats_module, "ATS_GPU_SETUP_FLAG", py::arithmetic())
        .value("ATS_GPU_SETUP_FLAG_CPU_BUFFER", ATS_GPU_SETUP_FLAG_CPU_BUFFER)
        .value("ATS_GPU_SETUP_FLAG_MAPPED_MEMORY", ATS_GPU_SETUP_FLAG_MAPPED_MEMORY)
        .value("ATS_GPU_SETUP_FLAG_DEINTERLEAVE", ATS_GPU_SETUP_FLAG_DEINTERLEAVE)
        .value("ATS_GPU_SETUP_FLAG_UNPACK", ATS_GPU_SETUP_FLAG_UNPACK)
        .export_values();
}