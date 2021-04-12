/**
 * @file
 *
 * @author Alazar Technologies Inc
 *
 * @copyright Copyright (c) 2016 Alazar Technologies Inc. All Rights
 * Reserved.  Unpublished - rights reserved under the Copyright laws
 * of the United States And Canada.
 * This product contains confidential information and trade secrets
 * of Alazar Technologies Inc. Use, disclosure, or reproduction is
 * prohibited without the prior express written permission of Alazar
 * Technologies Inc
 */

#ifndef _ALAZARCMD_H
#define _ALAZARCMD_H

/**
 * @brief Memory size per channel.
 */
enum MemorySizesPerChannel {
    MEM8K = 0,
    MEM64K,
    MEM128K,
    MEM256K,
    MEM512K,
    MEM1M,
    MEM2M,
    MEM4M,
    MEM8M,
    MEM16M,
    MEM32M,
    MEM64M,
    MEM128M,
    MEM256M,
    MEM512M,
    MEM1G,
    MEM2G,
    MEM4G,
    MEM8G,
    MEM16G
};

/**
 * @brief Capabilities that can be queried through AlazarQueryCapability()
 */
enum ALAZAR_CAPABILITIES {
    GET_SERIAL_NUMBER = 0x10000024UL,         ///< Board's serial number
    GET_FIRST_CAL_DATE = 0x10000025UL,        ///< First calibration date
    GET_LATEST_CAL_DATE = 0x10000026UL,       ///< Latest calibration date
    GET_LATEST_TEST_DATE = 0x10000027UL,      ///< Latest test date
    GET_LATEST_CAL_DATE_MONTH = 0x1000002DUL, ///< Month of latest calibration
    GET_LATEST_CAL_DATE_DAY = 0x1000002EUL,   ///< Day of latest calibration
    GET_LATEST_CAL_DATE_YEAR = 0x1000002FUL,  ///< Year of latest calibration
    GET_BOARD_OPTIONS_LOW = 0x10000037UL,     ///< Low part of the board options
    GET_BOARD_OPTIONS_HIGH = 0x10000038UL, ///< High part of the board options
    MEMORY_SIZE = 0x1000002AUL,            ///< The memory size in samples
    ASOPC_TYPE = 0x1000002CUL,             ///< The FPGA signature
    BOARD_TYPE =
        0x1000002BUL, ///< The board type as a member of ALAZAR_BOARDTYPES
    GET_PCIE_LINK_SPEED = 0x10000030UL, ///< PCIe link speed in Gb/s
    GET_PCIE_LINK_WIDTH = 0x10000031UL, ///< PCIe link width in lanes
    GET_MAX_PRETRIGGER_SAMPLES =
        0x10000046UL, ///< Maximum number of pre-trigger samples.
    GET_CPF_DEVICE =
        0x10000071UL, ///< User-programmable FPGA device. 1 = SL50, 2 = SE260

    /// Queries if the board supports NPT record footers. Returns 1 if the
    /// feature is supported and 0 otherwise
    HAS_RECORD_FOOTERS_SUPPORT = 0x10000073UL,

    /// Queries if the board supports the AutoDMA Traditional acquisition mode.
    /// Returns 1 if the feature is supported and 0 otherwise.
    CAP_SUPPORTS_TRADITIONAL_AUTODMA = 0x10000074UL,

    /// Queries if the board supports the AutoDMA NPT accquisition mode. Returns
    /// 1 if the feature is supported and 0 otherwise.
    CAP_SUPPORTS_NPT_AUTODMA = 0x10000075UL,

    /// Queries the maximum number of pre-trigger samples that can be requested
    /// in the AutoDMA NPT acquisition mode. This amount is shared between all
    /// the channels of the board.
    CAP_MAX_NPT_PRETRIGGER_SAMPLES = 0x10000076UL,

    /// Tests if this board of the virtual-FIFO type.
    CAP_IS_VFIFO_BOARD = 0x10000077UL,

    /// Tests if this board features native support for single-port
    /// acquisitions. Returns 1 if native support is present, and 0 otherwise.
    CAP_SUPPORTS_NATIVE_SINGLE_PORT = 0x10000078UL,

    /// Tests if this board supports 8-bit data packing. Returns 1 if this board
    /// has a native resolution of more than 8 bits and supports 8-bit packing.
    CAP_SUPPORT_8_BIT_PACKING = 0x10000079UL,

    /// Tests if this board supports 12-bit data packing. Returns 1 if support
    /// is present, and 0 otherwise.
    CAP_SUPPORT_12_BIT_PACKING = 0x10000080UL,

    /// @cond INTERNAL_DECLARATIONS
    /// Queries if the board supports dual-buffer mode. Returns 1 if the feature
    /// is supported, and 0 otherwise.
    HAS_DUAL_BUFFER_SUPPORT = 0x10000072UL,
    /// @endcond

    /// Tests if this board supports record headers. Returns 1 if support
    /// is present, and 0 otherwise.
    HAS_RECORD_HEADERS_SUPPORT = 0x10000081UL,

    /// Tests if this board supports samples interleaved in traditional mode.
    /// Returns 1 if support is present, and 0 otherwise.
    CAP_SUPPORT_TRADITIONAL_SAMPLES_INTERLEAVED = 0x10000082UL,
};

/**
 * @brief ECC Modes
 */
enum ALAZAR_ECC_MODES {
    ECC_DISABLE = 0, ///< Disable
    ECC_ENABLE = 1,  ///< Enable
};

/**
 * @brief Auxiliary input levels
 */
enum ALAZAR_AUX_INPUT_LEVELS {
    AUX_INPUT_LOW = 0,  ///< Low level
    AUX_INPUT_HIGH = 1, ///< High level
};

/**
 * @brief Data pack modes
 */
enum ALAZAR_PACK_MODES {
    PACK_DEFAULT = 0,            ///< Default pack mode of the board
    PACK_8_BITS_PER_SAMPLE = 1,  ///< 8 bits per sample
    PACK_12_BITS_PER_SAMPLE = 2, ///< 12 bits per sample
};

/**
 * @brief API trace states
 */
enum ALAZAR_API_TRACE_STATES {
    API_ENABLE_TRACE = 1,  ///< Trace enabled
    API_DISABLE_TRACE = 0, ///< Trace disabled
};

/**
 * @brief Parameters suitable to be used with AlazarSetParameter() and/or
 * AlazarGetParameter()
 */
enum ALAZAR_PARAMETERS {
    DATA_WIDTH = 0x10000009UL, ///< The number of bits per sample
    SETGET_ASYNC_BUFFSIZE_BYTES =
        0x10000039UL, ///< The size of API-allocated DMA buffers in bytes
    SETGET_ASYNC_BUFFCOUNT =
        0x10000040UL, ///< The number of API-allocated DMA buffers
    GET_ASYNC_BUFFERS_PENDING =
        0x10000050UL, ///< DMA buffers currently posted to the board
    GET_ASYNC_BUFFERS_PENDING_FULL = 0x10000051UL, ///< DMA buffers waiting to
                                                   /// be processed by the
                                                   /// application
    GET_ASYNC_BUFFERS_PENDING_EMPTY =
        0x10000052UL, ///< DMA buffers waiting to be filled by the board
    SET_DATA_FORMAT =
        0x10000041UL, ///< 0 if the data format is unsigned, and 1 otherwise
    GET_DATA_FORMAT =
        0x10000042UL, ///< 0 if the data format is unsigned, and 1 otherwise
    GET_SAMPLES_PER_TIMESTAMP_CLOCK =
        0x10000044UL, ///< Number of samples per timestamp clock
    GET_RECORDS_CAPTURED = 0x10000045UL, ///< Records captured since the start
                                         /// of the acquisition (single-port) or
                                         /// buffer (dual-port)
    ECC_MODE = 0x10000048UL, ///< ECC mode. Member of \ref ALAZAR_ECC_MODES
    GET_AUX_INPUT_LEVEL = 0x10000049UL, ///< Read the TTL level of the AUX
                                        /// connector. Member of \ref
                                        /// ALAZAR_AUX_INPUT_LEVELS
    GET_CHANNELS_PER_BOARD =
        0x10000070UL, ///< Number of analog channels supported by this digitizer
    GET_FPGA_TEMPERATURE = 0x10000080UL, ///< Current FPGA temperature in
                                         /// degrees Celcius. Only supported by
                                         /// PCIe digitizers.
    PACK_MODE = 0x10000072UL, ///< Get/Set the pack mode as a member of \ref
                              /// ALAZAR_PACK_MODES
    SET_SINGLE_CHANNEL_MODE = 0x10000043UL, ///< Reserve all the on-board memory
                                            /// to the channel passed as
                                            /// argument. Single-port only.
    API_FLAGS = 0x10000090UL, ///< State of the API logging as a member of \ref
                              /// ALAZAR_API_TRACE_STATES

    /// @cond INTERNAL_DECLARATIONS
    /// Get the calculated adjustment for lane 0 in LSBs.
    GET_ADCBC_LANE_0_ADJUSTMENT = 0x10000093UL,

    /// Get the calculated adjustment for lane 1 in LSBs.
    GET_ADCBC_LANE_1_ADJUSTMENT = 0x10000094UL,

    /// Get the calculated adjustment for lane 2 in LSBs.
    GET_ADCBC_LANE_2_ADJUSTMENT = 0x10000095UL,

    /// Get the calculated adjustment for lane 3 in LSBs.
    GET_ADCBC_LANE_3_ADJUSTMENT = 0x10000096UL,
    /// @endcond
};

/**
 * @brief Analog to digital converter modes.
 */
enum ALAZAR_ADC_MODES {
    ADC_MODE_DEFAULT = 0, ///< Default ADC mode
    ADC_MODE_DES = 1,     ///< Dual-edge sampling mode
};

/**
 * @brief Parameters suitable to be used with AlazarSetParameterUL() and/or
 * AlazarGetParameterUL()
 */
enum ALAZAR_PARAMETERS_UL {
    SET_ADC_MODE =
        0x10000047UL, ///< Set the ADC mode as a member of \ref ALAZAR_ADC_MODES

    /// @cond INTERNAL_DECLARATIONS
    ///
    /// Set or get the state of the ADC background compensation limit. 0 means
    /// the limit is inactive. Anything else means the limit is active.
    SETGET_ADCBC_LIMIT = 0x10000091UL,

    /// Set or get the number of points to use to compute the ADC background
    /// compensation. This must be a power of two between 512 and 65536
    /// included.
    SETGET_ADCBC_POINTS = 0x10000092UL,
    /// @endcond

    /// Configures the number of DMA buffers acquired after each trigger enable
    /// event. The default value is 1.
    ///
    /// @remark To set the number of buffers per trigger enable, this must be
    ///         called after AlazarBeforeAsyncRead() but before
    ///         AlazarStartCapture(), which means that AlazarBeforeAsyncRead()
    ///         must be called with #ADMA_EXTERNAL_STARTCAPTURE
    ///
    /// @remark This parameter is reset in between acquisitions.
    SET_BUFFERS_PER_TRIGGER_ENABLE = 0x10000097UL,

    /// Queries the status of the power monitor on the board. The value returned
    /// is zero if there is no problem. If it is not zero, please send the value
    /// returned to AlazarTech's technical support.
    GET_POWER_MONITOR_STATUS = 0x10000098UL,

    /// Configure external trigger range. Parameter is as a member of \ref
    /// ALAZAR_EXTERNAL_TRIGGER_RANGES
    SET_EXT_TRIGGER_RANGE = 0x1000001CUL,
};

/**
 * @cond INTERNAL_DECLARATIONS
 *
 * @{
 * Deprecated
 */
#define PRETRIGGER_AMOUNT 0x10000002UL
#define RECORD_LENGTH 0x10000003UL
#define AUTO_CALIBRATE 0x1000000AUL
#define ACF_SAMPLES_PER_RECORD 0x10000060UL
#define ADC_MODE_DES_WIDEBAND 2
#define ADC_MODE_RESET_ENABLE 0x8001
#define ADC_MODE_RESET_DISABLE 0x8002
#define GET_PCI_CONFIG_HEADER 0x10000033UL
#define SAMPLE_SIZE 0x10000009UL
#define NUMBER_OF_RECORDS 0x10000001UL
#define SAMPLE_RATE 0x10000007UL
#define TRIGGER_ENGINE 0x10000004UL
#define TRIGGER_DELAY 0x10000005UL
#define TRIGGER_TIMEOUT 0x10000006UL
#define CONFIGURATION_MODE 0x10000008UL // Independent, Master/Slave, Last Slave
#define CLOCK_SOURCE 0x1000000CUL
#define CLOCK_SLOPE 0x1000000DUL
#define IMPEDANCE 0x1000000EUL
#define INPUT_RANGE 0x1000000FUL
#define COUPLING 0x10000010UL
#define MAX_TIMEOUTS_ALLOWED 0x10000011UL
#define ATS_OPERATING_MODE 0x10000012UL
#define OPERATING_MODE 0x10000012UL // Single, Dual, Quad etc...
#define CLOCK_DECIMATION_EXTERNAL 0x10000013UL
#define LED_CONTROL 0x10000014UL
#define ATTENUATOR_RELAY 0x10000018UL
#define EXT_TRIGGER_COUPLING 0x1000001AUL
#define EXT_TRIGGER_ATTENUATOR_RELAY                                           \
    0x1000001CUL ///< Deprecated, use SET_EXT_TRIGGER_RANGE instead
#define TRIGGER_ENGINE_SOURCE 0x1000001EUL

#define TRIGGER_XXXXX 0x1000000BUL
#define TRIGGER_ENGINE_SLOPE 0x10000020UL
#define SEND_DAC_VALUE 0x10000021UL
#define SLEEP_DEVICE 0x10000022UL
#define GET_DAC_VALUE 0x10000023UL
#define SEND_RELAY_VALUE 0x10000028UL
#define DATA_FORMAT_UNSIGNED 0
#define DATA_FORMAT_SIGNED 1
#define EXT_TRIGGER_IMPEDANCE 0x10000065UL
#define EXT_TRIG_50_OHMS 0
#define EXT_TRIG_300_OHMS 1
#define MEMORY_SIZE_MSAMPLES 0x1000004AUL
/**
 * @}
 * @endcond
 */

/**
 * @brief AlazarTech board options. Lower 32-bits
 */
enum ALAZAR_BOARD_OPTIONS_LOW {
    OPTION_STREAMING_DMA = (1UL << 0),
    OPTION_EXTERNAL_CLOCK = (1UL << 1),
    OPTION_DUAL_PORT_MEMORY = (1UL << 2),
    OPTION_180MHZ_OSCILLATOR = (1UL << 3),
    OPTION_LVTTL_EXT_CLOCK = (1UL << 4),
    OPTION_SW_SPI = (1UL << 5),
    OPTION_ALT_INPUT_RANGES = (1UL << 6),
    OPTION_VARIABLE_RATE_10MHZ_PLL = (1UL << 7),
    OPTION_MULTI_FREQ_VCO = (1UL << 7),
    OPTION_2GHZ_ADC = (1UL << 8),
    OPTION_DUAL_EDGE_SAMPLING = (1UL << 9),
    OPTION_DCLK_PHASE = (1UL << 10),
    OPTION_WIDEBAND = (1UL << 11),
};

/**
 * @brief AlazarTech board options. Higher 32-bits
 */
enum ALAZAR_BOARD_OPTIONS_HIGH {
    OPTION_OEM_FPGA = (1ULL << 15),
};

/**
 * @cond INTERNAL_DECLARATIONS
 */
// sets and gets
// The transfer offset is defined as the place to start
// the transfer relative to trigger. The value is signed.
// -------TO>>>T>>>>>>>>>TE------------
#define TRANSFER_OFFET 0x10000030UL
#define TRANSFER_LENGTH 0x10000031UL // TO -> TE

// Transfer related constants
#define TRANSFER_RECORD_OFFSET 0x10000032UL
#define TRANSFER_NUM_OF_RECORDS 0x10000033UL
#define TRANSFER_MAPPING_RATIO 0x10000034UL

// only gets
#define TRIGGER_ADDRESS_AND_TIMESTAMP 0x10000035UL

// MASTER/SLAVE CONTROL sets/gets
#define MASTER_SLAVE_INDEPENDENT 0x10000036UL

// boolean gets
#define TRIGGERED 0x10000040UL
#define BUSY 0x10000041UL
#define WHO_TRIGGERED 0x10000042UL
#define ACF_RECORDS_TO_AVERAGE 0x10000061UL
#define ACF_MODE 0x10000062UL
#define ACF_NUMBER_OF_LAGS 0x10000063

/**
 * @endcond
 */

/**
 * @brief Sample rate identifiers
 */
enum ALAZAR_SAMPLE_RATES {
    SAMPLE_RATE_1KSPS = 0X00000001UL,
    SAMPLE_RATE_2KSPS = 0X00000002UL,
    SAMPLE_RATE_5KSPS = 0X00000004UL,
    SAMPLE_RATE_10KSPS = 0X00000008UL,
    SAMPLE_RATE_20KSPS = 0X0000000AUL,
    SAMPLE_RATE_50KSPS = 0X0000000CUL,
    SAMPLE_RATE_100KSPS = 0X0000000EUL,
    SAMPLE_RATE_200KSPS = 0X00000010UL,
    SAMPLE_RATE_500KSPS = 0X00000012UL,
    SAMPLE_RATE_1MSPS = 0X00000014UL,
    SAMPLE_RATE_2MSPS = 0X00000018UL,
    SAMPLE_RATE_5MSPS = 0X0000001AUL,
    SAMPLE_RATE_10MSPS = 0X0000001CUL,
    SAMPLE_RATE_20MSPS = 0X0000001EUL,
    SAMPLE_RATE_25MSPS = 0X00000021UL,
    SAMPLE_RATE_50MSPS = 0X00000022UL,
    SAMPLE_RATE_100MSPS = 0X00000024UL,
    SAMPLE_RATE_125MSPS = 0x00000025UL,
    SAMPLE_RATE_160MSPS = 0x00000026UL,
    SAMPLE_RATE_180MSPS = 0x00000027UL,
    SAMPLE_RATE_200MSPS = 0X00000028UL,
    SAMPLE_RATE_250MSPS = 0X0000002BUL,
    SAMPLE_RATE_400MSPS = 0X0000002DUL,
    SAMPLE_RATE_500MSPS = 0X00000030UL,
    SAMPLE_RATE_800MSPS = 0X00000032UL,
    SAMPLE_RATE_1000MSPS = 0x00000035UL,
    SAMPLE_RATE_1GSPS = SAMPLE_RATE_1000MSPS,
    SAMPLE_RATE_1200MSPS = 0x00000037UL,
    SAMPLE_RATE_1500MSPS = 0x0000003AUL,
    SAMPLE_RATE_1600MSPS = 0x0000003BUL,
    SAMPLE_RATE_1800MSPS = 0x0000003DUL,
    SAMPLE_RATE_2000MSPS = 0x0000003FUL,
    SAMPLE_RATE_2GSPS = SAMPLE_RATE_2000MSPS,
    SAMPLE_RATE_2400MSPS = 0x0000006AUL,
    SAMPLE_RATE_3000MSPS = 0x00000075UL,
    SAMPLE_RATE_3GSPS = SAMPLE_RATE_3000MSPS,
    SAMPLE_RATE_3600MSPS = 0x0000007BUL,
    SAMPLE_RATE_4000MSPS = 0x00000080UL,
    SAMPLE_RATE_4GSPS = SAMPLE_RATE_4000MSPS,
    SAMPLE_RATE_300MSPS = 0x00000090UL,
    SAMPLE_RATE_350MSPS = 0x00000094UL,
    SAMPLE_RATE_370MSPS = 0x00000096UL,
    SAMPLE_RATE_USER_DEF =
        0x00000040UL ///< User-defined sample rate. Used with external clock
};

/**
 * @brief Impedance indentifiers
 */
enum ALAZAR_IMPEDANCES {
    IMPEDANCE_1M_OHM = 0x00000001UL,
    IMPEDANCE_50_OHM = 0x00000002UL,
    IMPEDANCE_75_OHM = 0x00000004UL,
    IMPEDANCE_300_OHM = 0x00000008UL
};

/**
 * @cond INTERNAL_DECLARATIONS
 */
// user define sample rate - used with External Clock
// ATS665 Specific Setting for using the PLL
//
// The base value can be used to create a PLL frequency
// in a simple manner.
//
// Ex.
//        105 MHz = PLL_10MHZ_REF_100MSPS_BASE + 5000000
//        120 MHz = PLL_10MHZ_REF_100MSPS_BASE + 20000000
#define PLL_10MHZ_REF_100MSPS_BASE 0x05F5E100UL

// ATS665 Specific Decimation constants
#define DECIMATE_BY_8 0x00000008UL
#define DECIMATE_BY_64 0x00000040UL
/**
 * @endcond
 */

/**
 * @brief Clock source identifiers
 */
enum ALAZAR_CLOCK_SOURCES {
    INTERNAL_CLOCK = 0x00000001UL,           ///< Internal clock
    EXTERNAL_CLOCK = 0x00000002UL,           ///< External clock
    FAST_EXTERNAL_CLOCK = 0x00000002UL,      ///< Fast external clock
    MEDIUM_EXTERNAL_CLOCK = 0x00000003UL,    ///< Medium external clock
    SLOW_EXTERNAL_CLOCK = 0x00000004UL,      ///< Slow external clock
    EXTERNAL_CLOCK_AC = 0x00000005UL,        ///< AC external clock
    EXTERNAL_CLOCK_DC = 0x00000006UL,        ///< DC external clock
    EXTERNAL_CLOCK_10MHZ_REF = 0x00000007UL, ///< 10MHz external reference
    INTERNAL_CLOCK_10MHZ_REF = 0x00000008,   ///< Internal 10MHz reference
    EXTERNAL_CLOCK_10MHZ_PXI = 0x0000000A,   ///< External 10MHz PXI

    /// @cond INTERNAL_DECLARATIONS
    INTERNAL_CLOCK_DIV_4 = 0x0000000F,
    INTERNAL_CLOCK_DIV_5 = 0x00000010,
    MASTER_CLOCK = 0x00000011,
    INTERNAL_CLOCK_SET_VCO = 0x00000012,
    EXTERNAL_CLOCK_10MHz_REF = 0x00000007UL,
    INTERNAL_CLOCK_10MHz_REF = 0x00000008,
    EXTERNAL_CLOCK_10MHz_PXI = 0x0000000A
    /// @endcond
};

/**
 * @cond INTERNAL_DECLARATIONS
 */
#define MEDIMUM_EXTERNAL_CLOCK 0x00000003UL // TYPO
/**
 * @endcond
 */

/**
 * @brief Clock edge identifiers
 */
enum ALAZAR_CLOCK_EDGES {
    CLOCK_EDGE_RISING = 0x00000000UL,  ///< Rising clock edge
    CLOCK_EDGE_FALLING = 0x00000001UL, ///< Falling clock edge
};

/**
 * @brief Input range identifiers
 */
enum ALAZAR_INPUT_RANGES {
    INPUT_RANGE_PM_20_MV = 0x00000001UL,           ///< +/- 20mV
    INPUT_RANGE_PM_40_MV = 0x00000002UL,           ///< +/- 40mV
    INPUT_RANGE_PM_50_MV = 0x00000003UL,           ///< +/- 50mV
    INPUT_RANGE_PM_80_MV = 0x00000004UL,           ///< +/- 80mV
    INPUT_RANGE_PM_100_MV = 0x00000005UL,          ///< +/- 100mV
    INPUT_RANGE_PM_200_MV = 0x00000006UL,          ///< +/- 200mV
    INPUT_RANGE_PM_400_MV = 0x00000007UL,          ///< +/- 400mV
    INPUT_RANGE_PM_500_MV = 0x00000008UL,          ///< +/- 500mV
    INPUT_RANGE_PM_800_MV = 0x00000009UL,          ///< +/- 800mV
    INPUT_RANGE_PM_1_V = 0x0000000AUL,             ///< +/- 1V
    INPUT_RANGE_PM_2_V = 0x0000000BUL,             ///< +/- 2V
    INPUT_RANGE_PM_4_V = 0x0000000CUL,             ///< +/- 4V
    INPUT_RANGE_PM_5_V = 0x0000000DUL,             ///< +/- 5V
    INPUT_RANGE_PM_8_V = 0x0000000EUL,             ///< +/- 8V
    INPUT_RANGE_PM_10_V = 0x0000000FUL,            ///< +/- 10V
    INPUT_RANGE_PM_20_V = 0x00000010UL,            ///< +/- 20V
    INPUT_RANGE_PM_40_V = 0x00000011UL,            ///< +/- 40V
    INPUT_RANGE_PM_16_V = 0x00000012UL,            ///< +/- 16V
    INPUT_RANGE_HIFI = 0x00000020UL,               ///< no gain
    INPUT_RANGE_PM_1_V_25 = 0x00000021UL,          ///< +/- 1.25V
    INPUT_RANGE_PM_2_V_5 = 0x00000025UL,           ///< +/- 2.5V
    INPUT_RANGE_PM_125_MV = 0x00000028UL,          ///< +/- 125mV
    INPUT_RANGE_PM_250_MV = 0x00000030UL,          ///< +/- 250mV
    INPUT_RANGE_0_TO_40_MV = 0x00000031UL,         ///< 0 to 40mV
    INPUT_RANGE_0_TO_80_MV = 0x00000032UL,         ///< 0 to 80mV
    INPUT_RANGE_0_TO_100_MV = 0x00000033UL,        ///< 0 to 100mV
    INPUT_RANGE_0_TO_160_MV = 0x00000034UL,        ///< 0 to 160mV
    INPUT_RANGE_0_TO_200_MV = 0x00000035UL,        ///< 0 to 200mV
    INPUT_RANGE_0_TO_250_MV = 0x00000036UL,        ///< 0 to 250mV
    INPUT_RANGE_0_TO_400_MV = 0x00000037UL,        ///< 0 to 400mV
    INPUT_RANGE_0_TO_500_MV = 0x00000038UL,        ///< 0 to 500mV
    INPUT_RANGE_0_TO_800_MV = 0x00000039UL,        ///< 0 to 800mV
    INPUT_RANGE_0_TO_1_V = 0x0000003AUL,           ///< 0 to 1V
    INPUT_RANGE_0_TO_1600_MV = 0x0000003BUL,       ///< 0 to 1.6V
    INPUT_RANGE_0_TO_2_V = 0x0000003CUL,           ///< 0 to 2V
    INPUT_RANGE_0_TO_2_V_5 = 0x0000003DUL,         ///< 0 to 2.5V
    INPUT_RANGE_0_TO_4_V = 0x0000003EUL,           ///< 0 to 4V
    INPUT_RANGE_0_TO_5_V = 0x0000003FUL,           ///< 0 to 5V
    INPUT_RANGE_0_TO_8_V = 0x00000040UL,           ///< 0 to 8V
    INPUT_RANGE_0_TO_10_V = 0x00000041UL,          ///< 0 to 10V
    INPUT_RANGE_0_TO_16_V = 0x00000042UL,          ///< 0 to 16V
    INPUT_RANGE_0_TO_20_V = 0x00000043UL,          ///< 0 to 20V
    INPUT_RANGE_0_TO_80_V = 0x00000044UL,          ///< 0 to 80V
    INPUT_RANGE_0_TO_32_V = 0x00000045UL,          ///< 0 to 32V
    INPUT_RANGE_0_TO_MINUS_40_MV = 0x00000046UL,   ///< 0 to -40mV
    INPUT_RANGE_0_TO_MINUS_80_MV = 0x00000047UL,   ///< 0 to -80mV
    INPUT_RANGE_0_TO_MINUS_100_MV = 0x00000048UL,  ///< 0 to -100mV
    INPUT_RANGE_0_TO_MINUS_160_MV = 0x00000049UL,  ///< 0 to -160mV
    INPUT_RANGE_0_TO_MINUS_200_MV = 0x0000004AUL,  ///< 0 to -200mV
    INPUT_RANGE_0_TO_MINUS_250_MV = 0x0000004BUL,  ///< 0 to -250mV
    INPUT_RANGE_0_TO_MINUS_400_MV = 0x0000004CUL,  ///< 0 to -400mV
    INPUT_RANGE_0_TO_MINUS_500_MV = 0x0000004DUL,  ///< 0 to -500mV
    INPUT_RANGE_0_TO_MINUS_800_MV = 0x0000004EUL,  ///< 0 to -800mV
    INPUT_RANGE_0_TO_MINUS_1_V = 0x0000004FUL,     ///< 0 to -1V
    INPUT_RANGE_0_TO_MINUS_1600_MV = 0x00000050UL, ///< 0 to -1.6V
    INPUT_RANGE_0_TO_MINUS_2_V = 0x00000051UL,     ///< 0 to -2V
    INPUT_RANGE_0_TO_MINUS_2_V_5 = 0x00000052UL,   ///< 0 to -2.5V
    INPUT_RANGE_0_TO_MINUS_4_V = 0x00000053UL,     ///< 0 to -4V
    INPUT_RANGE_0_TO_MINUS_5_V = 0x00000054UL,     ///< 0 to -5V
    INPUT_RANGE_0_TO_MINUS_8_V = 0x00000055UL,     ///< 0 to -8V
    INPUT_RANGE_0_TO_MINUS_10_V = 0x00000056UL,    ///< 0 to -10V
    INPUT_RANGE_0_TO_MINUS_16_V = 0x00000057UL,    ///< 0 to -16V
    INPUT_RANGE_0_TO_MINUS_20_V = 0x00000058UL,    ///< 0 to 20V
    INPUT_RANGE_0_TO_MINUS_80_V = 0x00000059UL,    ///< 0 to 80V
    INPUT_RANGE_0_TO_MINUS_32_V = 0x00000060UL,    ///< 0 to 32V
};

/**
 * @brief Coupling identifiers
 */
enum ALAZAR_COUPLINGS {
    AC_COUPLING = 0x00000001UL, ///< AC coupling
    DC_COUPLING = 0x00000002UL  ///< DC coupling
};

/**
 * @brief Trigger engine identifiers
 */
enum ALAZAR_TRIGGER_ENGINES {
    TRIG_ENGINE_J = 0x00000000UL, ///< The J trigger engine
    TRIG_ENGINE_K = 0x00000001UL  ///< The K trigger engine
};

/**
 * @brief Trigger operation identifiers
 */
enum ALAZAR_TRIGGER_OPERATIONS {
    /**
     * The board triggers when a trigger event is detected by trigger engine J.
     * Events detected by engine K are ignored.
     */
    TRIG_ENGINE_OP_J = 0x00000000UL,

    /**
     * The board triggers when a trigger event is detected by trigger engine K.
     * Events detected by engine J are ignored.
     */
    TRIG_ENGINE_OP_K = 0x00000001UL,

    /**
     * The board triggers when a trigger event is detected by any of the J and K
     * trigger engines.
     */
    TRIG_ENGINE_OP_J_OR_K = 0x00000002UL,

    /**
     * This value is deprecated. It cannot be used.
     */
    TRIG_ENGINE_OP_J_AND_K = 0x00000003UL,

    /**
     * This value is deprecated. It cannot be used.
     */
    TRIG_ENGINE_OP_J_XOR_K = 0x00000004UL,

    /**
     * This value is deprecated. It cannot be used.
     */
    TRIG_ENGINE_OP_J_AND_NOT_K = 0x00000005UL,

    /**
     * This value is deprecated. It cannot be used.
     */
    TRIG_ENGINE_OP_NOT_J_AND_K = 0x00000006UL,
};

/**
 * @brief Trigger sources
 */
enum ALAZAR_TRIGGER_SOURCES {
    TRIG_CHAN_A = 0x00000000UL,   ///< Channel A
    TRIG_CHAN_B = 0x00000001UL,   ///< Channel B
    TRIG_EXTERNAL = 0x00000002UL, ///< External trigger source
    TRIG_DISABLE = 0x00000003UL,  ///< Disabled trigger
    TRIG_CHAN_C = 0x00000004UL,   ///< Channel C
    TRIG_CHAN_D = 0x00000005UL,   ///< Channel D
    TRIG_CHAN_E = 0x00000006UL,   ///< Channel E
    TRIG_CHAN_F = 0x00000007UL,   ///< Channel F
    TRIG_CHAN_G = 0x00000008UL,   ///< Channel G
    TRIG_CHAN_H = 0x00000009UL,   ///< Channel H
    TRIG_CHAN_I = 0x0000000AUL,   ///< Channel I
    TRIG_CHAN_J = 0x0000000BUL,   ///< Channel J
    TRIG_CHAN_K = 0x0000000CUL,   ///< Channel K
    TRIG_CHAN_L = 0x0000000DUL,   ///< Channel L
    TRIG_CHAN_M = 0x0000000EUL,   ///< Channel M
    TRIG_CHAN_N = 0x0000000FUL,   ///< Channel N
    TRIG_CHAN_O = 0x00000010UL,   ///< Channel O
    TRIG_CHAN_P = 0x00000011UL,   ///< Channel P
    TRIG_PXI_STAR = 0x00000100UL  ///< PXI Star channel
};

/**
 * @brief Trigger slope identifiers
 *
 * These identifiers selects whether rising or falling edges of the trigger
 * source signal are detected as trigger events.
 */
enum ALAZAR_TRIGGER_SLOPES {
    /**
     * The trigger engine detects a trigger event when sample values from the
     * trigger source rise above a specified level.
     */
    TRIGGER_SLOPE_POSITIVE = 0x00000001UL,

    /**
     * The trigger engine detects a trigger event when sample values from the
     * trigger source fall below a specified level.
     */
    TRIGGER_SLOPE_NEGATIVE = 0x00000002UL
};

/**
 * @brief Channel identifiers
 */
enum ALAZAR_CHANNELS {
    CHANNEL_ALL = 0x00000000, ///< All channels
    CHANNEL_A = 0x00000001,   ///< Channel A
    CHANNEL_B = 0x00000002,   ///< Channel B
    CHANNEL_C = 0x00000004,   ///< Channel C
    CHANNEL_D = 0x00000008,   ///< Channel D
    CHANNEL_E = 0x00000010,   ///< Channel E
    CHANNEL_F = 0x00000020,   ///< Channel F
    CHANNEL_G = 0x00000040,   ///< Channel G
    CHANNEL_H = 0x00000080,   ///< Channel H
    CHANNEL_I = 0x00000100,   ///< Channel I
    CHANNEL_J = 0x00000200,   ///< Channel J
    CHANNEL_K = 0x00000400,   ///< Channel K
    CHANNEL_L = 0x00000800,   ///< Channel L
    CHANNEL_M = 0x00001000,   ///< Channel M
    CHANNEL_N = 0x00002000,   ///< Channel N
    CHANNEL_O = 0x00004000,   ///< Channel O
    CHANNEL_P = 0x00008000,   ///< Channel P
};

/**
 * @brief Master/Slave configuration
 */
enum ALAZAR_MASTER_SLAVE_CONFIGURATION {
    BOARD_IS_INDEPENDENT = 0x00000000UL, ///< Independent board
    BOARD_IS_MASTER = 0x00000001UL,      ///< master board
    BOARD_IS_SLAVE = 0x00000002UL,       ///< slave board
    BOARD_IS_LAST_SLAVE = 0x00000003UL   ///< last slave of a board system
};

/**
 * @brief LED state identifiers
 */
enum ALAZAR_LED {
    LED_OFF = 0x00000000UL, ///< OFF LED
    LED_ON = 0x00000001UL   ///< ON LED
};

/**
 * @cond INTERNAL_DECLARATIONS
 */
//
// Attenuator Relay
//
#define AR_X1 0x00000000UL
#define AR_DIV40 0x00000001UL

//
// External Trigger Attenuator Relay
//
#define ETR_DIV5 0x00000000UL
#define ETR_X1 0x00000001UL
/**
 * @endcond
 */

/**
 * @brief External trigger range identifiers
 */
enum ALAZAR_EXTERNAL_TRIGGER_RANGES {
    /**
     * @cond INTERNAL_DECLARATIONS
     */
    ETR_5V = 0x00000000UL,  ///< 5V range
    ETR_1V = 0x00000001UL,  ///< 1V range
    ETR_2V5 = 0x00000003UL, ///< 2.5V range
    /**
     * @endcond
     */
    ETR_5V_50OHM = 0x00000000UL,  ///< 5V-50OHM range
    ETR_1V_50OHM = 0x00000001UL,  ///< 1V-50OHM range
    ETR_TTL = 0x00000002UL,       ///< TTL range
    ETR_2V5_50OHM = 0x00000003UL, ///< 2V5-50OHM range
    ETR_5V_300OHM = 0x00000004UL  ///< 5V-300OHM range
};

/**
 * @brief Power states
 */
enum ALAZAR_POWER_STATES {
    POWER_OFF = 0x00000000UL, ///< OFF
    POWER_ON = 0x00000001UL   ///< ON
};

/**
 * @brief Software events control. See AlazarEvents()
 */
enum ALAZAR_SOFTWARE_EVENTS_CONTROL {
    SW_EVENTS_OFF = 0x00000000UL,
    SW_EVENTS_ON = 0x00000001UL,
};

/**
 * @brief Timestamp reset options. See AlazarResetTimeStamp()
 */
enum ALAZAR_TIMESTAMP_RESET_OPTIONS {
    TIMESTAMP_RESET_FIRSTTIME_ONLY = 0x00000000UL,
    TIMESTAMP_RESET_ALWAYS = 0x00000001UL,
};

/**
 * @cond INTERNAL_DECLARATIONS
 */
#define CSO_DUMMY_CLOCK_DISABLE 0
#define CSO_DUMMY_CLOCK_TIMER 1
#define CSO_DUMMY_CLOCK_EXT_TRIGGER 2
#define CSO_DUMMY_CLOCK_TIMER_ON_TIMER_OFF 3
/**
 * @endcond
 */

/**
 * @brief Alazar AUX I/O identifiers
 */
enum ALAZAR_AUX_IO_MODES {
    /// Outputs a signal that is high whenever data is being acquired to
    /// on-board memory, and low otherwise. The \c parameter argument of
    /// AlazarConfigureAuxIO() is ignored in this mode.
    AUX_OUT_TRIGGER = 0U,

    /// Uses the edge of a pulse to the AUX I/O connector as an AutoDMA
    /// *trigger
    /// enable* signal. Please note that this is different from a standard
    /// *trigger* signal. In this mode, the \c parameter argument of
    /// AlazarConfigureAuxIO() can takes an element of \ref
    /// ALAZAR_TRIGGER_SLOPES, which defines on which edge of the input
    /// signal a
    /// trigger enable event is generated.
    AUX_IN_TRIGGER_ENABLE = 1U,

    /// Output the sample clock divided by the value passed to the \c
    /// parameter
    /// argument of AlazarConfigureAuxIO(). Please note that the divided
    /// must be
    /// greater than 2.
    AUX_OUT_PACER = 2U,

    /// Use the AUX I/O connector as a general purpose digital output. The
    /// \c
    /// paramter argument of AlazarConfigureAuxIO() specifies the TTL output
    /// level. 0 means TTL low level, whereas 1 means TTL high level.
    AUX_OUT_SERIAL_DATA = 14U,

    /// Configure the AUX connector as a digital input. Call
    /// AlazarGetParameter() with \ref GET_AUX_INPUT_LEVEL to read the
    /// digital
    /// input level.
    AUX_IN_AUXILIARY = 13U,

    /// @cond INTERNAL_DECLARATIONS
    AUX_OUT_BUSY = 4U,
    AUX_OUT_CLOCK = 6U,
    AUX_OUT_RESERVED = 8U,
    AUX_OUT_CAPTURE_ALMOST_DONE = 10U,
    AUX_OUT_AUXILIARY = 12U,

    AUX_IN_DIGITAL_TRIGGER = 3U,
    AUX_IN_GATE = 5U,
    AUX_IN_CAPTURE_ON_DEMAND = 7U,
    AUX_IN_RESET_TIMESTAMP = 9U,
    AUX_IN_SLOW_EXTERNAL_CLOCK = 11U,
    AUX_IN_SERIAL_DATA = 15U,

    AUX_INPUT_AUXILIARY = AUX_IN_AUXILIARY,
    AUX_INPUT_SERIAL_DATA = AUX_IN_SERIAL_DATA,
    /// @endcond
};

/**
 *  Enables software trigger enable. See AlazarConfigureAuxIO().
 */
#define AUX_OUT_TRIGGER_ENABLE 16U

/**
 * @brief Options for AlazarSetExternalTriggerOperationForScanning()
 */
enum ALAZAR_STOS_OPTIONS { STOS_OPTION_DEFER_START_CAPTURE = 1 };

// Data skipping

/**
 * @brief Data skipping modes. See AlazarConfigureSampleSkipping()
 */
enum ALAZAR_SAMPLE_SKIPPING_MODES {
    SSM_DISABLE = 0, ///< Disable sample skipping
    SSM_ENABLE = 1,  ///< Enable sample skipping
};

/**
 * @brief Coprocessor download options
 */
enum ALAZAR_COPROCESSOR_DOWNLOAD_OPTIONS {
    CPF_OPTION_DMA_DOWNLOAD = 1,
};

/**
 * @cond INTERNAL_DECLARATIONS
 */
// Coprocessor

#define CPF_REG_SIGNATURE 0
#define CPF_REG_REVISION 1
#define CPF_REG_VERSION 2
#define CPF_REG_STATUS 3

#define CPF_DEVICE_UNKNOWN 0
#define CPF_DEVICE_EP3SL50 1
#define CPF_DEVICE_EP3SE260 2
/**
 * @endcond
 */

/**
 * @brief Least significant bit identifiers
 */
enum ALAZAR_LSB {
    LSB_DEFAULT = 0,  ///< Default LSB setting
    LSB_EXT_TRIG = 1, ///< Use external trigger state as LSB
    LSB_AUX_IN_1 = 3, ///< Use AUX I/O 1 state as LSB
    LSB_AUX_IN_2 = 2  ///< Use AUX I/O 2 state as LSB
};

/**
 * @cond INTERNAL_DECLARATIONS
 */
#define LSB_AUX_IN_0 2 // deprecated
                       /**
                        * @endcond
                        */

/**
 * @brief AlazarTech board personalities
 */
enum ALAZAR_BOARD_PERSONALITIES {
    BOARD_PERSONALITY_DEFAULT = 0,
    BOARD_PERSONALITY_8KFFT = 1
};

/**
 * @cond INTERNAL_DECLARATIONS
 */
typedef enum ALAZAR_BOARD_PERSONALITIES ALAZAR_BOARD_PERSONALITIES;
/**
 * @endcond
 */

#endif //_ALAZARCMD_H
