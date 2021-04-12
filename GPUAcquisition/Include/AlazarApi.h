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

#ifndef _ALAZARAPI_H
#define _ALAZARAPI_H

#ifdef __cplusplus
extern "C" {
#endif

#if ((!defined(_WIN32)) && (!defined(_DRIVER_)))
#    include <stdbool.h>
#endif

#include "AlazarCmd.h"
#include "AlazarError.h"

#ifdef _WIN32
#    include "wchar.h"
#endif // _WIN32

/**
 *  @cond INTERNAL_DECLARATIONS
 */
#ifdef _WIN32
#    ifndef _DRIVER_
#        include <windows.h>
#    endif
typedef signed char S8, *PS8;
typedef unsigned char U8, *PU8, BOOLEAN;
typedef signed short S16, *PS16;
typedef unsigned short U16, *PU16;
typedef signed long S32, *PS32;
typedef unsigned long U32, *PU32;
typedef __int64 S64, *PS64;
typedef unsigned __int64 U64, *PU64;
typedef void *HANDLE;
#elif defined(_OSX_)
typedef uint8_t BOOLEAN;
typedef void *HANDLE;
typedef void *PVOID;
typedef int16_t S16;
typedef int32_t S32;
typedef int64_t S64;
typedef uint8_t UCHAR;
typedef uint16_t USHORT;
typedef uint8_t U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;
#else
#    include <linux/types.h>
#    ifndef __PCITYPES_H // Types are already defined in PciTypes.h
typedef __s8 S8, *PS8;
typedef __u8 U8, *PU8, BOOLEAN;
typedef __s16 S16, *PS16;
typedef __u16 U16, *PU16;
typedef __s32 S32, *PS32;
typedef __u32 U32, *PU32;
typedef __s64 LONGLONG, S64, *PS64;
typedef __u64 ULONGLONG, U64, *PU64;
typedef void *HANDLE;
typedef int PLX_DRIVER_HANDLE; // Linux-specific driver handle
typedef void VOID, *PVOID;
#        define INVALID_HANDLE_VALUE (HANDLE) - 1
#    endif // __PCITYPES_H
#endif     // _WIN32

#ifndef EXPORT
#    define EXPORT
#endif

#ifndef BOOL
#    define BOOL int
#endif

#ifndef TRUE
#    define TRUE 1
#endif

#ifndef FALSE
#    define FALSE 0
#endif

#ifndef BYTE
#    define BYTE U8
#endif

#ifndef INT64
#    define INT64 S64
#endif

/**
 *  @endcond
 */

/**
 * @brief Existing board models
 */
typedef enum BoardTypes {
    ATS_NONE = 0,
    ATS850 = 1,
    ATS310 = 2,
    ATS330 = 3,
    ATS855 = 4,
    ATS315 = 5,
    ATS335 = 6,
    ATS460 = 7,
    ATS860 = 8,
    ATS660 = 9,
    ATS665 = 10,
    ATS9462 = 11,
    ATS9434 = 12,
    ATS9870 = 13,
    ATS9350 = 14,
    ATS9325 = 15,
    ATS9440 = 16,
    ATS9410 = 17,
    ATS9351 = 18,
    ATS9310 = 19,
    ATS9461 = 20,
    ATS9850 = 21,
    ATS9625 = 22,
    ATG6500 = 23,
    ATS9626 = 24,
    ATS9360 = 25,
    AXI8870 = 26,
    ATS9370 = 27,
    ATU7825 = 28,
    ATS9373 = 29,
    ATS9416 = 30,
    ATS9637 = 31,
    ATS9120 = 32,
    ATS9371 = 33,
    ATS9130 = 34,
    ATS9352 = 35,
    ATS9453 = 36,
    ATS_LAST
} ALAZAR_BOARDTYPES;

/**
 * @cond INTERNAL_DECLARATIONS
 */
typedef struct _BoardDef {
    U32 RecordCount;
    U32 RecLength;
    U32 PreDepth;
    U32 ClockSource;
    U32 ClockEdge;
    U32 SampleRate;
    U32 CouplingChanA;
    U32 InputRangeChanA;
    U32 InputImpedChanA;
    U32 CouplingChanB;
    U32 InputRangeChanB;
    U32 InputImpedChanB;
    U32 TriEngOperation;
    U32 TriggerEngine1;
    U32 TrigEngSource1;
    U32 TrigEngSlope1;
    U32 TrigEngLevel1;
    U32 TriggerEngine2;
    U32 TrigEngSource2;
    U32 TrigEngSlope2;
    U32 TrigEngLevel2;
} BoardDef, *pBoardDef;

#define FPGA_GETFIRST 0xFFFFFFFF
#define FPGA_GETNEXT 0xFFFFFFFE
#define FPGA_GETLAST 0xFFFFFFFC

RETURN_CODE EXPORT AlazarGetOEMFPGAName(int opcodeID, char *FullPath,
                                        unsigned long *error);
RETURN_CODE EXPORT AlazarOEMSetWorkingDirectory(char *wDir,
                                                unsigned long *error);
RETURN_CODE EXPORT AlazarOEMGetWorkingDirectory(char *wDir,
                                                unsigned long *error);
RETURN_CODE EXPORT AlazarParseFPGAName(const char *FullName, char *Name,
                                       U32 *Type, U32 *MemSize, U32 *MajVer,
                                       U32 *MinVer, U32 *MajRev, U32 *MinRev,
                                       U32 *error);
RETURN_CODE EXPORT AlazarDownLoadFPGA(HANDLE handle, char *FileName,
                                      U32 *RetValue);
RETURN_CODE EXPORT AlazarOEMDownLoadFPGA(HANDLE handle, char *FileName,
                                         U32 *RetValue);

#define ADMA_CLOCKSOURCE 0x00000001
#define ADMA_CLOCKEDGE 0x00000002
#define ADMA_SAMPLERATE 0x00000003
#define ADMA_INPUTRANGE 0x00000004
#define ADMA_INPUTCOUPLING 0x00000005
#define ADMA_IMPUTIMPEDENCE 0x00000006
#define ADMA_INPUTIMPEDANCE 0x00000006
#define ADMA_EXTTRIGGERED 0x00000007
#define ADMA_CHA_TRIGGERED 0x00000008
#define ADMA_CHB_TRIGGERED 0x00000009
#define ADMA_TIMEOUT 0x0000000A
#define ADMA_THISCHANTRIGGERED 0x0000000B
#define ADMA_SERIALNUMBER 0x0000000C
#define ADMA_SYSTEMNUMBER 0x0000000D
#define ADMA_BOARDNUMBER 0x0000000E
#define ADMA_WHICHCHANNEL 0x0000000F
#define ADMA_SAMPLERESOLUTION 0x00000010
#define ADMA_DATAFORMAT 0x00000011
/**
 * @endcond
 */

/**
 * @brief Traditional Record Header Substructure 1
 */
struct _HEADER0 {
    unsigned int SerialNumber : 18; ///< 18-bit serial number of this board as a
                                    /// signed integer
    unsigned int
        SystemNumber : 4; ///< 4-bit system identifier number for this board
    unsigned int WhichChannel : 1; ///< 1-bit input channel of this header. 0 is
                                   /// channel A, 1 is channel B
    unsigned int
        BoardNumber : 4; ///< 4-bit board identifier number of this board
    unsigned int SampleResolution : 3; ///< 3-bit reserved field
    unsigned int DataFormat : 2;       ///< 2-bit reserved field
};

/**
 * @brief Traditional Record Header Substructure 1
 */
struct _HEADER1 {
    unsigned int
        RecordNumber : 24;      ///< 24-bit index of record in the acquisition
    unsigned int BoardType : 8; ///< 8-bit board type identifier. See \ref
                                /// BoardTypes for a list of existing board
    /// types.
};

/**
 * @brief Traditional Record Header Substructure 2
 */
struct _HEADER2 {
    unsigned int TimeStampLowPart; ///< Lower 32 bits of 40-bit record timestamp
};

/**
 * @brief Traditional Record Header Substructure 3
 */
struct _HEADER3 {
    unsigned int TimeStampHighPart : 8; ///< 8-bit field containing the upper
                                        /// part of the 40-bit record timestamp
    unsigned int ClockSource : 2;   ///< 2-bit clock source identifier. See \ref
                                    /// ALAZAR_CLOCK_SOURCES
    unsigned int ClockEdge : 1;     ///< 1-bit clock edge identifier. See \ref
                                    /// ALAZAR_CLOCK_EDGES
    unsigned int SampleRate : 7;    ///< 7-bit sample rate identifier. See \ref
                                    /// ALAZAR_SAMPLE_RATES
    unsigned int InputRange : 5;    ///< 5-bit input range identifier. See \ref
                                    /// ALAZAR_INPUT_RANGES
    unsigned int InputCoupling : 2; ///< 2-bit input coupling identifier. See
                                    ///\ref ALAZAR_COUPLINGS
    unsigned int InputImpedence : 2; ///< 2-bit input impedance identifier. See
                                     ///\ref ALAZAR_IMPEDANCES
    unsigned int ExternalTriggered : 1; ///< 1-bit field set if and only if TRIG
                                        /// IN on this board caused the board to
    /// capture this record.
    unsigned int ChannelBTriggered : 1; ///< 1-bit field set if and only if CH B
                                        /// on this board caused the board to
    /// capture this record.
    unsigned int ChannelATriggered : 1; ///< 1-bit field set if and only if CH A
                                        /// on this board caused the board to
    /// capture this record.
    unsigned int TimeOutOccurred : 1; ///< 1-bit field set if and only if a
                                      /// timeout on a trigger engine on this
    /// board caused it to capture this record.
    unsigned int ThisChannelTriggered : 1; ///< 1-bit field set if and only if
                                           /// the channel specified by \ref
                                           /// _HEADER0::WhichChannel caused the
    /// board to capture this record.
};

/**
 * @brief Traditional Record Header
 */
typedef struct _ALAZAR_HEADER {
    struct _HEADER0 hdr0; ///< Substructure 0
    struct _HEADER1 hdr1; ///< Substructure 1
    struct _HEADER2 hdr2; ///< Substructure 2
    struct _HEADER3 hdr3; ///< Substructure 3
} * PALAZAR_HEADER;

/**
 * @brief Traditional Record Header Typedef
 */
typedef struct _ALAZAR_HEADER ALAZAR_HEADER;

/**
 * @cond INTERNAL_DECLARATIONS
 */
typedef enum _AUTODMA_STATUS {
    ADMA_Completed = 0,
    ADMA_Buffer1Invalid,
    ADMA_Buffer2Invalid,
    ADMA_BoardHandleInvalid,
    ADMA_InternalBuffer1Invalid,
    ADMA_InternalBuffer2Invalid,
    ADMA_OverFlow,
    ADMA_InvalidChannel,
    ADMA_DMAInProgress,
    ADMA_UseHeaderNotSet,
    ADMA_HeaderNotValid,
    ADMA_InvalidRecsPerBuffer,
    ADMA_InvalidTransferOffset,
    ADMA_InvalidCFlags
} AUTODMA_STATUS;
#define ADMA_Success ADMA_Completed

typedef enum _MSILS {
    KINDEPENDENT = 0,
    KSLAVE = 1,
    KMASTER = 2,
    KLASTSLAVE = 3
} MSILS;

RETURN_CODE EXPORT AlazarReadWriteTest(HANDLE handle, U32 *Buffer,
                                       U32 SizeToWrite, U32 SizeToRead);
RETURN_CODE EXPORT AlazarMemoryTest(HANDLE handle, U32 *errors);
RETURN_CODE EXPORT AlazarBusyFlag(HANDLE handle, int *BusyFlag);
RETURN_CODE EXPORT AlazarTriggeredFlag(HANDLE handle, int *TriggeredFlag);
/**
 * @endcond
 */

/**
 * @class default_return_values
 *
 * @returns #ApiSuccess upon success, or an error code. See #RETURN_CODE for
 * more detailed information.
 */

/**
 * @class deprecated_function
 *
 * @deprecated This function is obsolete. Do not use it in new designs
 */

/**
 * @brief Get the driver library version. This is the version of ATSApi.dll
 * under Windows, or ATSApi.so under Linux.
 *
 * @param[out] major The SDK major version number
 * @param[out] minor The SDK minor version number
 * @param[out] revision The SDK revision number
 *
 * @copydoc default_return_values
 *
 * @remark
 * Note that the version number returned is that of the driver library file, not
 * the ATS-SDK version number. SDK releases are given a version number with the
 * format X.Y.Z where: X is the major release number, Y is the minor release
 * number, and Z is the minor revision number.
 *
 * @sa AlazarGetCPLDVersion()
 * @sa AlazarGetDriverVersion()
 */
RETURN_CODE EXPORT AlazarGetSDKVersion(U8 *major, U8 *minor, U8 *revision);

/**
 * @brief Get the device driver version of the most recently opened device.
 *
 * Driver releases are given a version number with the format X.Y.Z where: X is
 * the major release number, Y is the minor release number, and Z is the minor
 * revision number.
 *
 * @param[out] major The driver major version number
 * @param[out] minor The driver minor version number
 * @param[out] revision The driver revision number
 *
 * @copydoc default_return_values
 *
 * @sa AlazarGetSDKVersion() AlazarGetCPLDVersion()
 */
RETURN_CODE EXPORT AlazarGetDriverVersion(U8 *major, U8 *minor, U8 *revision);

/**
 * @brief Get the PCB hadware revision level of a digitizer board.
 *
 * AlazarTech periodically updates the PCB hadware of its digitizers to improve
 * functionality. Many PCIE digitizers can report the PCB hadware revision to
 * software. Note that this function is not supported on PCI digitizer boards.
 *
 * @param[in] handle The board handle
 * @param[out] major PCB major version number
 * @param[out] minor PCB minor version number
 *
 * @copydoc default_return_values
 *
 */
RETURN_CODE EXPORT AlazarGetBoardRevision(HANDLE handle, U8 *major, U8 *minor);

/**
 * @brief Determine the number of digitizer boards that were detected in all
 * board systems.
 *
 * @returns The total number of digitizer boards detected.
 *
 * @sa AlazarNumOfSystems()
 */
U32 EXPORT AlazarBoardsFound(void);

/**
 * @brief Open and initialize a board
 *
 * The ATS library manages board handles internally. This function should only
 * be used in applications that are written for single board digitizer systems.
 *
 * @copydoc deprecated_function
 *
 * @param[in] boardName Name of board created by driver. For example “ATS850-0”.
 */
HANDLE EXPORT AlazarOpen(char *boardName); // e.x. ATS850-0, ATS850-1 ....

/**
 * @brief This routine will close the AUTODMA capabilities of the device.
 *
 * Only call this upon exit or error.
 *
 * @copydoc deprecated_function
 *
 * @param[in] handle Board handle
 *
 * @copydoc default_return_values
 */
void EXPORT AlazarClose(HANDLE handle);

/**
 * @brief Get a board model identifier of the board associated with a board
 * handle.
 *
 * @param[in] handle Board handle
 *
 * @returns A non-zero board model identifier upon success. See \ref BoardTypes
 * for converting the identifier into a board model.
 *
 * @return Zero upon error.
 */
U32 EXPORT AlazarGetBoardKind(HANDLE handle);

/**
 * @brief Get the CPLD version number of the specified board.
 *
 * @param[in] handle Board handle
 * @param[out] major CPLD version number
 * @param[out] minor CPLD version number
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarGetCPLDVersion(HANDLE handle, U8 *major, U8 *minor);

/**
 * @cond INTERNAL_DECLARATIONS
 */
RETURN_CODE EXPORT AlazarAutoCalibrate(HANDLE handle);
/**
 * @endcond
 */

/**
 * @brief Get the total on-board memory in samples, and sample size in
 * bits per sample.
 *
 * @param[in] handle Board handle.
 *
 * @param[out] memorySize Total size of the on-board memory in samples.
 *
 * @param[out] bitsPerSample Bits per sample.
 *
 * @copydoc default_return_values
 *
 * @remark The memory size information is independant of how many channels the
 * board can acquire on simultaneously. When multiple channels acquire data,
 * they share this amount.
 *
 * @remark The memory size indication is given for the default packing mode. See
 * documentation about data packing for more information.
 */
RETURN_CODE EXPORT AlazarGetChannelInfo(HANDLE handle, U32 *memorySize,
                                        U8 *bitsPerSample);

/**
 * @brief Get the total on-board memory in samples, and sample size in
 * bits per sample.
 *
 * @param[in] handle Board handle.
 *
 * @param[out] memorySize Total size of the on-board memory in samples.
 *
 * @param[out] bitsPerSample Bits per sample.
 *
 * @copydoc default_return_values
 *
 * @remark The memory size information is independant of how many channels the
 * board can acquire on simultaneously. When multiple channels acquire data,
 * they share this amount.
 *
 * @remark The memory size indication is given for the default packing mode. See
 * documentation about data packing for more information.
 */
RETURN_CODE EXPORT AlazarGetChannelInfoEx(HANDLE handle, S64 *memorySize,
                                          U8 *bitsPerSample);

/**
 * @brief Select the input coupling, range, and impedance of a digitizer
 * channel.
 *
 * @param[in] handle Board handle.
 *
 * @param[in] channel The channel to control. See \ref ALAZAR_CHANNELS for a
 * list of possible values. This parameter only takes unsigned 8-bit values. To
 * configure channel I and above, see AlazarInputControlEx().
 *
 * @param[in] inputRange Specify the input range of the selected channel. See
 * \ref ALAZAR_INPUT_RANGES for a list of all existing input ranges. Consult
 * board-specific information to see which input ranges are supported by each
 * board.
 *
 * @param[in] coupling Specifies the coupling of the selected chanel. Must be an
 * element of \ref ALAZAR_COUPLINGS
 *
 * @param[in] impedance Specify the input impedance to set for the selected
 * channel. See \ref ALAZAR_IMPEDANCES for a list of all existing values. See
 * the board-specific documentation to see impedances supported by various
 * boards.
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarInputControl(HANDLE handle, U8 channel, U32 coupling,
                                      U32 inputRange, U32 impedance);

/**
 * @brief Select the input coupling, range and impedance of a digitizer channel.
 *
 * This function is the equivalent of AlazarInputControl() with a U32-typed
 * parameter to pass the channel. This allows for boards with more than 8
 * channels to be configured.
 */
RETURN_CODE EXPORT AlazarInputControlEx(HANDLE handle, U32 channel,
                                        U32 couplingId, U32 rangeId,
                                        U32 impedenceId);

/**
 *  @cond INTERNAL_DECLARATIONS
 */
RETURN_CODE EXPORT AlazarSetPosition(HANDLE handle, U8 Channel, int PMPercent,
                                     U32 InputRange);
/**
 *  @endcond
 */

/**
 * @brief Set the external trigger range and coupling
 *
 * @param[in] handle Board handle
 *
 * @param[in] couplingId Specifies the external trigger coupling. See \ref
 * ALAZAR_COUPLINGS for existing values. Consult board-specific information to
 * see which values are supported by each board.
 *
 * @param[in] rangeId Specifies the external trigger range. See \ref
 * ALAZAR_EXTERNAL_TRIGGER_RANGES for a list of all existing values. Consult
 * board-specific information to see which values are supported by each board.
 */
RETURN_CODE EXPORT AlazarSetExternalTrigger(HANDLE handle, U32 couplingId,
                                            U32 rangeId);

/**
 * @brief Set the time, in sample clocks, to wait after receiving a trigger
 * event
 * before capturing a record for the trigger.
 *
 * @param[in] handle Board handle
 *
 * @param[in] Delay Trigger delay in sample clocks. Must be a value between 0
 * and 9 999 999. It must also be a multiple of a certain value that varies from
 * one board to another. See board-specific information to know which multiplier
 * must be respected.
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetTriggerDelay(HANDLE handle, U32 Delay);

/**
 * @brief Set the time to wait for a trigger event before automatically
 * generating a trigger event.
 *
 * @param[in] handle Board handle
 *
 * @param[in] timeout_ticks The trigger timeout value in ticks. A tick is 10μs.
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetTriggerTimeOut(HANDLE handle, U32 timeout_ticks);

/**
 * @cond INTERNAL_DECLARATIONS
 */
U32 EXPORT AlazarTriggerTimedOut(HANDLE h);
/**
 * @endcond
 */

/**
 * @brief Get the timestamp and trigger address of the trigger event in a record
 * acquired to on-board memory.
 *
 * The following code fragment demonstrates how to convert the trigger timestamp
 * returned by AlazarGetTriggerAddress() from counts to seconds.
 *
 * @code{.cpp}
 * __int64 timeStamp_cnt;
 * timeStamp_cnt = ((__int64) timestampHighPart) << 8;
 * timeStamp_cnt |= timestampLowPart & 0x0ff;
 * double samplesPerTimestampCount = 2; // board specific constant
 * double samplesPerSec = 50.e6; // sample rate
 * double timeStamp_sec = (double) samplesPerTimestampCount *
 * timeStamp_cnt / samplesPerSec;
 * @endcode
 *
 * The sample per timestamp count value depends on the board model. See
 * board-specific information to know which value applies to which board.
 *
 * @param[in] handle Board handle
 *
 * @param[in] Record Record in acquisition (1-indexed)
 *
 * @param[out] TriggerAddress The trigger address
 *
 * @param[out] TimeStampHighPart The most significant 32-bits of a record
 * timestamp
 *
 * @param[out] TimeStampLowPart The least significant 8-bits of a record
 * timestamp
 *
 * @returns #ApiError2 (604) if it is called after a dual-port acquisition. This
 * function should be called after a single-port acquisition only.
 *
 * @copydoc default_return_values
 *
 * @remark This function can be used in single-port acquisitions only.
 */
RETURN_CODE EXPORT AlazarGetTriggerAddress(HANDLE handle, U32 Record,
                                           U32 *TriggerAddress,
                                           U32 *TimeStampHighPart,
                                           U32 *TimeStampLowPart);

/**
 * @brief Configures the trigger system.
 *
 * @param[in] handle Board handle
 *
 * @param[in] TriggerOperation Specify how the two independant trigger engines
 * generate a trigger. This can be one of \ref ALAZAR_TRIGGER_OPERATIONS
 *
 * @param[in] TriggerEngine1 First trigger engine to configure. Can be one of
 * \ref ALAZAR_TRIGGER_ENGINES.
 *
 * @param[in] Source1 Signal source for the first trigger engine. Can be one of
 * \ref ALAZAR_TRIGGER_SOURCES.
 *
 * @param[in] Slope1 Sign Direction of the trigger voltage slope that will
 * generate a trigger event for the first engine. Can be one of \ref
 * ALAZAR_TRIGGER_SLOPES.
 *
 * @param[in] Level1 Select the voltage level that the trigger signal must cross
 * to generate a trigger event.
 *
 * @param[in] TriggerEngine2 Second trigger engine to configure. Can be one of
 * \ref ALAZAR_TRIGGER_ENGINES.
 *
 * @param[in] Source2 Signal source for the second trigger engine. Can be one of
 * \ref ALAZAR_TRIGGER_SOURCES.
 *
 * @param[in] Slope2 Sign Direction of the trigger voltage slope that will
 * generate a trigger event for the second engine. Can be one of \ref
 * ALAZAR_TRIGGER_SLOPES.
 *
 * @param[in] Level2 Select the voltage level that the trigger signal must cross
 * to generate a trigger event.
 *
 * @remark The trigger level is specified as an unsigned 8-bit code that
 * represents a fraction of the full scale input voltage of the trigger source:
 * 0 represents the negative limit, 128 represents the 0 level, and 255
 * represents the positive limit. For example, if the trigger source is CH A,
 * and the CH A input range is ± 800 mV, then 0 represents a –800 mV trigger
 * level, 128 represents a 0 V trigger level, and 255 represents +800 mV trigger
 * level.\n
 *
 * @remark If the trigger source is external, the full scale input voltage for
 * the external trigger connector is dictated by the AlazarSetExternalTrigger()
 * function.
 *
 * @remark All PCI Express boards except ATS9462 support only one external
 * trigger level. If both \c Source1 and \c Source2 are set to \ref
 * TRIG_EXTERNAL of \ref ALAZAR_TRIGGER_SOURCES, \c Level1 is ignored and only
 * \c Level2 is used. All other boards support using different values for the
 * two levels.
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetTriggerOperation(HANDLE handle,
                                             U32 TriggerOperation,
                                             U32 TriggerEngine1, U32 Source1,
                                             U32 Slope1, U32 Level1,
                                             U32 TriggerEngine2, U32 Source2,
                                             U32 Slope2, U32 Level2);

/**
 * @brief Retrieve the timestamp, in sample clock periods, of a record acquired
 * to
 * on-board memory.
 *
 * @param[in] handle Board handle
 *
 * @param[in] Record 1-indexed record in acquisition
 *
 * @param[in] Timestamp_samples Record timestamp, in sample clock periods
 *
 * @copydoc default_return_values
 *
 * @remark This function is part of the single-port data acquisition API. It
 * cannot be used to retrieve the timestamp of records acquired using dual-port
 * AutoDMA APIs.
 */
RETURN_CODE EXPORT AlazarGetTriggerTimestamp(HANDLE handle, U32 Record,
                                             U64 *Timestamp_samples);

/**
 * Configure the trigger engines of a board to use an external trigger input
 * and, optionally, synchronize the start of an acquisition with the next
 * external trigger event after AlazarStartCapture() is called.
 *
 * @param[in] handle Board handle
 *
 * @param[in] slopeId Select the direction of the rate of change of the external
 * trigger signal when it crosses the trigger voltage level that is required to
 * generate a trigger event. Must be an element of \ref ALAZAR_TRIGGER_SLOPES.
 *
 * @param[in] level Specify a trigger level code representing the trigger level
 * in volts that an external trigger signal connected must pass through to
 * generate a trigger event. See the Remarks section below.
 *
 * @param[in] options The options parameter may be one of \ref
 * ALAZAR_STOS_OPTIONS
 *
 * @copydoc default_return_values
 *
 * @remark AlazarSetTriggerOperationForScanning() is intended for scanning
 * applications that supply both external clock and external trigger signals to
 * the digitizer, where the external clock is not suitable to drive the
 * digitizer in between trigger events.
 *
 * @remark This function configures a board to use trigger operation
 * TRIG\_ENGINE\_OP\_J, and the source of TRIG\_ENGINE\_J to be TRIG\_EXTERNAL.
 * The application must call AlazarSetExternalTrigger() to set the full-scale
 * external input range and coupling of the external trigger signal connected to
 * the TRIG IN connector. An application should call
 * AlazarSetTriggerOperationForScanning() or AlazarSetTriggerOperation(), but
 * not both.
 *
 * @remark The trigger level is specified as an unsigned 8-bit code that
 * represents a fraction of the full scale input voltage of the external trigger
 * range: 0 represents the negative limit, 128 represents the 0 level, and 255
 * represents the positive limit.
 *
 * @remark AlazarSetTriggerOperationForScanning() in currently only supported on
 * ATS9462 with FPGA 35.0 or later.
 */
RETURN_CODE EXPORT AlazarSetTriggerOperationForScanning(HANDLE handle,
                                                        U32 slopeId, U32 level,
                                                        U32 options);

/**
 * @brief Abort an acquisition to on-board memory.
 *
 * @param[in] handle Board handle
 *
 * @copydoc default_return_values
 *
 * @note This function is part of the single-port API. It should be used only in
 * this context. To abort dual-port acquisitions, see AlazarAbortAsyncRead().
 */
RETURN_CODE EXPORT AlazarAbortCapture(HANDLE handle);

/**
 * @brief Generate a software trigger event.
 *
 * @param[in] handle Board handle
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarForceTrigger(HANDLE handle);

/**
 * @brief Generate a software trigger enable event.
 *
 * @param[in] handle Board handle
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarForceTriggerEnable(HANDLE handle);

/**
 * @brief Arm a board to start an acquisition
 *
 * @copydoc default_return_values
 *
 * @remark Only call on the master board in a master-slave system.
 */
RETURN_CODE EXPORT AlazarStartCapture(HANDLE handle);

/**
 * @cond INTERNAL_DECLARATIONS
 */
RETURN_CODE EXPORT AlazarCaptureMode(HANDLE handle, U32 Mode);

RETURN_CODE EXPORT AlazarStreamCapture(HANDLE handle, void *Buffer,
                                       U32 BufferSize, U32 DeviceOption,
                                       U32 ChannelSelect, U32 *error);
/**
 * @endcond
 */

/**
 * @brief Enable the on-board FPGA to process records acquired to on-board
 * memory, and transfer the processed data to host memory.
 *
 * HyperDisp processing enables the on-board FPGA to divide a record
 * acquired to on-board memory into intervals, find the minimum and maximum
 * sample values during each interval, and transfer an array of minimum and
 * maximum sample values to a buffer in host memory. This allows the acquisition
 * of relatively long records to on-board memory, but the transfer of relatively
 * short, processed records to a buffer in host memory.
 *
 * For example, it would take an ATS860-256M about ~2.5 seconds to transfer
 * a 250,000,000 sample record from on-board memory, across the PCI bus, to
 * a buffer in host memory. With HyperDisp enabled, it would take the
 * on-board FPGA a fraction of a second to process the record and transfer
 * a few hundred samples from on-board memory, across the PCI bus, to a
 * buffer in host memory.
 *
 * @param[in] handle Board handle
 *
 * @param[in] buffer Reseved (Set to NULL)
 *
 * @param[in] bufferSize Number of samples to process
 *
 * @param[out] viewBuffer Buffer to receive processed data
 *
 * @param[in] viewBufferSize Size of processed data buffer in bytes
 *
 * @param[in] numOfPixels Number of HyperDisp points
 *
 * @param[in] option Processing mode. Pass 1 to enable HyperDisp processing.
 *
 * @param[in] channelSelect Channel to process
 *
 * @param[in] record Record to process (1-indexed)
 *
 * @param[in] transferOffset The offset, in samples, of first sample to process
 * relative to the trigger position in record.
 *
 * @param[out] error Pointer to value to receive a result code.
 *
 * @copydoc default_return_values
 *
 * @note This function is part of the single-port data acquisition API. It
 * cannot be used with the dual-port AutoDMA APIs.
 */
RETURN_CODE EXPORT AlazarHyperDisp(HANDLE handle, void *buffer, U32 bufferSize,
                                   U8 *viewBuffer, U32 viewBufferSize,
                                   U32 numOfPixels, U32 option,
                                   U32 channelSelect, U32 record,
                                   long transferOffset, U32 *error);

/**
 * @cond INTERNAL_DECLARATIONS
 */
RETURN_CODE EXPORT AlazarFastPRRCapture(HANDLE handle, void *Buffer,
                                        U32 BufferSize, U32 DeviceOption,
                                        U32 ChannelSelect, U32 *error);
/**
 * @endcond
 */

/**
 * @brief Determines if an acquisition is in progress.
 *
 * @param[in] handle Board handle
 *
 * @returns If the board is busy acquiring data to on-board memory, this
 * function returns 1. Otherwise, it returns 0.
 *
 * @note This function is part of the single-port data acquisition API. It
 * cannot be used with the dual-port AutoDMA APIs.
 */
U32 EXPORT AlazarBusy(HANDLE handle);

/**
 * @brief Determine if a board has triggered during the current acquisition
 *
 * @param[in] handle Board handle
 *
 * @returns If the board has received at least one trigger event since the last
 * call to AlazarStartCapture(), this function returns 1. Otherwise, it returns
 * 0.
 *
 * @note This function is part of the single-port data acquisition API. It
 * cannot be used with the dual-port AutoDMA APIs.
 */
U32 EXPORT AlazarTriggered(HANDLE handle);

/**
 * @brief Return a bitmask with board status information.
 *
 * @param[in] handle Board handle
 *
 * @return If the function fails, the return value is 0xFFFFFFFF. Upon success,
 * the return value is a bit mask of the following values:
 *  - 1 : At least one trigger timeout occured.
 *  - 2 : At least one channel A sample was out of range during the last
 * acquisition.
 *  - 4 : At least one channel B sample was out of range during the last
 * acquisition.
 *  - 8 : PLL is locked (ATS660 only)
 *
 * @note This function is part of the single-port data acquisition API. It
 * cannot be used with the dual-port AutoDMA APIs.
 */
U32 EXPORT AlazarGetStatus(HANDLE handle);

/**
 * @cond INTERNAL_DECLARATIONS
 */
U32 EXPORT AlazarDetectMultipleRecord(HANDLE handle);
/**
 * @endcond
 */

/**
 * @brief Select the number of records to capture to on-board memory.
 *
 * @param[in] handle Board handle
 *
 * @param[in] Count The number of records to acquire to on-board memory during
 * the acquisition.
 *
 * @copydoc default_return_values
 *
 * @remark The maximum number of records per capture is a function of the board
 * type, the maximum number of samples per channel (SPC), and the current number
 * of samples per record (SPR) :
 *  - ATS850, ATS310, ATS330   : min(SPC / (SPR + 16), 10000)
 *  - ATS460, ATS660, ATS9462  : min(SPC / (SPR + 16), 256000)
 *  - ATS860, ATS9325, ATS935X : min(SPC / (SPR + 32), 256000)
 *  - ATS9850, ATS9870         : min(SPC / (SPR + 64), 256000)
 *
 * @note This function is part of the single-port API, and cannot be used in a
 * dual-port context.
 *
 */
RETURN_CODE EXPORT AlazarSetRecordCount(HANDLE handle, U32 Count);

/**
 * @brief Set the number of pre-trigger and post-trigger samples per record.
 *
 * @param[in] handle Board handle
 *
 * @param[in] preTriggerSamples Number of samples before the trigger position in
 * each record.
 *
 * @param[in] postTriggerSamples Number of samples after the trigger position in
 * each record.
 *
 * @remark The number of pre-trigger samples must not exceed the number of
 * samples per record minus 64.
 *
 * @remark The number of samples per record is the sum of the pre- and
 * post-trigger samples. It must follow requirements specific to each board
 * listed in the board-specific documentation.
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetRecordSize(HANDLE handle, U32 preTriggerSamples,
                                       U32 postTriggerSamples);

/**
 * @brief Configure the sample clock source, edge and decimation.
 *
 * @param[in] handle Board handle
 *
 * @param[in] source Clock source identifiers. Must be a member of \ref
 * ALAZAR_CLOCK_SOURCES. See board-specific information for a list of valid
 * values for each board. For external clock types, the identifier to choose may
 * depend on the clock's frequency. See board-specific information for a list of
 * frequency ranges for all clock types.
 *
 * @param[in] sampleRateIdOrValue If the clock source chosen is \ref
 * INTERNAL_CLOCK, this value is a member of \ref ALAZAR_SAMPLE_RATES that
 * defines the internal sample rate to choose. Valid values for each board vary.
 * If the clock source chosen is \ref EXTERNAL_CLOCK_10MHZ_REF, pass the value
 * of the sample clock to generate from the reference in herts. The values that
 * can be generated depend on the board model. Otherwise, the clock source is
 * external, pass \ref SAMPLE_RATE_USER_DEF to this parameter.
 *
 * @param[in] edgeId The external clock edge on which to latch sample rate. Must
 * be a member of \ref ALAZAR_CLOCK_EDGES.
 *
 * @param[in] decimation Decimation value. May be an integer between 0 and
 * 100000 with the following exceptions. Note that a decimation value of 0 means
 * disable decimation.
 *
 * - If an ATS460/ATS660/ATS860 is configured to use a \ref SLOW_EXTERNAL_CLOCK
 * clock source, the maximum decimation value is 1.
 *
 * - If an ATS9350 is configured to use an \ref EXTERNAL_CLOCK_10MHZ_REF clock
 * source, the decimation value must be 1, 2, 4 or any multiple of 5. Note that
 * the sample rate identifier value must be 500000000, and the sample rate will
 * be 500 MHz divided by the decimation value.
 *
 * - If an ATS9360 / ATS9371 / ATS9373 is configured to use an \ref
 * EXTERNAL_CLOCK_10MHZ_REF clock source, the maximum decimation value is 1.
 *
 * - If an ATS9850 is configured to use an \ref EXTERNAL_CLOCK_10MHZ_REF clock
 * source, the decimation value must be 1, 2, 4 or any multiple of 10. Note that
 * the sample rate identifier value must be 500000000, and the sample rate will
 * be 500 MHz divided by the decimation value.
 *
 * - If an ATS9870 is configured to use an \ref EXTERNAL_CLOCK_10MHZ_REF clock
 * source, the decimation value must be 1, 2, 4 or any multiple of 10. Note that
 * the sample rate identifier value must be 1000000000, and the sample rate will
 * be 1 GHz divided by the decimation value.
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetCaptureClock(HANDLE handle, U32 source,
                                         U32 sampleRateIdOrValue, U32 edgeId,
                                         U32 decimation);

/**
 * @brief Set the external clock comparator level.
 *
 * @param[in] handle Board handle
 *
 * @param[in] level_percent The external clock comparator level, in percent.
 *
 * @remark Only the following boards support this feature:
 * ATS460, ATS660, ATS860, ATS9350, ATS9351, ATS9352,
 * ATS9440, ATS9462, ATS9625, ATS9626, ATS9870.
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetExternalClockLevel(HANDLE handle,
                                               float level_percent);

/**
 * @cond INTERNAL_DECLARATIONS
 */
RETURN_CODE EXPORT AlazarSetClockSwitchOver(HANDLE handleBoard, U32 uMode,
                                            U32 uDummyClockOnTime_ns,
                                            U32 uReserved);

#define CSO_DISABLE 0
#define CSO_ENABLE_DUMMY_CLOCK 1
#define CSO_TRIGGER_LOW_DUMMY_CLOCK 2
/**
 * @endcond
 */

/**
 * @brief Read all of part of a record from on-board memory to host memory
 * (RAM).
 *
 * The record must be less than 2,147,483,648 samples long.
 *
 * @param[in] handle Board handle
 *
 * @param[in] channelId The channel identifier of the record to read.
 *
 * @param[out] buffer Buffer to receive sample data
 *
 * @param[in] elementSize Number of bytes per sample
 *
 * @param[in] record Index of the record to transfer (1-indexed)
 *
 * @param[in] transferOffset The offset, in samples, from the trigger position
 * in
 * the record, of the first sample to transfer.
 *
 * @param[in] transferLength The number of samples to transfer.
 *
 * @copydoc default_return_values
 *
 * @note AlazarRead() is part of the single-port API, it cannot be used in a
 * dual-port context.
 *
 * @remark AlazarRead() can transfer segments of a record. This may be useful if
 * a full record is too large to transfer as a single clock, or if only part of
 * a record is of interest.
 *
 * @remark Use AlazarReadEx() To transfer records with more than 2 billion
 * samples.
 */
U32 EXPORT AlazarRead(HANDLE handle, U32 channelId, void *buffer,
                      int elementSize, long record, long transferOffset,
                      U32 transferLength);

/**
 *  @brief Read all or part of a record from an acquisition to on-board memory
 *  from on-board memory to a buffer in hsot memory. The record may be longer
 *  than 2 billion samples.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] channelId channel identifier of record to read
 *
 *  @param[out] buffer Buffer to receive sample data
 *
 *  @param[in] elementSize number of bytes per sample
 *
 *  @param[in] record record in on-board memory to transfer to buffer
 *  (1-indexed).
 *
 *  @param[in] transferOffset The offset in samples from the trigger position in
 *  the record of the first sample in the record in on-board memory to transfer
 *  to the buffer
 *
 *  @param[in] transferLength The number of samples to transfer from the record
 *  in on-board memory to the buffer.
 *
 *  @copydoc default_return_values
 *
 *  @note AlazarReadEx() is part of the single-port data acquisition API. It
 *  cannot be used with the dual-port AutoDMA APIs.
 *
 *  @remark AlazarReadEx() can transfer segments of a record to on-board memory.
 *  This may be useful if a full record is too large to transfer as a single
 *  block, or if only part of a record is of interest.
 *
 *  Use AlazarRead() or AlazarReadEx() to transfer records with less than 2
 *  billion samples. Use AlazarReadEx() to transfer records with more than 2
 *  billion samples.
 */
U32 EXPORT AlazarReadEx(HANDLE handle, U32 channelId, void *buffer,
                        int elementSize, long record, INT64 transferOffset,
                        U32 transferLength);

#if defined(WIN32) && (!defined(_DRIVER_))
/**
 *  @brief Adds a buffer to the end of a list of available buffers to be filled
 *  by the board. When the board receives sufficient trigger events to fill the
 *  buffer, the event in the \c OVERLAPPED will be set to the signaled state.
 *
 *  You must call AlazarBeforeAsyncRead() before calling AlazarAsyncRead().
 *
 *  The \c bytesToRead parameter must be equal to the product of the number of
 *  bytes per record, the number of recods per buffer and the number of enabled
 *  channels. If record headers are enabled, the number of bytes per record must
 *  include the size of the record header (16 bytes).
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] buffer Pointer to a buffer to receive sample data from the
 *  digitizer board
 *
 *  @param[in] bytesToRead Number of bytes to read from the board
 *
 *  @param[in] overlapped Pointer to an \c OVERLAPPED structure. The event in
 *  thestructure is set to the signaled state when the read operation completes.
 *
 *  @returns If the function succeeds in adding the buffer to end of the list of
 *  buffers available to be filled by the board, it returns #ApiDmaPending.
 *  When the board fills the buffer, the event in the OVERLAPPED structure is
 *  set to the signaled state.
 *
 *  @returns If the function fails because the board overflowed its on board
 *  memory, it returns #ApiBufferOverflow. The board may overflow its on board
 *  memory because the rate at which it is acquiring data is faster than the
 *  rate at which it is transferring data from on-board memory to host memory.
 *  If this is the case, try reducing the sample rate, number of enabled
 *  channels, or amount of time spent processing each buffer.
 *
 *  @returns If the function fails because the buffer is too large for the
 *  driver or operating system to prepare for scatter-gather DMA transfer, it
 *  returns #ApiLockAndProbePagesFailed. Try reducing the size of each buffer,
 *  or reducing the number of buffers queued by the application.
 *
 *  @returns If the function fails for some other reason, it returns an error
 *  code that indicates the reason that it failed. See #RETURN_CODE for more
 *  information.
 *
 *  @remark AlazarAsyncRead() is only available under Windows
 *
 *  @warning You must call AlazarAbortAsyncRead() before your application exits
 *  if you have called AlazarAsyncRead() and buffers are pending.
 */
RETURN_CODE EXPORT AlazarAsyncRead(HANDLE handle, void *buffer, U32 bytesToRead,
                                   OVERLAPPED *overlapped);
#endif

/**
 * @brief Set a device parameter as a signed long value
 *
 * @param[in] handle Board handle
 *
 * @param[in] channel The channel to control. See \ref ALAZAR_CHANNELS for a
 * list of possible values. This parameter only takes unsigned 8-bit values.
 *
 * @param[in] parameter The Parameter to modify. This can be one of \ref
 * ALAZAR_PARAMETERS.
 *
 * @param[in] value Parameter value
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetParameter(HANDLE handle, U8 channel, U32 parameter,
                                      long value);

/**
 * @brief Set a device parameter as an unsigned long value
 *
 * @param[in] handle Board handle
 *
 * @param[in] channel The channel to control. See \ref ALAZAR_CHANNELS for a
 * list of possible values. This parameter only takes unsigned 8-bit values.
 *
 * @param[in] parameter The Parameter to modify. This can be one of \ref
 * ALAZAR_PARAMETERS.
 *
 * @param[in] value Parameter value. See \ref ALAZAR_PARAMETERS for details
 * about valid values
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetParameterUL(HANDLE handle, U8 channel,
                                        U32 parameter, U32 value);

/**
 * @brief Set a device parameter as a long long value
 *
 * @param[in] handle Board handle
 *
 * @param[in] channel The channel to control. See \ref ALAZAR_CHANNELS for a
 * list of possible values. This parameter only takes unsigned 8-bit values.
 *
 * @param[in] parameter The Parameter to modify. This can be one of \ref
 * ALAZAR_PARAMETERS.
 *
 * @param[in] value Parameter value
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetParameterLL(HANDLE handle, U8 channel,
                                        U32 parameter, S64 value);

/**
 * @brief Get a device parameter as a signed long value
 *
 * @param[in] handle Board handle
 *
 * @param[in] channel The channel to control. See \ref ALAZAR_CHANNELS for a
 * list of possible values. This parameter only takes unsigned 8-bit values.
 *
 * @param[in] parameter The Parameter to modify. This can be one of \ref
 * ALAZAR_PARAMETERS.
 *
 * @param[in] retValue Parameter's value
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarGetParameter(HANDLE handle, U8 channel, U32 parameter,
                                      long *retValue);

/**
 * @brief Get a device parameter as an unsigned long value
 *
 * @param[in] handle Board handle
 *
 * @param[in] channel The channel to control. See \ref ALAZAR_CHANNELS for a
 * list of possible values. This parameter only takes unsigned 8-bit values.
 *
 * @param[in] parameter The Parameter to modify. This can be one of \ref
 * ALAZAR_PARAMETERS.
 *
 * @param[in] retValue Parameter's value
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarGetParameterUL(HANDLE handle, U8 channel,
                                        U32 parameter, U32 *retValue);

/**
 * @brief Get a device parameter as a long long value
 *
 * @param[in] handle Board handle
 *
 * @param[in] channel The channel to control. See \ref ALAZAR_CHANNELS for a
 * list of possible values. This parameter only takes unsigned 8-bit values.
 *
 * @param[in] parameter The Parameter to modify. This can be one of \ref
 * ALAZAR_PARAMETERS.
 *
 * @param[in] retValue Parameter's value
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarGetParameterLL(HANDLE handle, U8 channel,
                                        U32 parameter, S64 *retValue);

/**
 *  @brief Return the handle of the master board in the specified board system.
 *
 *  @param[in] systemId System identification number
 *
 *  @remark If the board system specified contains a single, independent board,
 *  this function returns a handle to that board.
 *
 *  @copydoc default_return_values
 */
HANDLE EXPORT AlazarGetSystemHandle(U32 systemId);

/**
 *  @brief Get the total number of board systems detected
 *
 *  A *board system* is a group of one or more digitizer oards that share clock
 *  and trigger signals. A board system may be composed of a single independent
 *  board, or a group of two or more digitizer boards connected together with a
 *  *SyncBoard*.
 *
 *  @return The total number of board systems detected
 */
U32 EXPORT AlazarNumOfSystems(void);

/**
 *  @brief Returns the number of digitizer boards in a board system specified by
 *  its system identifier.
 *
 *  If this function is called with the identifier of a master-slave system, it
 *  returns the total number of boards in the system, including the master.
 *
 *  If this function is called with the identifier of an independent board
 *  system, it returns one.
 *
 *  If this fucntion is called with the identifier of an invalid board system,
 *  it returns zero.
 *
 *  @param[in] systemId The system identification number
 *
 *  @copydoc default_return_values
 */
U32 EXPORT AlazarBoardsInSystemBySystemID(U32 systemId);

/**
 *  @brief Return the number of digitizer boards in a board system specified by
 *  the handle of its master board.
 *
 *  If this function is called with the handle of to the master board in a
 *  master-slave system, it returns the total number of boards in the system.
 *
 *  If this function is called with the handle of an independent board, it
 *  returns 1.
 *
 *  If it is called with the handle to a slave in a master-slave system or with
 * an invalid
 *  handle, it returns 0.
 *
 *  @copydoc default_return_values
 */
U32 EXPORT AlazarBoardsInSystemByHandle(HANDLE systemHandle);

/**
 *  @brief Get a handle to a board in a board system where the board and system
 *  are identified by their ID.
 *
 *  Detailed description
 *
 *  @param[in] systemId The system identifier
 *
 *  @param[in] boardId The board identifier
 *
 *  @returns A handle to the specified board if it was found.
 *
 *  @returns NULL if the board with the specified \c systemId and \c boardId was
 *  not found.
 */
HANDLE EXPORT AlazarGetBoardBySystemID(U32 systemId, U32 boardId);

/**
 *  @brief Get a handle to a board in a board system where the board system is
 *  specified by a handle to its master board and the board by its identifier
 *  within the system.
 *
 *  @param[in] systemHandle Handle to master board
 *
 *  @param[in] boardId Board identifier in the board system
 *
 *  @returns A handle to the specified board if it was found
 *
 *  @returns NULL if the master board handle is invalid, or a board with the
 *  specified board identifier was not found in the specified board system.
 */
HANDLE EXPORT AlazarGetBoardBySystemHandle(HANDLE systemHandle, U32 boardId);

/**
 *  @brief Control the LED on a board's mounting bracket
 *
 *  @param[in] handle Board handle
 *
 *  @param[in] state to put the LED in. Must be a member of \ref ALAZAR_LED
 *
 *  @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetLED(HANDLE handle, U32 state);

/**
 *  @brief Get a device attribute as a unsigned 32-bit integer
 *
 *  @param[in] handle Board handle
 *
 *  @param[in] capability The board capability to query. Must be a member of
 *  \ref ALAZAR_CAPABILITIES.
 *
 *  @param[in] reserved Pass 0
 *
 *  @param[out] retValue Capability value
 *
 *  @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarQueryCapability(HANDLE handle, U32 capability,
                                         U32 reserved, U32 *retValue);

/**
 *  @brief Get a device attribute as a 64-bit integer
 *
 *  @param[in] handle Board handle
 *
 *  @param[in] capability The board capability to query. Must be a member of
 *  \ref ALAZAR_CAPABILITIES.
 *
 *  @param[in] reserved Pass 0
 *
 *  @param[out] retValue Capability value
 *
 *  @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarQueryCapabilityLL(HANDLE handle, U32 capability,
                                           U32 reserved, S64 *retValue);

/**
 * @cond INTERNAL_DECLARATIONS
 */
U32 EXPORT AlazarMaxSglTransfer(ALAZAR_BOARDTYPES bt);
/**
 * @endcond
 */

/**
 *  @brief Calculate the maximum number of records that can be captured to
 *  on-board memory given the requested number of samples per record.
 *
 *  @param[in] handle Board handle
 *
 *  @param[in] samplesPerRecord The desired number of samples per record
 *
 *  @param[out] maxRecordsPerCapture The maximum number of records per capture
 *  possible with the requested value of samples per record.
 *
 *  @copydoc default_return_values
 *
 *  @note This function is part of the single-port API. It should not be used
 *  with AutoDMA API functions.
 */
RETURN_CODE EXPORT AlazarGetMaxRecordsCapable(HANDLE handle,
                                              U32 samplesPerRecord,
                                              U32 *maxRecordsPerCapture);

/**
 *  @brief Return which event caused a board system to trigger and capture a
 *  record to on-board memory.
 *
 *  @param[in] systemHandle Handle to a master board in a board system.
 *
 *  @param[in] boardId Board identifier of a board in the specified system.
 *
 *  @param[in] recordNumber Record in acquisition (1-indexed)
 *
 *  @returns One of the following values:
 *   - 0 : This board did not cause the system to trigger
 *   - 1 : CH A on this board caused the system to trigger
 *   - 2 : CH B on this board caused the system to trigger
 *   - 3 : EXT TRIG IN on this board caused the system to trigger
 *   - 4 : Both CH A and CH B on this board caused the system to trigger
 *   - 5 : Both CH A and EXT TRIG IN on this board caused the system to trigger
 *   - 6 : Both CH B and EXT TRIG IN on this board caused the system to trigger
 *   - 7 : A trigger timeout on this board caused the system to trigger
 *
 *  @note This function is part of the single-port API. It cannot be used with
 *  the dual-port AutoDMA APIs.
 *
 *  @warning This API routine will not work with ATS850 version 1.2 hardware.
 *  Version 1.3 and higher version number of ATS850 are fully supported, as are
 *  all versions of ATS330 and ATS310.
 */
U32 EXPORT AlazarGetWhoTriggeredBySystemHandle(HANDLE systemHandle, U32 boardId,
                                               U32 recordNumber);

/**
 *  @brief Return which event caused a board system to trigger and capture a
 *  record to on-board memory.
 *
 *  @param[in] systemId System indentifier
 *
 *  @param[in] boardId Board identifier of a board in the specified system.
 *
 *  @param[in] recordNumber Record in acquisition (1-indexed)
 *
 *  @returns One of the following values:
 *   - 0 : This board did not cause the system to trigger
 *   - 1 : CH A on this board caused the system to trigger
 *   - 2 : CH B on this board caused the system to trigger
 *   - 3 : EXT TRIG IN on this board caused the system to trigger
 *   - 4 : Both CH A and CH B on this board caused the system to trigger
 *   - 5 : Both CH A and EXT TRIG IN on this board caused the system to trigger
 *   - 6 : Both CH B and EXT TRIG IN on this board caused the system to trigger
 *   - 7 : A trigger timeout on this board caused the system to trigger
 *
 *  @note This function is part of the single-port API. It cannot be used with
 *  the dual-port AutoDMA APIs.
 *
 *  @warning This API routine will not work with ATS850 version 1.2 hardware.
 *  Version 1.3 and higher version number of ATS850 are fully supported, as are
 *  all versions of ATS330 and ATS310.
 */
U32 EXPORT AlazarGetWhoTriggeredBySystemID(U32 systemId, U32 boardId,
                                           U32 recordNumber);

/**
 *  @brief Activates the bandwith limiter of an input channel. Not all boards
 *  support a bandwidth limiter. See board-specific documentation for more
 *  information.
 *
 *  @remark The bandwidth limiter is disabled by default. When enabled, the
 *  bandwith is limited to approximatively 20 MHz.
 *
 *  @param[in] handle Board handle
 *
 *  @param[in] channel The channel identifier. Must be a channel from \ref
 *  ALAZAR_CHANNELS.
 *
 *  @param[in] enable Pass 1 to enable the bandwith limit, or zero otherwise.
 *
 *  @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetBWLimit(HANDLE handle, U32 channel, U32 enable);

/**
 * @brief Control power to ADC devices
 *
 * @param[in] handle Handle to board
 *
 * @param[in] sleepState Specifies the power state of the ADC converters. This
 * paramter can be one of \ref ALAZAR_POWER_STATES.
 */
RETURN_CODE EXPORT AlazarSleepDevice(HANDLE handle, U32 sleepState);

/**
 *  @brief AutoDMA acquisition modes. See AlazarBeforeAsyncRead().
 */
enum ALAZAR_ADMA_MODES {
    /// Acquire multiple records: one per trigger event. Each record may include
    /// pre-and post-trigger samples, and a record header that includes its
    /// trigger timestamp. If a board has on-board memory and sample interleave
    /// is not enabled, each buffer will contain samples organized as follows:
    /// `R1A, R1B, R2A, R2B ...`
    ///
    /// If a board does not have on-board memory, or sample interleave is
    /// enabled, the buffer will contain samples organized as follows:
    /// `R1[AB...], R2[AB...] ...`
    ADMA_TRADITIONAL_MODE = 0x00000000,

    /// Acquire a single, gapless record spanning multiple buffers. Do not wait
    /// for trigger event before starting the acquisition.
    ///
    /// If a board has on-board memory and sample interleave is not enabled,
    /// each buffer will contain samples organized as follows: `R1A, R1B`.
    ///
    /// If a board does not have on-board memory, or sample interleave is
    /// enabled, the buffer will contain samples organized as follows:
    /// `R1[AB...]`
    ADMA_CONTINUOUS_MODE = 0x00000100,

    /// Acquire multiple records: one per trigger event. Each record contains
    /// only post- trigger samples.
    ///
    /// If a board has on-board memory and sample interleave is not enabled,
    /// each buffer will contain samples organized as follows: `R1A, R2A, ...
    /// R1B, R2B ...`
    ///
    /// If a board does not have on-board memory, or sample interleave is
    /// enabled, the buffer will contain samples organized as follows:
    /// `R1[AB...], R2[AB...] ...`
    ADMA_NPT = 0x00000200,

    /// Acquire a single, gapless record spanning multiple buffers. Wait for a
    /// trigger event before starting the acquisition.
    ///
    /// If a board has on-board memory and sample interleave is not enabled,
    /// each buffer will contain samples organized as follows: `R1A, R1B`.
    ///
    /// If a board does not have on-board memory, or sample interleave is
    /// enabled, the buffer will contain samples organized as follows:
    /// `R1[AB...]`
    ADMA_TRIGGERED_STREAMING = 0x00000400,
};

/**
 *  @brief AutoDMA acquisition options. See AlazarBeforeAsyncRead().
 */
enum ALAZAR_ADMA_FLAGS {
    /// The acquisition only starts when AlazarStartCapture() is called if this
    /// flag is set. Otherwise, it starts before the current function returns.
    ADMA_EXTERNAL_STARTCAPTURE = 0x00000001,

    /// If this flag is set, precede each record in each buffer
    /// with a 16-byte header that includes the record’s trigger timestamp.
    ///
    /// Note that this flag can only be used in “traditional” AutoDMA mode.
    /// Record headers are not available in NPT, streaming, or triggered
    /// streaming modes.
    ADMA_ENABLE_RECORD_HEADERS = 0x00000008,

    /// @cond INTERNAL_DECLARATIONS
    ADMA_SINGLE_DMA_CHANNEL = 0x00000010,
    /// @endcond

    /// If this flag is set, the API will allocate and manage a list of DMA
    /// buffers. This flag may be used by LabVIEW, and in other high-level
    /// development environments, where it may be more convenient for the
    /// application to let the API manage a list of DMA buffers, and to receive
    /// a copy of data in an application buffer. When this flag is set, the
    /// application must call AlazarWaitNextAsyncBufferComplete() to wait for a
    /// buffer to complete and receive a copy of the data. The application can
    /// specify the number of DMA buffers for the API to allocate by calling
    /// AlazarSetParameter with the parameter \ref SETGET_ASYNC_BUFFCOUNT before
    /// calling AlazarBeforeAsyncRead.
    ADMA_ALLOC_BUFFERS = 0x00000020,

    /// Enable the board to data from its on-FPGA FIFO rather than from on-board
    /// memory. When the flag is set, each buffer contains data organized as
    /// follows: `R0[ABAB...], R1[ABAB...], R2[ABAB] ....` That is, each sample
    /// from CH A is followed by a sample from CH B.
    ///
    /// When this flag is not set, each record in a buffer contains a contiguous
    /// array of samples for CH A followed by a contiguous array of samples for
    /// CH B, where the record arrangement depends on the acquisition mode. Note
    /// that this flag must be set if your board does not have on-board memory.
    /// For example, an ATS9462- FIFO requires this flag. Also note that this
    /// flag must not be set if your board has on-board memory.
    ADMA_FIFO_ONLY_STREAMING = 0x00000800,

    /// Enable a board to interleave samples from both digitizer channels in
    /// dual-channel acquisition mode. This results in higher data transfer
    /// rates on boards that support this option.
    ///
    /// Note that this flag has no effect in single channel mode, and is
    /// supported by only PCIe digitizers (except the ATS9462).
    ///
    /// When the flag is set, each buffer contains data organized as follows:
    /// `R0[ABAB...], R1[ABAB...], R2[ABAB] ....` That is, each sample from CH A
    /// is followed by a sample from CH B.
    ///
    /// When this flag is not set, each record in a buffer contains a contiguous
    /// array of samples for CH A followed by a contiguous array of samples for
    /// CH B, where the record arrangement depends on the acquisition mode.
    ADMA_INTERLEAVE_SAMPLES = 0x00001000,

    /// Enable the API to process each buffer So that the sample data in a
    /// buffer is Always arranged as in NPT mode: `R0A, R1A, R2A, ... RB0, R1B,
    /// R2B`.
    ///
    /// If this flag is not set, the data Arrangement in a buffer depends on The
    /// acquisition mode.
    ///
    /// LabVIEW and other higher-level Applications may use this flag to
    /// Simplify data processing since all data Buffers will have the same
    /// Arrangement independent of the Acquisition mode.
    ///
    /// Note that the \ref ADMA_ALLOC_BUFFERS flag Must also be set to use this
    /// option.
    ADMA_GET_PROCESSED_DATA = 0x00002000,

    /// Activates the DSP mode that must be used for using the on-FPGA DSP
    /// modules such as the on-FPGA FFT.
    ADMA_DSP = 0x00004000,

    /// Activate record footers, that are appended to each acquired record.
    /// Please note that this feature is not available on all boards, and can
    /// only be activated in NPT mode.
    ADMA_ENABLE_RECORD_FOOTERS = 0x00010000,

    /// @cond INTERNAL_DECLARATIONS
    /// Activates a dual-buffer mode which outputs data from the two acquired
    /// channels in two consecutive DMA buffers. This feature is only available
    /// for dual-channel acquisitions, and for specific boards. Consult
    /// board-specific documentation for more details.
    ADMA_DUAL_BUFFER_MODE = 0x00020000,
    /// @endcond
};

/**
 *  @brief Configure a board to make an asynchronous AutoDMA acquisition.
 *
 *  In non-DSP mode, when record headers are not enabled, the total number of
 *  bytes per AutoDMA buffer is given by
 *
 *      bytesPerBuffer = bytesPerSample * samplesPerRecord * recordsPerBuffer;
 *
 *  When record headers are enabled, the formula changes to:
 *
 *      bytesPerBuffer = (16 + bytesPerSample * samplesPerRecord) *
 *                       recordsPerBuffer;
 *
 *  For best performance, AutoDMA parameters should be selected so that the
 *  total number of bytes per buffer is greater than about 1 MB. This allows for
 *  relatively long DMA transfer times compared to the time required to prepare
 *  a buffer for DMA transfer and re-arm the DMA engines.
 *
 *  ATS460, ATS660 and ATS860 digitizer boards require that AutoDMA parameters
 *  be selected so that the total number of bytes per buffer is less than 4 MB.
 *  Other boards require that the total number of bytes per buffer be less than
 *  64 MB. It is however recommended to keep the DMA buffer size below 16 MB for
 *  all boards.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] channelSelect Select the channel(s) to control. This can be one
 *  or more of the channels of \ref ALAZAR_CHANNELS, assembled with the OR
 *  bitwise operator.
 *
 *  @param[in] transferOffset Specify the first sample from each on-board record
 *  to transfer from on-board to host memory. This value is a sample relative to
 *  the trigger position in an on-board record.
 *
 *  @param[in] transferLength Specify the number of samples from each record
 *  to transfer from on-board to host memory. In DSP-mode, it takes the number
 *  of bytes instead of samples. See remarks.
 *
 *  @param[in] recordsPerBuffer The number of records in each buffer. See
 *  remarks.
 *
 *  @param[in] recordsPerAcquisition The number of records to acquire during one
 *  acquisition. Set this value to `0x7FFFFFFF` to acquire indefinitely until
 *  the acquisition is aborted. This parameter is ignored in Triggered Streaming
 *  and Continuous Streaming modes. See remarks.
 *
 *  @param[in] flags Specifies AutoDMA mode and option. Must be one element of
 *  \ref ALAZAR_ADMA_MODES combined with zero or more element(s) of \ref
 *  ALAZAR_ADMA_FLAGS using the bitwise OR operator.
 *
 *  @copydoc default_return_values
 *
 *  @remark \c transferLength must meet certain alignment criteria which depend
 *  on the board model and the acquisition type. Please refer to board-specific
 *  documentation for more information.
 *
 *  @remark \c recordsPerBuffer must be set to 1 in continuous streaming and
 *  triggered streaming AutoDMA modes.
 *
 *  @remark \c recordsPerAcquisition must be `0x7FFFFFFF` in Continuous
 *  Streaming and Triggered Streaming modes. The acquisition runs continuously
 *  until AlazarAbortAsyncRead() is called. In other modes, it must be either:
 *   - A multiple of \c recordsPerBuffer
 *   - `0x7FFFFFFF` to indicate that the acquisition should continue
 *     indefinitely.
 */
RETURN_CODE EXPORT AlazarBeforeAsyncRead(HANDLE handle, U32 channelSelect,
                                         long transferOffset,
                                         U32 transferLength,
                                         U32 recordsPerBuffer,
                                         U32 recordsPerAcquisition, U32 flags);

/**
 *  @brief Aborts a dual-port acquisition, and any in-process DMA transfers.
 *
 *  @param[in] handle Handle to board
 *
 *  @remark If you have started an acquisition and/or posted DMA buffers to a
 *  board, you *must* call AlazarAbortAsyncRead() before your application exits.
 *  If you do not, when your program exists, Microsoft Windows may stop with a
 *  blue screen error number `0x000000CB (DRIVER_LEFT_LOCKED_PAGES_IN_PROCESS)`.
 *  Linux may leak the memory used by the DMA buffers.
 *
 *  @copydoc default_return_values
 *
 *  @note This function is part of the dual-port API. It should be used only in
 *  this context. To abort single-port acquisitions using, see
 *  AlazarAbortCapture().
 */
RETURN_CODE EXPORT AlazarAbortAsyncRead(HANDLE handle);

/**
 * @brief Posts a DMA buffer to a board.
 *
 * This function adds a DMA buffer to the end of a list of buffers available to
 * be filled by the board. Use AlazarWaitAsyncBufferComplete() to determine if
 * the board has received sufficient trigger events to fill this buffer.
 *
 * @param[in] handle Handle to board
 *
 * @param[in] buffer Pointer to buffer that will eventually receive data from
 * the digitizer board.
 *
 * @param[in] bufferLength_bytes The length of the buffer in bytes.
 *
 * @copydoc default_return_values
 *
 * @remark You must call AlazarBeforeAsyncRead() before calling
 * AlazarPostAsyncBuffer().
 *
 * @warning You must call AlazarAbortAsyncRead() before your application exits
 * if you have called AlazarPostAsyncBuffer() and buffers are pending when your
 * application exits.
 *
 * @remark The \c bufferLength_bytes parameter must be equal to the product of
 * the number of bytes per record, the number of records per buffer and the
 * number of enabled channels. If record headers are enabled, the number of
 * bytes per record must include the size of the record header (16 bytes).
 */
RETURN_CODE EXPORT AlazarPostAsyncBuffer(HANDLE handle, void *buffer,
                                         U32 bufferLength_bytes);

/**
 *  @brief This function returns when a board has received sufficient triggers
 *  to fill the specified buffer, or when the timeout internal elapses.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[out] buffer Pointer to a buffer to receive sample data form the
 *  digitizer board
 *
 *  @param[in] timeout_ms The time to wait for the buffer to be filled, in
 *  milliseconds.
 *
 *  @returns If the board receives sufficien ttrigger events to fill this buffer
 *  before the timeout interval elapses, the function returns #ApiSuccess.
 *
 *  @returns If the timeout interval elapses before the board receives
 *  sufficient trigger events to fill the buffer, the function returns
 *  #ApiWaitTimeout.
 *
 *  @returns If the board overflows its on-board memory, the function returns
 *  #ApiBufferOverflow. This happens if the rate at which data is acquired is
 *  fater than the rate at which data is being transferred from on-board memory
 *  to host memory across the host bus interface.
 *
 *  @returns If this buffer was not found in the list of buffers available to be
 *  filled by the board, the function returns #ApiBufferNotReady.
 *
 *  @returns If this buffer is not the buffer at the head of the list of buffers
 *  to be filled by the board, this returns #ApiDmaInProgress.
 *
 *  @returns If the function fails for some other reason, it returns an error
 *  code that indicates the reason that it failed. See \ref RETURN_CODE for more
 *  information.
 *
 *  @remark You must call AlazarBeforeAsyncRead() and AlazarPostAsyncBuffer()
 *  before calling AlazarWaitAsyncBufferComplete().
 *
 *  @warning You must call AlazarAbortAsyncRead() before your application exits
 *  if your have called AlazarPostAsyncBuffer() and buffers are pending.
 *
 *  Each call to AlazarPostAsyncBuffer() adds a buffer to the end of a
 *  list fo buffers to be filled by the board. AlazarWaitAsyncBufferComplete()
 *  expects to wait on the buffer at the head of this list. As a result, you
 *  must wait for buffers in the same order than they were posted.
 *
 *  When AlazarWaitAsyncBufferComplete() returns #ApiSuccess, the
 *  buffer is removed from the list of buffers to be filled by the board.
 *
 *  The arrangement of sample data in each buffer depends on the AutoDMA
 *  mode specified in the call to AlazarBeforeAsyncRead().
 */
RETURN_CODE EXPORT AlazarWaitAsyncBufferComplete(HANDLE handle, void *buffer,
                                                 U32 timeout_ms);

/**
 *  @brief This function returns when the board has received sufficient trigger
 *  events to fill the buffer, or the timeout interval has elapsed.
 *
 *  To use this function, AlazarBeforeAsyncRead() must be called with \ref
 *  ADMA_ALLOC_BUFFERS.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[out] buffer Pointer to a buffer to receive sample data from the
 *  digitizer board.
 *
 *  @param[in] bytesToCopy The number of bytes to copy into the buffer
 *
 *  @param[in] timeout_ms The time to wait for the buffer to buffer to be
 *  filled, in milliseconds.
 *
 *  @returns If the board receives sufficient trigger events to fill the next
 *  available buffer before the timeout interval elapses, and the buffer is not
 *  the last buffer in the acquisition, the function returns #ApiSuccess.
 *
 *  @returns If the board receives sufficient trigger events to fill the next
 * available
 *  buffer before the timeout interval elapses, and the buffer is the last
 *  buffer in the acquisition, the function returns #ApiTransferComplete.
 *
 *  @returns If the timeout interval elapses before the board receives
 *  sufficient trigger events to fill the next available buffer, the function
 *  returns #ApiWaitTimeout.
 *
 *  @returns If the board overflows its on-board memory, the function returns
 *  #ApiBufferOverflow. The board may overflow its on-board memory because
 *  the rate at which it is acquiring data is faster than the rate at which the
 *  data is being transferred from on-board memory to host memory across the
 *  host bus interface (PCI or PCIe). If this is the case, try reducing the
 *  sample rate, number of enabled channels, or amount of time spent processing
 *  each buffer.
 *
 *  @returns If the function fails for some other reason, it returns an error
 *  code that indicates the reason that it failed.
 *
 *  You must call AlazarBeforeAsyncRead() with the \ref ADMA_GET_PROCESSED_DATA
 *  flag before calling AlazarWaitNextAsyncBufferComplete().
 *
 *  To discard buffers, set the bytesToCopy parameter to zero. This will cause
 *  AlazarWaitNextAsyncBufferComplete() to wait for a buffer to complete, but
 * not
 *  copy any data into the application buffer.
 *
 *  To enable disk streaming using high-performance disk I/O functions, call
 *  AlazarCreateStreamFile() before calling AlazarWaitNextAsyncBufferComplete().
 *  For best performance, set the bytesToCopy parameter to zero so that data is
 *  streamed to disk without making any intermediate copies in memory.
 *
 *  If AlazarBeforeAsyncRead() is called with the \ref ADMA_GET_PROCESSED_DATA
 *  flag, AlazarWaitNextAsyncBuferComplete() will process buffers so that the
 *  data always appears in NPT format: R1A, R2A, ... RnA, R1B, R2B, ... RnB.
 * This
 *  may simply you application, but it comes at the expense of added processing
 *  time for each buffer. If AlazarBeforeAsyncRead() is not called with the
 *  called with the \ref ADMA_GET_PROCESSED_DATA flag set, then arrangement of
 *  sample data in a buffer depends on the AutoDMA mode.
 */
RETURN_CODE EXPORT AlazarWaitNextAsyncBufferComplete(HANDLE handle,
                                                     void *buffer,
                                                     U32 bytesToCopy,
                                                     U32 timeout_ms);

/**
 *  @class create_stream_file
 *
 *  @brief Creates a binary data file for this board, and enables saving AutoDMA
 *  data from thie board to disk.
 *
 *  If possible, select AlazarBeforeAsyncRead() parameters that result in DMA
 *  buffers whose length in bytes is evenly divisible into sectors of the volume
 *  selected by \c filePath. If the DMA buffer length is evenly divisible into
 *  records, AlazarCreateStreamFile() disables file caching to obtain the
 *  highest possible sequential write performance.
 *
 *  An AutoDMA buffer is saved to disk when an application calls
 *  AlazarWaitNextAsyncBufferComplete(). For best performance, set the \c
 *  bytesToCopy parameter in AlazarWaitNextAsyncBufferComplete() to zero so that
 *  data is written to disk without copying it to the user-supplied buffer.
 *
 *  This function must be called after AlazarBeforeAsyncRead() and before
 *  AlazarStartCapture(). File streaming is only active for the acquisition that
 *  is about to start when this function is called. You should call this
 *  function again for each acquisition with which you want file streaming.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] filePath Pointer to a NULL-terminated string that specifies the
 *  name of the file.
 *
 *  @copydoc default_return_values
 */

#ifdef WIN32
/**
 * @copydoc create_stream_file
 */
RETURN_CODE EXPORT AlazarCreateStreamFileA(HANDLE handle, const char *filePath);

/**
 * @copydoc create_stream_file
 */
RETURN_CODE EXPORT AlazarCreateStreamFileW(HANDLE handle,
                                           const wchar_t *filePath);
#    ifdef UNICODE
/**
 * @copydoc create_stream_file
 */
#        define AlazarCreateStreamFile AlazarCreateStreamFileW
#    else // UNICODE
/**
 * @copydoc create_stream_file
 */
#        define AlazarCreateStreamFile AlazarCreateStreamFileA
#    endif // UNICODE
#else      // WIN32
RETURN_CODE EXPORT
    /**
     * @copydoc create_stream_file
     */
    AlazarCreateStreamFile(HANDLE handle, const char *filePath);
#endif

/**
 * @brief Resets the record timestamp counter
 *
 * @param[in] handle Handle to board
 *
 * @param[in] option Record timestamp reset option. Can be one of \ref
 * ALAZAR_TIMESTAMP_RESET_OPTIONS.
 *
 * @copydoc default_return_values
 *
 * @remark This function is not supported by ATS310, ATS330 and ATS850
 */
RETURN_CODE EXPORT AlazarResetTimeStamp(HANDLE handle, U32 option);

/**
 * @cond INTERNAL_DECLARATIONS
 */
RETURN_CODE EXPORT AlazarReadRegister(HANDLE handle, U32 offset, U32 *retVal,
                                      U32 pswrd);
RETURN_CODE EXPORT AlazarWriteRegister(HANDLE handle, U32 offset, U32 Val,
                                       U32 pswrd);

RETURN_CODE EXPORT ReadC(HANDLE handle, U8 *DmaBuffer, U32 SizeToRead,
                         U32 LocalAddress);

void EXPORT WriteC(HANDLE handle, U8 *DmaBuffer, U32 SizeToRead,
                   U32 LocalAddress);

RETURN_CODE EXPORT AlazarGetTriggerAddressA(HANDLE handle, U32 Record,
                                            U32 *TriggerAddress,
                                            U32 *TimeStampHighPart,
                                            U32 *TimeStampLowPart);
RETURN_CODE EXPORT AlazarGetTriggerAddressB(HANDLE handle, U32 Record,
                                            U32 *TriggerAddress,
                                            U32 *TimeStampHighPart,
                                            U32 *TimeStampLowPart);

#ifndef PLX_DRIVER

RETURN_CODE EXPORT ATS9462FlashSectorPageRead(HANDLE handle, U32 address,
                                              U16 *PageBuff);
RETURN_CODE EXPORT ATS9462PageWriteToFlash(HANDLE handle, U32 address,
                                           U16 *PageBuff);
RETURN_CODE EXPORT ATS9462FlashSectorErase(HANDLE handle, int sectorNum);
RETURN_CODE EXPORT ATS9462FlashChipErase(HANDLE h);
RETURN_CODE EXPORT SetControlCommand(HANDLE handle, int cmd);

#endif // PLX_DRIVER

RETURN_CODE EXPORT AlazarDACSetting(HANDLE handle, U32 SetGet,
                                    U32 OriginalOrModified, U8 Channel,
                                    U32 DACNAME, U32 Coupling, U32 InputRange,
                                    U32 Impedance, U32 *getVal, U32 setVal,
                                    U32 *error);

RETURN_CODE EXPORT AlazarWrite(HANDLE handle, void *buffer, long bufLen,
                               int DmaChannel, U32 firstPoint, U32 startAddress,
                               U32 endAddress, BOOL waitTillEnd, U32 *error);
/**
 * @endcond
 */

/**
 *  @brief Configures the AUX I/O connector as an input or output signal.
 *
 *  The AUX I/O connector generates TTL level signals when configured as
 *  an output, and expects TLL level signals when configured as an input.
 *
 *  AUX I/O output signals may be limited by the bandwidth of the AUX
 *  output drivers.
 *
 *  @remark The ATS9440 has two AUX I/O connectors: AUX 1 and AUX 2. AUX 1 is
 *  configured by firmware as a trigger output signal, while AUX 2 is configured
 *  by software using AlazarConfigureAuxIO(). A firmware update is required to
 *  change the operation of AUX 1.
 *
 *  @remark ATS9625 and ATS9626 have two AUX I/O connectors; AUX 1 and AUX 2.
 *  AUX 1 is configured by software using AlazarConfigureAuxIO(), while AUX 2 is
 *  configured by default as a trigger output signal. A custom user-programmable
 *  FGPA can control the operation of AUX 2 as required by the FPGA designer.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] mode The AUX I/O mode. Can be selected from \ref
 *  ALAZAR_AUX_IO_MODES. If an output mode is selected, the parameter may be
 *  OR'ed with \ref AUX_OUT_TRIGGER_ENABLE to enable the board to use software
 *  trigger enable. When this flag is set, the board will wait for software to
 *  call AlazarForceTriggerEnable() to generate a trigger enable event; then
 *  wait for sufficient trigger events to capture the records in an AutoDMA
 *  buffer; then wait for the next trigger enable event and repeat.
 *
 *  @param[in] parameter The meaning of this value varies depending on \c mode.
 *  See \ref ALAZAR_AUX_IO_MODES for more details.
 *
 *  @copydoc default_return_values
 *
 */
RETURN_CODE EXPORT AlazarConfigureAuxIO(HANDLE handle, U32 mode, U32 parameter);

/**
 *  @brief Converts a numerical return code to a \c NULL terminated string.
 *
 *  @param[in] retCode Return code from an AlazarTech API function
 *
 *  @returns A string containing the identifier name of the error code
 *
 *  @remark It is often easier to work with a descriptive error name than an
 * error number.
 */
const char EXPORT *AlazarErrorToText(RETURN_CODE retCode);

/**
 *  @brief Makes the digitizer sub-sample post trigger data in arbitrary,
 *  non-uniform intervals.
 *
 *  The application specifies which sample clock edges after a trigger event the
 *  digitizer should use to generate sample points, and which sample clock edges
 *  the digitizer should ignore.
 *
 *  To enable data skipping, first create a bitmap in memory that
 *  specifies which sample clock edges should generate a sample point, and which
 *  sample clock edges should be ignored.
 *
 *  - 1’s in the bitmap specify the clock edges that should generate a sample
 *  point. The total number of 1’s in the bitmap must be equal to the number of
 *  post-trigger samples per record specified in the call to
 *  AlazarSetRecordSize().
 *
 *  - 0’s in the bitmap specify the clock edges that should not be used to
 *  generate a sample point.
 *
 *  - The total total number of bits in the bitmap is equal to the number of
 *  sample clocks in one record.
 *
 *  For example, to receive 16 samples from 32 sample clocks where every other
 *  sample clock is ignored, create a bitmap of 32 bits with values `{ 1 0 1 0 1
 *  0 ... 1 0 }`, or `{ 0x5555, 0x5555 }`. Note that 16 of the 32 bits are 1’s.
 *
 *  And to receive 24 samples from 96 sample clocks where data from every 3 of 4
 *  samples clocks is ignored, create a bitmap of 96 bits with values `{ 1 0 0 0
 *  1 0 0 0 1 0 0 0 ... 1 0 0 0 }`, or in `{ 0x1111, 0x1111, 0x1111, 0x1111,
 *  0x1111, 0x1111 }`. Note that 24 of the 96 bits are 1’s.
 *
 *  After creating a bitmap, call AlazarConfigureSampleSkipping() with:
 *  - Mode equal to \ref SSM_ENABLE
 *  - SampleClocksPerRecord equal to the total number of sample clocks per
 *    record.
 *  - pSampleSkipBitmap with the address of the U16 array.
 *
 *  To disable data skipping, call AlazarConfigureSampleSkipping with Mode equal
 *  to \ref SSM_DISABLE. The SampleClocksPerRecord and pSampleSkipBitmap
 *  parameters are ignored.
 *
 *  Note that data skipping currently is supported by the ATS9371, ATS9373,
 *  ATS9360, ATS9350, ATS9351, ATS9352 and ATS9440.
 *  For ATS9440, data skipping only works with post-trigger data acquired
 *  at 125 MSPS or 100 MSPS.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] mode The data skipping mode. 0 means disable sample skipping and
 *  1 means enable sample skipping.
 *
 *  @param[in] sampleClocksPerRecord The number of sample clocks per record.
 *  This value cannot exceed 65536.
 *
 *  @param[in] sampleSkipBitmap An array of bits that specify which sample
 *  clock edges should be used to capture a sample point (value = 1) and which
 *  should be ignored (value = 0).
 *
 *  @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarConfigureSampleSkipping(HANDLE handle, U32 mode,
                                                 U32 sampleClocksPerRecord,
                                                 U16 *sampleSkipBitmap);

/**
 *  @brief Reads the content of a user-programmable FPGA register
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] offset Register offset
 *
 *  @param[out] value Address of a variable to receive the register's value
 *
 *  @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarCoprocessorRegisterRead(HANDLE handle, U32 offset,
                                                 U32 *value);

/**
 *  @brief Writes a value to a user-programmable coprocessor FPGA register
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] offset Register offset
 *
 *  @param[in] value Value to write
 *
 *  @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarCoprocessorRegisterWrite(HANDLE handle, U32 offset,
                                                  U32 value);

/**
 *  @class coprocessor_download
 *
 *  @brief Downloads a FPGA image in RBF (raw binary file) format to the
 *  coprocessor FPGA.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] fileName Path to RBF file
 *
 *  @param[in] options Download options chosen from \ref
 *  ALAZAR_COPROCESSOR_DOWNLOAD_OPTIONS
 *
 *  @copydoc default_return_values
 */

/**
 *  @copydoc coprocessor_download
 */
RETURN_CODE EXPORT AlazarCoprocessorDownloadA(HANDLE handle, char *fileName,
                                              U32 options);

#ifdef WIN32

/**
 *  @copydoc coprocessor_download
 */
RETURN_CODE EXPORT AlazarCoprocessorDownloadW(HANDLE handle, wchar_t *fileName,
                                              U32 options);

#endif

/**
 *  @cond INTERNAL_DECLARATIONS
 */
#ifdef WIN32
#    ifdef UNICODE
/**
 *  @copydoc coprocessor_download
 */
#        define AlazarCoprocessorDownload AlazarCoprocessorDownloadW
#    else
/**
 *  @copydoc coprocessor_download
 */
#        define AlazarCoprocessorDownload AlazarCoprocessorDownloadA
#    endif
#else
/**
 *  @copydoc coprocessor_download
 */
#    define AlazarCoprocessorDownload AlazarCoprocessorDownloadA
#endif
/**
 *  @endcond
 */

/**
 *  @cond INTERNAL_DECLARATIONS
 */
RETURN_CODE AlazarCoprocessorDownloadBuffer(HANDLE handle, U8 *pbBuffer,
                                            U32 uBytesToWrite, U32 options);
#define CPFREG_SIGNATURE 0
#define CPFREG_REVISION 1
#define CPFREG_VERSION 2
#define CPFREG_STATUS 3
/**
 *  @endcond
 */

/**
 *  @brief Configures a digitizer to co-add ADC samples from a specified number
 *  of records in an accumulator record, and transfer accumulator records rather
 *  than the ADC sample values.
 *
 *  When FPGA record averaging is enabled, the digitizer transfers one
 *  accumulator record to host memory after \c recordsPerAverage trigger events
 *  have been captured.
 *
 *  Each accumulator record has interleaved samples from CH A and CH B. FPGA
 *  accumulators are 32-bit wide, so each accumulator value occupies 4 bytes in
 *  a buffer. The digitizer transfers multi-byte values in little-endian byte
 *  order.
 *
 *  CH A and CH B accumulator records are always transferred to host memory. As
 *  a result, the number of bytes per accumulator record is given by:
 *
 *      samplesPerRecord * 2 (channels) * 4 (bytes per accumulator sample)
 *
 *  The maximum value of \c recordsPerAverage for 8-bit digitizers is 16777215
 *
 *  Note that \c recordsPerAverage does not have to be equal to the number of
 *  records per buffer in AutoDMA mode.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] mode Averaging mode. Should be one element of \ref
 *  ALAZAR_CRA_MODES.
 *
 *  @param[in] samplesPerRecord The number of ADC samples per accumulator
 * record.
 *
 *  @param[in] recordsPerAverage The number of records to accumulate per
 * average.
 *
 *  @param[in] options The averaging options. Can be one of \ref
 *  ALAZAR_CRA_OPTIONS.
 *
 *  @copydoc default_return_values
 *
 *  @remark FPGA record averaging is currently supported on the following
 *  digitizers:
 *   - ATS9870 with FPGA version 180.0 and above, and driver version 5.9.8
 *     and above
 *   - AXI9870 with FPGA version 180.0 and above, and driver version 5.9.23
 *     and above
 *
 *  @note This function is part of the dual-port API. It should be used only in
 *  this context. To abort single-port acquisitions using, see
 *  AlazarAbortCapture().
 */
RETURN_CODE EXPORT AlazarConfigureRecordAverage(HANDLE handle, U32 mode,
                                                U32 samplesPerRecord,
                                                U32 recordsPerAverage,
                                                U32 options);

/**
 *  @brief AlazarConfigureRecordAverage() modes
 */
enum ALAZAR_CRA_MODES {
    CRA_MODE_DISABLE = 0,        ///< Disables record average
    CRA_MODE_ENABLE_FPGA_AVE = 1 ///< Enables record average
};

/**
 * @brief AlazarConfigureRecordAverage() options
 */
enum ALAZAR_CRA_OPTIONS {
    CRA_OPTION_UNSIGNED = (0U << 1), ///< Unsigned data
    CRA_OPTION_SIGNED = (1U << 1)    ///< Signed data
};

/**
 * @brief Allocates a buffer for DMA transfer for an 8-bit digitizer
 *
 * @param[in] handle Handle to board
 *
 * @param[in] sampleCount Buffer size in samples
 *
 * @returns If the function is successful, it returns the base address of a
 * page-aligned buffer in the virtual address space of the calling process. If
 * it fails, it returns NULL.
 *
 * @remark The buffer must be freed using AlazarFreeBufferU8()
 */
EXPORT U8 *AlazarAllocBufferU8(HANDLE handle, U32 sampleCount);

/**
 * @brief Frees a buffer allocated with AlazarAllocBufferU8()
 *
 * @param[in] handle Handle to board
 *
 * @param[in] buffer Base address of the buffer to free
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarFreeBufferU8(HANDLE handle, U8 *buffer);

/**
 * @brief Allocates a buffer for DMA transfer for an 16-bit digitizer
 *
 * @param[in] handle Handle to board
 *
 * @param[in] sampleCount Buffer size in samples
 *
 * @returns If the function is successful, it returns the base address of a
 * page-aligned buffer in the virtual address space of the calling process. If
 * it fails, it returns NULL.
 *
 * @remark The buffer must be freed using AlazarFreeBufferU16()
 */
EXPORT U16 *AlazarAllocBufferU16(HANDLE handle, U32 sampleCount);

/**
 * @brief Frees a buffer allocated with AlazarAllocBufferU16()
 *
 * @param[in] handle Handle to board
 *
 * @param[in] buffer Base address of the buffer to free
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarFreeBufferU16(HANDLE handle, U16 *buffer);

/**
 * @brief This function acts like AlazarAllocBufferU8() and additionally allows
 * allocation of
 * a buffer over 4GS for DMA transfer for an 8-bit digitizer
 *
 * @param[in] handle Handle to board
 *
 * @param[in] sampleCount Buffer size in samples
 *
 * @returns If the function is successful, it returns the base address of a
 * page-aligned buffer in the virtual address space of the calling process. If
 * it fails, it returns NULL.
 *
 * @remark The buffer must be freed using AlazarFreeBufferU8Ex()
 */
EXPORT U8 *AlazarAllocBufferU8Ex(HANDLE handle, U64 sampleCount);

/**
 * @brief Frees a buffer allocated with AlazarAllocBufferU8Ex()
 *
 * @param[in] handle Handle to board
 *
 * @param[in] buffer Base address of the buffer to free
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarFreeBufferU8Ex(HANDLE handle, U8 *buffer);

/**
 * @brief This function acts like AlazarAllocBufferU16() and additionally allows
 * allocation of
 * a buffer over 4GS for DMA transfer for an 16-bit digitizer
 *
 * @param[in] handle Handle to board
 *
 * @param[in] sampleCount Buffer size in samples
 *
 * @returns If the function is successful, it returns the base address of a
 * page-aligned buffer in the virtual address space of the calling process. If
 * it fails, it returns NULL.
 *
 * @remark The buffer must be freed using AlazarFreeBufferU16Ex()
 */
EXPORT U16 *AlazarAllocBufferU16Ex(HANDLE handle, U64 sampleCount);

/**
 * @brief Frees a buffer allocated with AlazarAllocBufferU16Ex()
 *
 * @param[in] handle Handle to board
 *
 * @param[in] buffer Base address of the buffer to free
 *
 * @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarFreeBufferU16Ex(HANDLE handle, U16 *buffer);

/**
 *  @brief Repurposes unused least significant bits in 12- and 14-bit boards
 *
 *  12- and 14-bit digitizers return 16-bit sample values per sample by default,
 *  with the actual sample codes stored in the most significant bits. By
 *  default, the least significant bits of each sample value are zero-filled.
 *  Use this option to use these otherwise unused bits as digital outputs.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] valueLsb0 Specifies the signal to output to the least significant
 *  bit of each sample value. Must be one of \ref ALAZAR_LSB.
 *
 *  @param[in] valueLsb1 Specifies the signal to output to the least significant
 *  bit of each sample value. Must be one of \ref ALAZAR_LSB.
 *
 *  @copydoc default_return_values
 *
 *  This feature is not available on all boards. See board-specific
 *  documentation for more information.
 */
RETURN_CODE EXPORT AlazarConfigureLSB(HANDLE handle, U32 valueLsb0,
                                      U32 valueLsb1);

/**
 *  @brief NPT Footer structure that can be retrieved using
 *  AlazarExtractNPTFooters().
 */
typedef struct _NPTFooter {
    U64 triggerTimestamp; ///< Timestamp of the trigger event in this
    /// acquisition.
    U32 recordNumber;  ///< Record number
    U32 frameCount;    ///< Frame count
    BOOL aux_in_state; ///< AUX I/O state received during the record's
                       /// acquisition
} NPTFooter;

/**
 *  @brief Extracts NPT footers from a buffer acquired during an FFT
 *  acquisition.
 *
 *  Before calling this function, it is important to make sure that the buffers
 *  have been acquired in NPT mode with the NPT footers active. In addition, the
 *  acquisition *must* have used on-FPGA FFT computation.
 *
 *  @param[in] buffer Base address of the DMA buffer to process
 *
 *  @param[in] recordSize_bytes Bytes per record in the DMA buffer passed as
 *  argument as returned by AlazarFFTSetup().
 *
 *  @param[in] bufferSize_bytes Bytes per buffer in the DMA buffer passed as
 *  argument
 *
 *  @param[out] footersArray Base address of an array of \ref NPTFooter
 *  structures which will be filled by this function
 *
 *  @param[in] numFootersToExtract Maximum numbers of footers to extract. This
 *  can be a number from zero to the number of records in the DMA buffer passed
 *  as argument.
 *
 *  @warning \c footersArray must contain at least \c numFootersToExtract
 *  elements.
 */
RETURN_CODE EXPORT AlazarExtractFFTNPTFooters(void *buffer,
                                              U32 recordSize_bytes,
                                              U32 bufferSize_bytes,
                                              NPTFooter *footersArray,
                                              U32 numFootersToExtract);

/**
 *  @brief Extracts NPT footers from a buffer acquired during a time-domain
 *  acquisition.
 *
 *  Before calling this function, it is important to make sure that the buffers
 *  have been acquired in NPT mode with the NPT footers active. In addition, the
 *  acquisition must *not* have used on-FPGA FFT computation.
 *
 *  @param[in] buffer Base address of the DMA buffer to process
 *
 *  @param[in] recordSize_bytes Bytes per record in the DMA buffer passed as
 *  argument
 *
 *  @param[in] bufferSize_bytes Bytes per buffer in the DMA buffer passed as
 *  argument
 *
 *  @param[out] footersArray Base address of an array of \ref NPTFooter
 *  structures which will be filled by this function
 *
 *  @param[in] numFootersToExtract Maximum numbers of footers to extract. This
 *  can be a number from zero to the number of records in the DMA buffer passed
 *  as argument.
 *
 *  @warning \c footersArray must contain at least \c numFootersToExtract
 *  elements.
 */
RETURN_CODE EXPORT AlazarExtractTimeDomainNPTFooters(void *buffer,
                                                     U32 recordSize_bytes,
                                                     U32 bufferSize_bytes,
                                                     NPTFooter *footersArray,
                                                     U32 numFootersToExtract);

/**
 *  @brief Extracts NPT footers from a buffer that contains them
 *
 *  @warning This function has been deprecated in favor of
 *  AlazarExtractTimeDomainNPTFooters() and AlazarExtractFFTNPTFooters(). It is
 *  still usable, but only works on NPT footers acquired as part of an FFT
 *  acquisition.
 *
 *  Before calling this function, it is important to make sure that the buffers
 *  have been acquired in NPT mode with the NPT footers active.
 *
 *  @param[in] buffer Base address of the DMA buffer to process
 *
 *  @param[in] recordSize_bytes Bytes per record in the DMA buffer passed as
 *  argument
 *
 *  @param[in] bufferSize_bytes Bytes per buffer in the DMA buffer passed as
 *  argument
 *
 *  @param[out] footersArray Base address of an array of \ref NPTFooter
 *  structures which will be filled by this function
 *
 *  @param[in] numFootersToExtract Maximum numbers of footers to extract. This
 *  can be a number from zero to the number of records in the DMA buffer passed
 *  as argument.
 *
 *  @warning \c footersArray must contain at least \c numFootersToExtract
 *  elements.
 */
RETURN_CODE EXPORT AlazarExtractNPTFooters(void *buffer, U32 recordSize_bytes,
                                           U32 bufferSize_bytes,
                                           NPTFooter *footersArray,
                                           U32 numFootersToExtract);

/**
 * @cond INTERNAL_DECLARATIONS
 */
RETURN_CODE EXPORT AlazarDisableDSP(HANDLE boardHandle);
/**
 * @endcond
 */

/**
 *  @brief Enables or disables the *OCT ignore bad clock* mechanism.
 *
 *  This function must be called before an acquisition starts. It informs the
 *  digitizer about portions of time during which the external clock signal is
 *  valid, and others during which it is invalid and should be ignored.
 *
 *  "good" clock portions are durations of time during which the external clock
 *  signal is valid, i.e. within the board's specifications. "bad" clock
 *  portions are durations of time during which the clock signal is invalid.
 *
 *  When *OCT Ignore Bad Clock* is active, the digitizer must be set in external
 *  TTL trigger mode, and in external clock mode.
 *
 *  The external clock signal must be good when trigger events are received on
 *  the external trigger connector. The duration of time after the trigger event
 *  during which the clock signal is good is specified in \c
 *  goodClockDuration_seconds. After this good duration, the portion of time
 *  during which the clock may be bad is specified in \c
 *  badClockDuration_seconds.
 *
 *  The sum of \c goodClockDuration_seconds and \c badClockDuration_seconds must
 *  be less than the trigger cycle time. This means that the clock signal must
 *  be back to being good before the next trigger event.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] enable Enables (1) or disables (0) the feature
 *
 *  @param[in] goodClockDuration_seconds Good clock duration in seconds
 *
 *  @param[in] badClockDuration_seconds Bad clock duration in seconds
 *
 *  @param[out] triggerCycleTime_seconds Trigger cycle time measured by the
 *  board
 *
 *  @param[out] triggerPulseWidth_seconds Trigger pulse width measured by the
 *  board
 *
 *  @remark This function must be called prior to calling
 *  AlazarBeforeAsyncRead(). Trigger source must be set to \ref TRIG_EXTERNAL
 *  (AlazarSetTriggerOperation()). Trigger input range must be \ref ETR_TTL
 *  (AlazarSetExternalTrigger()). Clock source must be set to \ref
 *  FAST_EXTERNAL_CLOCK (AlazarSetCaptureClock()).
 */
RETURN_CODE EXPORT AlazarOCTIgnoreBadClock(HANDLE handle, U32 enable,
                                           double goodClockDuration_seconds,
                                           double badClockDuration_seconds,
                                           double *triggerCycleTime_seconds,
                                           double *triggerPulseWidth_seconds);

/**
 *  @brief Activates or deactivates the ADC background compensation.
 *
 *  @remark This feature does not exist on all boards. Please check
 *  board-specific information for more details.
 *
 *  @param[in] handle Handle to board
 *
 *  @param[in] active Determines whether this function activates or deactivates
 *  the ADC background compensation.
 *
 *  @copydoc default_return_values
 */
RETURN_CODE EXPORT AlazarSetADCBackgroundCompensation(HANDLE handle,
                                                      BOOL active);

#ifdef __cplusplus
}
#endif

#endif //_ALAZARAPI_H
