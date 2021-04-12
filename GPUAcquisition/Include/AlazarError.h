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
 *
 * This file defines all the error codes for the AlazarTech SDK
 */
#ifndef __ALAZARERROR_H
#define __ALAZARERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @cond INTERNAL_DECLARATIONS
 */
#define API_RETURN_CODE_STARTS 0x200 /* Starting return code */
/**
 *  @endcond
 */

/**
 *  @brief API functions return codes. Failure is #ApiSuccess
 */
enum RETURN_CODE {
    /// 512 - The operation completed without error
    ApiSuccess = API_RETURN_CODE_STARTS,

    /// The operation failed
    ApiFailed = 513,

    /// Access denied
    ApiAccessDenied = 514,

    /// Channel selection is unavailable
    ApiDmaChannelUnavailable = 515,

    /// Channel selection in invalid
    ApiDmaChannelInvalid = 516,

    /// Channel selection is invalid
    ApiDmaChannelTypeError = 517,

    /// A data transfer is in progress. This error code indicates that the
    /// current action cannot be performed while an acquisition is in progress.
    /// It also returned by AlazarPostAsyncBuffer() if this function is called
    /// with an invalid DMA buffer.
    ApiDmaInProgress = 518,

    /// DMA transfer is finished
    ApiDmaDone = 519,

    /// DMA transfer was paused
    ApiDmaPaused = 520,

    /// DMA transfer is not paused
    ApiDmaNotPaused = 521,

    /// A DMA command is invalid
    ApiDmaCommandInvalid = 522,

    /// One of the parameters of the function is NULL and should not be
    ApiNullParam = 531,

    /// This function is not supported by the API. Consult the manual for more
    /// information.
    ApiUnsupportedFunction = 533,

    /// Invalid PCI space
    ApiInvalidPciSpace = 534,

    /// Invalid IOP space
    ApiInvalidIopSpace = 535,

    /// Invalid size passed as argument to the function
    ApiInvalidSize = 536,

    /// Invalid address
    ApiInvalidAddress = 537,

    /// Invalid access type requested
    ApiInvalidAccessType = 538,

    /// Invalid index
    ApiInvalidIndex = 539,

    /// Invalid register
    ApiInvalidRegister = 543,

    /// Access for configuration failed
    ApiConfigAccessFailed = 550,

    /// Invalid device information
    ApiInvalidDeviceInfo = 551,

    /// No active driver for the board. Please ensure that a driver is installed
    ApiNoActiveDriver = 552,

    /// There were not enough system resources to complete this operation. The
    /// most common reason of this return code is using too many DMA buffers, or
    /// using DMA buffers that are too big. Please try reducing the number of
    /// buffers posted to the board at any time, and/or try reducing the DMA
    /// buffer sizes.
    ApiInsufficientResources = 553,

    /// The API has not been properly initialized for this function call. Please
    /// review one of the code samples from the ATS-SDK to confirm that API
    /// calls are made in the right order.
    ApiNotInitialized = 556,

    /// Power state requested is not valid
    ApiInvalidPowerState = 558,

    /// The operation cannot be completed because the device is powered down.
    /// For example, this error code is output if the computer enters
    /// hiberanation while an acquisition is running.
    ApiPowerDown = 559,

    /// The API call is not valid with this channel selection.
    ApiNotSupportThisChannel = 561,

    /// The function has requested no action to be taken
    ApiNoAction = 562,

    /// HotSwap is not supported
    ApiHSNotSupported = 563,

    /// Vital product data not enabled
    ApiVpdNotEnabled = 565,

    /// Offset argument is not valid
    ApiInvalidOffset = 567,

    /// Timeout on the PCI bus
    ApiPciTimeout = 569,

    /// Invalid handle passed as argument
    ApiInvalidHandle = 572,

    /// The buffer passed as argument is not ready to be called with this API.
    /// This error code is most often seen is the order of buffers posted to the
    /// board is not respected when querying them.
    ApiBufferNotReady = 573,

    /// Generic invalid parameter error. Check the function's documentation for
    /// more information about valid argument values.
    ApiInvalidData = 574,

    ApiDoNothing = 575,

    /// Unable to lock buffer and build SGL list
    ApiDmaSglBuildFailed = 576,

    /// Power management is not supported
    ApiPMNotSupported = 577,

    /// Invalid driver version
    ApiInvalidDriverVersion = 578,

    /// The operation did not finish during the timeout interval. try the
    /// operation again, or abort the acquisition.
    ApiWaitTimeout = 579,

    /// The operation was cancelled.
    ApiWaitCanceled = 580,

    /// The buffer used is too small. Try increasing the buffer size.
    ApiBufferTooSmall = 581,

    /// The board overflowed its internal (on-board) memory. Try reducing the
    /// sample rate, reducing the number of enabled channels. Also ensure that
    /// DMA buffer size is between 1 MB and 8 MB.
    ApiBufferOverflow = 582,

    /// The buffer passed as argument is not valid.
    ApiInvalidBuffer = 583,

    /// The number of reocrds per buffer passed as argument is invalid.
    ApiInvalidRecordsPerBuffer = 584,

    /// 585 - An asynchronous I/O operation was successfully started on the
    /// board. It will be completed when sufficient trigger events are supplied
    /// to the board to fill the buffer.
    ApiDmaPending,

    /// The buffer is too large for the driver or operating system to prepare
    /// for scatter-gather DMA transfer. Try reducing the size of each buffer,
    /// or reducing the number of buffers queued by the application.
    ApiLockAndProbePagesFailed = 586,

    /// This buffer is the last in the current acquisition
    ApiTransferComplete = 589,

    /// The on-board PLL circuit could not lock. If the acquisition used an
    /// internal sample clock, this might be a symptom of a hardware problem;
    /// contact AlazarTech. If the acquisition used an external 10 MHz PLL
    /// signal, please make sure that the signal is fed in properly.
    ApiPllNotLocked = 590,

    /// The requested acquisition is not possible with two channels. This can be
    /// due to the sample rate being too fast for DES boards, or to the number
    /// of samples per record being too large. Try reducing the number of
    /// samples per channel, or switching to single channel mode.
    ApiNotSupportedInDualChannelMode = 591,

    /// The requested acquisition is not possible with four channels. This can
    /// be due to the sample rate being too fast for DES boards, or to the
    /// number of samples per record being too large. Try reducing the number of
    /// samples per channel, or switching to single channel mode.
    ApiNotSupportedInQuadChannelMode = 592,

    /// A file read or write error occured.
    ApiFileIoError = 593,

    /// The requested ADC clock frequency is not supported.
    ApiInvalidClockFrequency = 594,

    /// Invalid skip table passed as argument
    ApiInvalidSkipTable = 595,

    /// This DSP module is not valid for the current operation.
    ApiInvalidDspModule = 596,

    /// Dual-edge sampling mode is only supported in signel-channel mode. Try
    /// disabling dual-edge sampling (lowering the sample rate if using internal
    /// clock), or selecting only one channel.
    ApiDESOnlySupportedInSingleChannelMode = 597,

    /// Successive API calls of the same acuqiisiton have received inconsistent
    /// acquisition channel masks.
    ApiInconsistentChannel = 598,

    /// DSP acquisition was run with a finite number of records per acqusiition.
    /// Set this value to inifinite.
    ApiDspFiniteRecordsPerAcquisition = 599,

    /// Not enough NPT footers in the buffer for extraction
    ApiNotEnoughNptFooters = 600,

    /// Invalid NPT footer found
    ApiInvalidNptFooter = 601,

    /// OCT ignore bad clock is not supported
    ApiOCTIgnoreBadClockNotSupported = 602,

    /// The requested number of records in a single-port acquisition exceeds the
    /// maximum supported by the digitizer. Use dual-ported AutoDMA to acquire
    /// more records per acquisition.
    ApiError1 = 603,

    /// The requested number of records in a single-port acquisition
    /// exceeds the maximum supported by the digitizer.
    ApiError2 = 604,

    /// No trigger is detected as part of the OCT ignore bad clock
    /// feature.
    ApiOCTNoTriggerDetected = 605,

    /// Trigger detected is too fast for the OCT ignore bad clock feature.
    ApiOCTTriggerTooFast = 606,

    /// There was a network-related issue. Make sure that the network connection
    /// and settings are correct.
    ApiNetworkError = 607,

    /// On-FPGA FFT cannot support FFT that large. Try reducing the FFT
    /// size, or querying the maximum FFT size with AlazarDSPGetInfo()
    ApiFftSizeTooLarge = 608,

    /// GPU returned an error. See log for more information
    ApiGPUError = 609,

    /// This board only supports this acquisition mode in FIFO only streaming
    /// mode. Please set the \ref ADMA_FIFO_ONLY_STREAMING flag in
    /// AlazarBeforeAsyncRead().
    ApiAcquisitionModeOnlySupportedInFifoStreaming = 610,

    /// This board does not support sample interleaving in traditional
    /// acquisition mode. Please refer to the SDK guide for more information.
    ApiInterleaveNotSupportedInTraditionalMode = 611,

    /// This board does not support record headers. Please refer to the SDK
    /// guide for more information.
    ApiRecordHeadersNotSupported = 612,

    /// This board does not support record footers. Please refer to the SDK
    /// guide for more information.
    ApiRecordFootersNotSupported = 613,

    /// @cond INTERNAL_DECLARATIONS
    /// --------- Add error codes above this comment --------

    ApiLastError,

    /// @{ Unused return code
    ApiDmaManReady = 523,
    ApiDmaManNotReady = 524,
    ApiDmaInvalidChannelPriority = 525,
    ApiDmaManCorrupted = 526,
    ApiDmaInvalidElementIndex = 527,
    ApiDmaNoMoreElements = 528,
    ApiDmaSglInvalid = 529,
    ApiDmaSglQueueFull = 530,
    ApiInvalidBusIndex = 532,
    ApiMuNotReady = 540,
    ApiMuFifoEmpty = 541,
    ApiMuFifoFull = 542,
    ApiDoorbellClearFailed = 544,
    ApiInvalidUserPin = 545,
    ApiInvalidUserState = 546,
    ApiEepromNotPresent = 547,
    ApiEepromTypeNotSupported = 548,
    ApiEepromBlank = 549,
    ApiObjectAlreadyAllocated = 554,
    ApiAlreadyInitialized = 555,
    ApiBadConfigRegEndianMode = 557,
    ApiFlybyNotSupported = 560,
    ApiVPDNotSupported = 564,
    ApiNoMoreCap = 566,
    ApiBadPinDirection = 568,
    ApiDmaChannelClosed = 570,
    ApiDmaChannelError = 571,
    ApiWaitAbandoned = 587,
    ApiWaitFailed = 588,
    /// @}

    /// @endcond
};

/// @cond INTERNAL_DECLARATIONS
typedef enum RETURN_CODE RETURN_CODE;
/// @endcond

#ifdef __cplusplus
}
#endif

#endif //__ALAZARERROR_H
