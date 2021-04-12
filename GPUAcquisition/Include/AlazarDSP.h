/**
 * @file AlazarDSP.h
 *
 * @author Alazar Technologies Inc
 *
 * @copyright Copyright (c) 2015 Alazar Technologies Inc. All Rights
 * Reserved.  Unpublished - rights reserved under the Copyright laws
 * of the United States And Canada.
 * This product contains confidential information and trade secrets
 * of Alazar Technologies Inc. Use, disclosure, or reproduction is
 * prohibited without the prior express written permission of Alazar
 * Technologies Inc
 *
 * Contains declarations of functions to do DSP processing on data
 * acquired from AlazarTech digitizers.
 */

#include "AlazarApi.h"
#include "AlazarCmd.h"

#ifndef ALAZARDSP_H
#    define ALAZARDSP_H

#    ifdef __cplusplus
extern "C" {
#    endif

/**
 * @brief Various types of window functions
 *
 * Used by AlazarDSPGenerateWindowFunction().
 */
enum DSP_WINDOW_ITEMS {
    DSP_WINDOW_NONE = 0,
    DSP_WINDOW_HANNING,
    DSP_WINDOW_HAMMING,
    DSP_WINDOW_BLACKMAN,
    DSP_WINDOW_BLACKMAN_HARRIS,
    DSP_WINDOW_BARTLETT,
    NUM_DSP_WINDOW_ITEMS
};

/**
 * @brief DSP module type
 *
 * Used by AlazarDSPGetInfo().
 */
enum DSP_MODULE_TYPE {
    DSP_MODULE_NONE = 0xFFFF, // avoids confusion with internal register.
    DSP_MODULE_FFT,           // FFT multisample
    DSP_MODULE_PCD,           // Pc Decoder Averager
    DSP_MODULE_SSK,           // Sample SKipper
    DSP_MODULE_DIS            // DeInterlacer re-Scaling
};

/**
 * @brief Handle descriptor to a on-FPGA DSP module
 */
struct dsp_module_descriptor;
/**
 * @brief Handle to a on-FPGA DSP module
 */
typedef struct dsp_module_descriptor *dsp_module_handle;

/**
 * @brief Queries the list of DSP modules in a given board.
 *
 * This function allows to query the list of DSP modules for a
 * digitizer board. \c modules is a pointer to an array of DSP modules
 * to be filled by this function. The \c numEntries parameter
 * specifies how many modules can be added by the function to the \c
 * modules array. Lastly, the \c numModules array specifies how many
 * modules are avaiable on the specified board.
 *
 * \c modules can be \c NULL. In this case, the only interest of this
 * function is to return the number of modules available. Please note
 * that \c numEntries must be zero if \c modules is \c NULL.
 *
 * \c numModules can be \c NULL. In this case, it is ignored.
 *
 * This function is typically called twice. First without a \c modules
 * array to query the number of available modules, and a second time
 * after allocating an appropriate array.
 *
 * @code{.c}
 * U32 numModules;
 *
 * U32 retCode = AlazarDSPGetModules(handle, 0, NULL, &numModules);
 *
 * // Error handling
 *
 * dsp_module_handle modules[numModules];
 *
 * retCode = AlazarDSPGetModules(handle, numModules, modules, NULL);
 *
 * // Error handling
 * @endcode
 *
 * @param boardHandle The handle of the board to query DSP modules
 * for.
 *
 * @param numEntries The maximum number of entries that the function
 * can fill in the \c modules array. Must be zero if \c modules is \c
 * NULL.
 *
 * @param modules The array where this function fills the \c
 * dsp_module_handle elements. Can be \c NULL.
 *
 * @param numModules Returns the number of DSP modules available on
 * this board. Ignored if \c NULL.
 *
 * @returns ApiSuccess upon success.
 */
RETURN_CODE EXPORT AlazarDSPGetModules(HANDLE boardHandle, U32 numEntries,
                                       dsp_module_handle *modules,
                                       U32 *numModules);

/**
 * @brief Get information about a specific On-FPGA DSP implementation
 *
 * Use this function to query the type of a DSP module, as well as
 * other information.
 *
 * @param dspHandle The handle to the DSP module to query.
 *
 * @param dspModuleId The identifier of the DSP module. This describes
 * what the type of this module is, and can be compared against the
 * \ref DSP_MODULE_TYPE enum.
 *
 * @param versionMajor The major version number of the DSP implementation.
 *
 * @param versionMinor The minor version number of the DSP implementation.
 *
 * @param maxLength The maximum length of the records that can be
 * processed.
 *
 * @param reserved0 Reserved parameter. Ignored
 *
 * @param reserved1 Reserved parameter. Ignored
 *
 * @returns ApiSuccess upon success.
 */
RETURN_CODE EXPORT AlazarDSPGetInfo(dsp_module_handle dspHandle,
                                    U32 *dspModuleId, U16 *versionMajor,
                                    U16 *versionMinor, U32 *maxLength,
                                    U32 *reserved0, U32 *reserved1);

/**
 * @brief Fills an array with a generated window function and pads it
 * with zeros.
 *
 * \image html window_function-250dpi.png
 *
 * Please note that the windows length can take any integer value. It does not
 * need to meet the alignment requirements that apply to the record length, nor
 * the power-of-two requirement of the FFT length. This can allow users a very
 * high level of control over the effective acquired record length.
 *
 * For example, if a laser source guarantees 1396 good data points at a
 * particular frequency, the number of samples per record on ATS9360 should be
 * set to 1408 (the next multiple of 128) and the FFT length should be 2048
 * points. The window function will be generated with a \c windowLength_samples
 * of 1396, and a \c paddingLength_samples of 652 (2048 - 1396).
 *
 * @param windowType Type of window to generate. Pass an item from
 * \ref DSP_WINDOW_ITEMS enum.
 *
 * @param window Array to be filled with the window function. It must
 * be at least \c windowLength_samples + \c paddingLength_samples
 * long.
 *
 * @param windowLength_samples The size of the window to generate.
 *
 * @param paddingLength_samples The number of samples after the window
 * function to pad with zeros.
 *
 * @remark Using Python, the window array is not allocated first then passed as
 * an output parameter. Instead, it is directly returned from the function as a
 * newly allocated NumPy array.
 *
 * @returns ApiSuccess upon sucess.
 */
RETURN_CODE EXPORT AlazarDSPGenerateWindowFunction(U32 windowType,
                                                   float *window,
                                                   U32 windowLength_samples,
                                                   U32 paddingLength_samples);

/**
 * @brief Queries the maximum trigger repeat rate that the FFT engine can
 * support without overflow.
 *
 * This utility function is useful to calculate the theoretical maximum speed at
 * which FFTs can be computed on a specific digitizer. The value returned only
 * takes into account the FFT processing speed of the on-board module. Other
 * parameters such as bus transfer speed must still be taken into account to
 * ensure that an acquisition is possible on a given board.
 *
 * @warning This function is available for FFT modules versions 4.5 and up.
 *
 * @param[in] dspHandle The board for which to calculate the maximum trigger
 * rate.
 *
 * @param[in] fftSize The number of points acquired by the board per FFT
 * operation.
 *
 * @param[out] maxTriggerRepeatRate Output parameter that gets assigned the
 * maximum trigger rate supported by this board's FFT processing module in
 * Hertz.
 *
 * @returns ApiSucces upon success
 *
 * @returns ApiInvalidDspModule if the FFT module is invalid (wrong type or
 * version)
 */
RETURN_CODE EXPORT AlazarFFTGetMaxTriggerRepeatRate(
    dsp_module_handle dspHandle, U32 fftSize, double *maxTriggerRepeatRate);

/**
 * @brief Download the record for the background subration feature to a board.
 *
 * Pass this function a pointer to an 16-bit integer array containing the record
 * you want to download, and the size of this record in samples.
 *
 * This function should be called before or between acquisitions, not during
 * one.
 */
RETURN_CODE EXPORT AlazarFFTBackgroundSubtractionSetRecordS16(
    dsp_module_handle dspHandle, const S16 *record, U32 size_samples);

/**
 * @brief Reads the background subtraction record from a board.
 *
 * This function can be called to read which record the board uses for the
 * background subtraction feature. It is used by allocating an array of the
 * right size, then passing it to \c backgroundRecord along with it's size in
 * samples to \c size_samples.
 *
 * This function should be called before or between acquisitions, not during
 * one.
 */
RETURN_CODE EXPORT AlazarFFTBackgroundSubtractionGetRecordS16(
    dsp_module_handle dspHandle, S16 *backgroundRecord, U32 size_samples);

/**
 * @brief Controls the activation of the background subtraction feature.
 *
 * Passing \c true to \c enabled activates background subtraction. Passing \c
 * false deactivates it.
 *
 * This function should be called before or between acquisitions, not during
 * one.
 */
RETURN_CODE EXPORT AlazarFFTBackgroundSubtractionSetEnabled(
    dsp_module_handle dspHandle, BOOL enabled);

/**
 * @brief Sets the window function to use with an on-FPGA FFT module.
 *
 * Downloads a window function to an AlazarTech digitizer's
 * memory. This window function will be used during all subsequent
 * acquisitions that use the on-FPGA DSP module.
 *
 * This function should be called before AlazarFFTSetup(). It does not
 * have to be called every time an acquisition is done. It can be
 * located in the board configuration section.
 *
 * @warning Please note that the window function is not compatible with the FFT
 * verification mode.
 *
 * @param dspHandle The handle of the FFT DSP module to set the window
 * function for.
 *
 * @param samplesPerRecord The number of samples in the window
 * function array.
 *
 * @param realWindowArray The real window function array. Passing \c NULL is
 * equivalent to passing an array filled with ones. The values of the window
 * function must be in the interval \f$[-1, 1]\f$.
 *
 * @param imagWindowArray The imaginary window function array. Passing \c NULL
 * is equivalent to passing an array filled with zeros. The values of the window
 * function must be in the interval \f$[-1, 1]\f$.
 */
RETURN_CODE EXPORT AlazarFFTSetWindowFunction(dsp_module_handle dspHandle,
                                              U32 samplesPerRecord,
                                              float *realWindowArray,
                                              float *imagWindowArray);

/**
 *  @cond INTERNAL_DECLARATIONS
 *
 *  @brief Reads the window function from an on-FPGA FFT module.
 *
 *  This function is used for debug purposes, to read back the contents of the
 *  Window RAM after it has been written.
 *
 *  @warning This function is not part of the official AlazarDSP API. It can be
 *  deprecated, removed or changed in future versions.
 *
 *  @endcond
 */
/// @cond INTERNAL_DECLARATIONS
RETURN_CODE EXPORT AlazarFFTGetWindowFunction(dsp_module_handle dspHandle,
                                              U32 samplesPerRecord,
                                              float *realWindowArray,
                                              float *imagWindowArray);
/// @endcond
/**
 * @brief Use a pre-determined record as input to the FFT module.
 *
 * This function configures a simulation mode where the input of the
 * FFT module is replaced with a predefined complex record. In this
 * mode, the FFT module of the board can be tested with known data
 * without having to connect anything to the board.
 *
 * This function is not compatible with the hardware windowing
 * feature. Please see \ref AlazarFFTSetWindowFunction for more
 * information. Users need to emulate the effect of the hardware
 * windowing in software if they want to reproduce it. This is the
 * reason why the record input has real and imaginary components.
 *
 * @param dspHandle The handle of the DSP FFT module to set in
 * verification mode.
 *
 * @param enable Enables the simulation mode.
 *
 * @param realArray The real part of the simulated record to pass to the input
 * of the FFT. Pass \c NULL if \c enable is set to \c FALSE. The board will use
 * the 12 most significant bits of the values passed in this array. The 4 least
 * significant bits of each value should be zero.
 *
 * @param imagArray The imaginary part of the simulated record to pass to the
 * input of the FFT. Pass \c NULL if \c enable is set to \c FALSE or to get the
 * same behavior as passing an array of zeros. The board will use the 12 most
 * significant bits of the values passed in this array. The 4 least significant
 * bits of each value should be zero.
 *
 * @param recordLength_samples The size of the record array in samples.
 *
 */
RETURN_CODE EXPORT AlazarFFTVerificationMode(dsp_module_handle dspHandle,
                                             BOOL enable, S16 *realArray,
                                             S16 *imagArray,
                                             size_t recordLength_samples);

/**
 * @brief Configure the board for an FFT acquisition.
 *
 * This function needs to be called in the board configuration
 * procedure, therefore before AlazarBeforeAsyncRead().
 *
 * The output format of the fft is controlled by the \c outputFormat parameter,
 * with the \ref FFT_OUTPUT_FORMAT enumeration. All elements of \ref
 * FFT_OUTPUT_FORMAT except \ref FFT_OUTPUT_FORMAT_RAW_PLUS_FFT describe a data
 * type (unsigned 8-bit integer, floating point number, etc.) as well as a scale
 * (logarithmic or amplitude squared). It is mandatory to select one (and only
 * one) of these.
 *
 * On the other hand, when \ref FFT_OUTPUT_FORMAT_RAW_PLUS_FFT is OR'ed (using
 * the C \c | operator) to another symbol, it has the meaning of asking the
 * board to output both the time-domain (raw) and FFT data.
 *
 * @param dspHandle The FFT module to configure.
 *
 * @param inputChannelMask The channels to acquire data from. This
 * must be \ref CHANNEL_A.
 *
 * @param recordLength_samples The number of points per record to
 * acquire. This needs to meet the usual requirements for the number
 * of samples per record. Please see the documentation of
 * AlazarBeforeAsyncRead() for more information.
 *
 * @param fftLength_samples The number of points per FFT. This value
 * must be:
 * - A power of two;
 * - Greater than or equal to \c recordLength_samples;
 * - Less than or equal to the maximum FFT size, as returned by the
 *   AlazarDSPGetInfo() function.
 *
 * @param outputFormat Describes what data is output from the FFT
 * post-processing module. This can be any element of the \ref FFT_OUTPUT_FORMAT
 * enum except \ref FFT_OUTPUT_FORMAT_RAW_PLUS_FFT, optionnaly OR'ed with \ref
 * FFT_OUTPUT_FORMAT_RAW_PLUS_FFT.
 *
 * @param footer Describes if a footer is attached to the returned
 * records. Must be an element of the \ref FFT_FOOTER enum.
 *
 * @param reserved Reserved for future use. Pass 0.
 *
 * @param bytesPerOutputRecord Returns the number of bytes in each
 * record coming out of the FFT module. This value can be used to know
 * how long the allocated DMA buffers must be.
 */
RETURN_CODE EXPORT AlazarFFTSetup(dsp_module_handle dspHandle,
                                  U16 inputChannelMask,
                                  U32 recordLength_samples,
                                  U32 fftLength_samples, U32 outputFormat,
                                  U32 footer, U32 reserved,
                                  U32 *bytesPerOutputRecord);

/**
 * @brief Sets internal scaling and slicing parameters in the FFT module
 *
 * @remark This function is only valid for on-FPGA FFT modules with version
 * *less than* 5.
 * \if FFTV5
 * @remark
 * For modules with version 5.0 and above, please refer to the
 * various scaling parameters available in AlazarDSPSetParameterFloat() and
 * AlazarDSPSetParameterS32().
 * \endif
 *
 * @warning This function is intended for advanced users only. Calling it with
 * the wrong parameters can prevent any meaningful data from being output by the
 * FFT module.
 *
 * This function modifies internal parameters used by the on-FPGA FFT module to
 * convert the output of the FFT engine to the desired format. Please refer to
 * the figure below for details as to where conversions happen.
 *
 * To use this function in your program, it is necessary to call it **after**
 * AlazarFFTSetup(), because this is where default scaling and slicing values
 * are set.
 *
 * \image html post-fft-module-250dpi.png
 *
 * @param dspHandle Handle to DSP module
 *
 * @param slice_pos This parameter indicates the position of the most
 * significant bit of the output of slicing operations with respect to the
 * input. Lowering this value by one has the effect of multiplying the output of
 * the FFT module by 2. Default value is 7 for log outputs and 38 otherwise. On
 * the block diagram, this parameter applies to all blocks marked
 * 'Slice'.
 *
 * @param loge_ampl_mult This controls a multiplicative factor used after the
 * log conversion in the FFT module. Hence, it does not apply to 'amplitude
 * squared' outputs. Default value is 4.3429446 for U8 log and float log
 * outputs, and 1111.7938176 for U16 log output.
 */
RETURN_CODE EXPORT AlazarFFTSetScalingAndSlicing(dsp_module_handle dspHandle,
                                                 U8 slice_pos,
                                                 float loge_ampl_mult);

/**
 * @brief Waits until a buffer becomes available or an error occurs.
 *
 * This function should be called instead of
 * AlazarWaitAsyncBufferComplete() in a standard acquisition
 * configuration.
 *
 * @param boardHandle Board that filled the buffer we want to retrieve
 *
 * @param buffer Pointer to the DMA buffer we want to retrieve. This
 * must correspond to the first DMA buffer posted to the board that
 * has not yet been retrieved.
 *
 * @param timeout_ms Time to wait for the buffer to be ready before
 * returning with an ApiWaitTimeout error.
 */
RETURN_CODE EXPORT AlazarDSPGetBuffer(HANDLE boardHandle, void *buffer,
                                      U32 timeout_ms);

/**
 * @brief Equivalent of AlazarDSPGetBuffer() to call with \ref
 * ADMA_ALLOC_BUFFERS
 *
 * This function should be called instead of
 * AlazarWaitNextAsyncBufferComplete() in a standard acquisition
 * configuration. See the documentation of this function for more
 * information.
 *
 * @param boardHandle Board that filled the buffer we want to retrieve
 *
 * @param buffer Pointer to a buffer to receive sample data from the
 * digitizer board.
 *
 * @param bytesToCopy The number of bytes to copy into the buffer.
 *
 * @param timeout_ms Time to wait for the buffer to be ready before
 * returning with an ApiWaitTimeout error.
 */
RETURN_CODE EXPORT AlazarDSPGetNextBuffer(HANDLE boardHandle, void *buffer,
                                          U32 bytesToCopy, U32 timeout_ms);

/**
 * @brief Parameters that can be queried with AlazarDSPGetParameter*()
 *
 * See \ref AlazarDSPGetParameterU32() for information about the way to use
 * these parameters.
 *
 * @internal Elements should take positive values only
 */
enum DSP_PARAMETERS_U32 {
    DSP_RAW_PLUS_FFT_SUPPORTED = 0, ///< Tells if an FFT module supports
                                    /// RAW+FFT mode.
                                    /// This parameter returns \c 0 if
                                    /// RAW+FFT mode is not supported,
                                    /// and \c 1 if it is.
    DSP_FFT_SUBTRACTOR_SUPPORTED    ///< Tells if an FFT module supports
                                    /// the background subtraction feature.
                                    /// This parameter returns \c 0 if
                                    /// the feature is not supported,
                                    /// and \c 1 if it is.
};

/**
 * @brief Generic interface to retrieve U32-typed parameters
 *
 * This function is called with an element of \ref DSP_PARAMETERS_U32 as \c
 * parameter. Depending on which value is selected, the function will query a
 * different parameter internally and pass the return value to \c result.
 *
 * This function returns ApiSuccess upon success, and standard errors otherwise.
 */
RETURN_CODE EXPORT AlazarDSPGetParameterU32(dsp_module_handle dspHandle,
                                            U32 parameter, U32 *result);

/**
 * @brief Generic interface to set U32-typed parameters
 *
 * This function is called with an element of \ref DSP_PARAMETERS_U32 as \c
 * parameter. Depending on which value is selected, the function will write
 * value
 * to different parameter internally.
 *
 * This function returns ApiSuccess upon success, and standard errors otherwise.
 */
RETURN_CODE EXPORT AlazarDSPSetParameterU32(dsp_module_handle dspHandle,
                                            U32 parameter, U32 value);

/**
 * @brief Parameters that can be queried with AlazarDSPGetParameter*()
 *                            or set with AlazarDSPSetParameter*()
 *
 * See \ref AlazarDSPGetParameterS32() and \ref AlazarDSPGetParameterS32()
 * for information about the way to use these parameters.
 *
 */
enum DSP_PARAMETERS_S32 {
    DSP_FFT_POSTPROC_REAL_A = 0, ///< 25-bit signed integer value of "a"
                                 /// for real FFT output value
                                 /// calculation "(Re + a) * b + c".
                                 /// To set this parameter in your program,
                                 /// it is necessary to set it **after**
                                 /// AlazarFFTSetup() call, because this is
                                 /// where its default value is set.
    DSP_FFT_POSTPROC_IMAG_A      ///< 25-bit signed integer value of "a"
                                 /// for imaginary FFT output value
                                 /// calculation "(Im + a) * b + c".
                                 /// To set this parameter in your program,
                                 /// it is necessary to set it **after**
                                 /// AlazarFFTSetup() call, because this is
                                 /// where its default value is set.
};

/**
 * @brief Generic interface to retrieve S32-typed parameters
 *
 * This function is called with an element of \ref DSP_PARAMETERS_S32 as \c
 * parameter. Depending on which value is selected, the function will query a
 * different parameter internally and pass the return value to \c result.
 *
 * This function returns ApiSuccess upon success, and standard errors otherwise.
 */
RETURN_CODE EXPORT AlazarDSPGetParameterS32(dsp_module_handle dspHandle,
                                            U32 parameter, S32 *result);

/**
 * @brief Generic interface to set S32-typed parameters
 *
 * This function is called with an element of \ref DSP_PARAMETERS_S32 as \c
 * parameter. Depending on which value is selected, the function will write
 * value to different parameter internally.
 *
 * This function returns ApiSuccess upon success, and standard errors otherwise.
 */
RETURN_CODE EXPORT AlazarDSPSetParameterS32(dsp_module_handle dspHandle,
                                            U32 parameter, S32 value);

/**
 * @brief Parameters that can be queried with AlazarDSPGetParameter*()
 *                            or set with AlazarDSPSetParameter*()
 *
 * See \ref AlazarDSPGetParameterFloat() and \ref AlazarDSPGetParameterFloat()
 * for information about the way to use these parameters.
 *
 */
enum DSP_PARAMETERS_FLOAT {
    DSP_FFT_POSTPROC_REAL_B = 0,     ///< IEEE754 single precision value of "b"
                                     /// for real FFT output value
                                     /// calculation "(Re + a) * b + c".
                                     /// To set this parameter in your program,
                                     /// it is necessary to set it **after**
                                     /// AlazarFFTSetup() call, because this is
                                     /// where its default value is set.
    DSP_FFT_POSTPROC_REAL_C,         ///< IEEE754 single precision value of "c"
                                     /// for real FFT output value
                                     /// calculation "(Re + a) * b + c".
                                     /// To set this parameter in your program,
                                     /// it is necessary to set it **after**
                                     /// AlazarFFTSetup() call, because this is
                                     /// where its default value is set.
    DSP_FFT_POSTPROC_IMAG_B,         ///< IEEE754 single precision value of "b"
                                     /// for imaginary FFT output value
                                     /// calculation "(Im + a) * b + c".
                                     /// To set this parameter in your program,
                                     /// it is necessary to set it **after**
                                     /// AlazarFFTSetup() call, because this is
                                     /// where its default value is set.
    DSP_FFT_POSTPROC_IMAG_C,         ///< IEEE754 single precision value of "c"
                                     /// for imaginary FFT output value
                                     /// calculation "(Im + a) * b + c".
                                     /// To set this parameter in your program,
                                     /// it is necessary to set it **after**
                                     /// AlazarFFTSetup() call, because this is
                                     /// where its default value is set.
    DSP_FFT_POSTPROC_SCALE_OUT_MAIN, ///< IEEE754 single precision value of
                                     /// the scaler multiplier for
                                     /// the main output.
                                     /// To set this parameter in your program,
                                     /// it is necessary to set it **after**
                                     /// AlazarFFTSetup() call, because this is
                                     /// where its default value is set.
    DSP_FFT_POSTPROC_SCALE_OUT_SEC   ///< IEEE754 single precision value of
                                     /// the scaler multiplier for
                                     /// the secondary output.
                                     /// To set this parameter in your program,
                                     /// it is necessary to set it **after**
                                     /// AlazarFFTSetup() call, because this is
                                     /// where its default value is set.
};

/**
 * @brief Generic interface to retrieve Float-typed parameters
 *
 * This function is called with an element of \ref DSP_PARAMETERS_FLOAT as \c
 * parameter. Depending on which value is selected, the function will query a
 * different parameter internally and pass the return value to \c result.
 *
 * This function returns ApiSuccess upon success, and standard errors otherwise.
 */
RETURN_CODE EXPORT AlazarDSPGetParameterFloat(dsp_module_handle dspHandle,
                                              U32 parameter, float *result);

/**
 * @brief Generic interface to set Float-typed parameters
 *
 * This function is called with an element of \ref DSP_PARAMETERS_FLOAT as \c
 * parameter. Depending on which value is selected, the function will write
 * value to different parameter internally.
 *
 * This function returns ApiSuccess upon success, and standard errors otherwise.
 */
RETURN_CODE EXPORT AlazarDSPSetParameterFloat(dsp_module_handle dspHandle,
                                              U32 parameter, float value);

/**
 * @brief Configure self trigger of the board DSP module.
 *
 * This function needs to be called in the board configuration
 * procedure, therefore before AlazarBeforeAsyncRead().
 *
 * @param dspHandle The DSP module to configure.
 *
 * @param enable Set to TRUE to enable self trigger.
 *
 * @param counter Active ADC clock cycles to count between self-triggers.
 *                Each clock cycle typically holds 8 samples per channel.
 *
 * This function returns ApiSuccess upon success, and standard errors otherwise.
 */
RETURN_CODE EXPORT AlazarDSPConfigureSelfTrigger(dsp_module_handle dspHandle,
                                                 BOOL enable, U32 counter);

/**
 * @brief Configure sample skipping of SSK DSP module.
 *
 * This function needs to be called in the board configuration
 * procedure, therefore before AlazarBeforeAsyncRead().
 *
 * @param dspHandle The DSP module to configure.
 *
 * @param independentMode Use independent channel mask if TRUE else use common
 * mask.
 *
 * @param count Vector size.
 *              If 0 passthrough mode is enabled.
 *
 * @param vector An array sample skipping data.
 *               If NULL passthrough mode is enabled.
 *
 * This function returns ApiSuccess upon success, and standard errors otherwise.
 */
RETURN_CODE EXPORT AlazarDSPConfigureSampleSkipping(dsp_module_handle dspHandle,
                                                    BOOL independentMode,
                                                    U32 count, U16 *vector);

/**
 * @brief Aborts any in-progress DMA transfer, cancels any pending
 * transfers and does DSP-related cleanup.
 *
 * This function should be called instead of AlazarAbortAsyncRead() in
 * a standard acquisition configuration. In addition to handling
 * pending and in-flight DMA transfers, it takes care of some cleanup
 * related to the DSP post-processing.
 *
 * @warning Whereas it is not necessary to call AlazarAbortAsyncRead()
 * to clean after a standard acquisition, calling
 * AlazarDSPAbortCapture() is *strictly required*.
 *
 * @param boardHandle The board to stop the acquisition for.
 *
 */
RETURN_CODE EXPORT AlazarDSPAbortCapture(HANDLE boardHandle);

/**
 * @brief FFT output format enumeration
 */
enum FFT_OUTPUT_FORMAT {
    /**
     * 32-bit unsigned integer amplitude squared output.
     * \if FFTV5
     * Available only for version below 5.0.
     * \endif
     */
    FFT_OUTPUT_FORMAT_U32_AMP2 = 0x0,
    /**
     * 16-bit unsigned integer logarithmic amplitude output.
     * \if FFTV5
     * Available only for version below 5.0.
     * \endif
     */
    FFT_OUTPUT_FORMAT_U16_LOG = 0x1,
    /**
     * 16-bit unsigned integer amplitude squared output.
     * \if FFTV5
     * Available only for version below 5.0.
     * \endif
     */
    FFT_OUTPUT_FORMAT_U16_AMP2 = 0x101,
    /**
     * 8-bit unsigned integer logarithmic amplitude output.
     * \if FFTV5
     * Available only for version below 5.0.
     * \endif
     */
    FFT_OUTPUT_FORMAT_U8_LOG = 0x2,
    /**
     * 8-bit unsigned integer amplitude squared output.
     * \if FFTV5
     * Available only for version below 5.0.
     * \endif
     */
    FFT_OUTPUT_FORMAT_U8_AMP2 = 0x102,
    /**
     * 32-bit signed integer real part of FFT output.
     */
    FFT_OUTPUT_FORMAT_S32_REAL = 0x3,
    /**
     * 32-bit signed integer imaginary part of FFT output.
     */
    FFT_OUTPUT_FORMAT_S32_IMAG = 0x4,
    /**
     * 32-bit floating point amplitude squared output.
     */
    FFT_OUTPUT_FORMAT_FLOAT_AMP2 = 0xA,
    /**
     * 32-bit floating point logarithmic output.
     */
    FFT_OUTPUT_FORMAT_FLOAT_LOG = 0xB,
    /**
     * Prepend each FFT output record with a signed 16-bit version of the
     * time-domain data.
     */
    FFT_OUTPUT_FORMAT_RAW_PLUS_FFT = 0x1000,

    /// @cond FFTV5
    /**
     * 32-bit signed integer amplitude squared output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S32_AMP2 = 0x400,
    /**
     * 32-bit signed integer logarithmic amplitude output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S32_LOG,
    /**
     * 32-bit signed integer phase output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S32_PHASE,
    /**
     * 16-bit signed integer real part of FFT output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S16_REAL,
    /**
     * 16-bit signed integer imaginary part of FFT output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S16_IMAG,
    /**
     * 16-bit signed integer logarithmic amplitude output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S16_LOG,
    /**
     * 16-bit signed integer amplitude squared output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S16_AMP2,
    /**
     * 16-bit signed integer phase output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S16_PHASE,
    /**
     * 8-bit signed integer real part of FFT output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S8_REAL,
    /**
     * 8-bit signed integer imaginary part of FFT output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S8_IMAG,
    /**
     * 8-bit signed integer logarithmic amplitude output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S8_LOG,
    /**
     * 8-bit signed integer amplitude squared output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S8_AMP2,
    /**
     * 8-bit signed integer phase output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S8_PHASE,
    /**
     * 32-bit floating point real part of FFT output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_FLOAT_REAL,
    /**
     * 32-bit floating point imaginary part of FFT output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_FLOAT_IMAG,
    /**
     * 32-bit floating point phase output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_FLOAT_PHASE,
    /**
     * 16-bit signed integer amplitude squared plus phase output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S16_AMP2_AND_PHASE,
    /**
     * 16-bit signed integer logarithmic amplitude plus phase output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S16_LOG_AND_PHASE,
    /**
     * 16-bit signed integer real plus imaginary parts of FFT output.
     * Available only for version 5.0 and above.
     */
    FFT_OUTPUT_FORMAT_S16_REAL_AND_IMAG,
    /// @endcond

    /// @cond INTERNAL_DECLARATIONS
    FFT_OUTPUT_FORMAT_DEBUG_DP2_M5,
    FFT_OUTPUT_FORMAT_DEBUG_DP2_M6,
    FFT_OUTPUT_FORMAT_DEBUG_DP2_M7,

    FFT_OUTPUT_FORMAT_U32 = 0x0,
    FFT_OUTPUT_FORMAT_REAL_S32 = 0x3,
    FFT_OUTPUT_FORMAT_IMAG_S32 = 0x4
    /// @endcond
};

/**
 * @brief FFT footer enumeration
 */
enum FFT_FOOTER { FFT_FOOTER_NONE = 0x0, FFT_FOOTER_NPT = 0x1 };

#    ifdef __cplusplus
}
#    endif

#endif // ALAZARFFT_H
