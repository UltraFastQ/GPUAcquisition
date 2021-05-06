import os
import sys

ATS9373_LIB_PATH = 'C:/Users/femtoQLab/source/repos/GPUAcquisition/x64/Release/'
ATS_GPU_LIB_PATH = 'C:/AlazarTech/ATS-GPU/3.7.0/base/library/x64'
if ATS9373_LIB_PATH not in sys.path:
    sys.path.append(ATS9373_LIB_PATH)
if ATS_GPU_LIB_PATH not in os.environ['PATH']: # os.environ['LD_LIBRARY_PATH'] on *NIX systems
    os.environ['PATH'] += f';{ATS_GPU_LIB_PATH};'

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import GPUAcquisition
from GPUAcquisition import ATS, config, info

has_eclk = (input('Do you have an external clock [Y/n]: ').upper() or 'Y') == 'Y'

source = ATS.EXTERNAL_CLOCK_10MHZ_REF if has_eclk else ATS.INTERNAL_CLOCK
rate = 2000000000 if has_eclk else ATS.SAMPLE_RATE_2000MSPS
samples_per_buff = 1<<23
num_buffs = 10

while True:
    acq = GPUAcquisition.Acquisition()
                
    conf = config.AcquisitionConfig({
        'capture_clock' : {
            'source' : source,
            'sample_rate' : rate,
            'edge' : ATS.CLOCK_EDGE_RISING,
            'decimation' : 1 if has_eclk else 0
        },
        'input_control' : [
            {
                'channel' : ATS.CHANNEL_A,
                'coupling' : ATS.DC_COUPLING,
                'input_range' : ATS.INPUT_RANGE_PM_400_MV,
                'impedance' : ATS.IMPEDANCE_50_OHM,
            },
            {
                'channel' : ATS.CHANNEL_B,
                'coupling' : ATS.DC_COUPLING,
                'input_range' : ATS.INPUT_RANGE_PM_400_MV,
                'impedance' : ATS.IMPEDANCE_50_OHM,
            }
        ],
        'trigger_operation' : {
            'trigger_operation' : ATS.TRIG_ENGINE_OP_J,
            'trigger_engine1' : ATS.TRIG_ENGINE_J,
            'source1' : ATS.TRIG_DISABLE, #ATS.TRIG_EXTERNAL, #ATS.TRIG_CHAN_A,
            'slope1' : ATS.TRIGGER_SLOPE_POSITIVE,
            'level1' : 179,
            'trigger_engine2' : ATS.TRIG_ENGINE_K,
            'source2' : ATS.TRIG_DISABLE,
            'slope2' : ATS.TRIGGER_SLOPE_POSITIVE,
            'level2' : 128
        },
        'external_trigger' : {
            'coupling' : ATS.AC_COUPLING,
            'range' : ATS.ETR_2V5
        },
        'trigger_delay' : 0,
        'trigger_timeout_ticks' : 0,
        'aux_io' : {
            'mode' : ATS.AUX_OUT_PACER, # ATS.AUX_OUT_TRIGGER,
            'parameter' : 125
        },
        'acquisition_setup' : {
            'channels' : ATS.CHANNEL_A, #| ATS.CHANNEL_B,
            'transfer_offset' : 0,
            'pre_trigger_samples' : 0,
            'post_trigger_samples' : int(samples_per_buff),
            'records_per_buffer' : 1,
            'records_per_acquisition': 0x7FFFFFFF,
            'adma_flags' : ATS.ADMA_CONTINUOUS_MODE | ATS.ADMA_EXTERNAL_STARTCAPTURE | ATS.ADMA_INTERLEAVE_SAMPLES,
            'gpu_flags' : 0
        },
        'data_writing': {
            'fname': '',
            'num_buffs_to_write': 0
        },
        'num_gpu_buffers' : int(num_buffs)
    })

    acq.configure_devices(conf)

    sample_time_ps = 1000

    total_window = 1000 * sample_time_ps
    win_on = sample_time_ps * 500
    win_off = total_window - win_on
    win_offset = 0

    ops_chan_a = []
    ops_chan_b = []
    acq.set_ops(ops_chan_a, ops_chan_b)

    fig, ax = plt.subplots()
    chan_a_xdata = []
    chan_a_ydata = []
    chan_b_xdata = []
    chan_b_ydata = []
    chan_a_line, = plt.plot(chan_a_xdata, chan_a_ydata, label='Channel A')
    chan_b_line, = plt.plot(chan_b_xdata, chan_b_ydata, label='Channel B')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()

    data_size = min(2**12, samples_per_buff)
    # data_size = min(2**21, samples_per_buff)

    start_time = time.time()

    def anim_init():
        if ops_chan_a and isinstance(ops_chan_a[0], GPUAcquisition.BoxCarOp):
            ax.set_xlim(0, 30)
        else:
            ax.set_xlim(0, 300)
        #ax.set_ylim(-0.1, 0.2)
        ax.set_ylim(-0.45, 0.45)

        return [chan_a_line, chan_b_line]

    def plot_raw(chan):
        get_data = acq.get_chan_b if chan.upper() == 'B' else acq.get_chan_a

        chan_data = get_data(data_size).cast('f').tolist()
        if not chan_data:
            return xdata, ydata

        chan_data = np.array(chan_data)
        if chan_data.max() > 0.01:
            try:
                idx = np.argwhere(chan_data > chan_data.max() / 2)[0][0]
            except IndexError:
                idx = 120
            chan_data = chan_data[idx - 120 : idx + 600]

        xdata = list(range(len(chan_data)))
        ydata = chan_data
        return xdata, ydata

    def plot_boxcar():
        global chan_a_xdata, chan_a_ydata, start_time
        if chan_a_xdata and chan_a_xdata[-1] >= 30:
            chan_a_xdata = []
            chan_a_ydata = []
            start_time = time.time()

        samples_per_window = (win_on + win_off) / sample_time_ps
        chan_a = acq.get_chan_a(int(np.ceil(data_size / samples_per_window))).cast('f').tolist()

        curr_time = time.time()
        dt = curr_time - start_time
        chan_a_xdata.extend([dt + i*total_window*1e-12 for i in range(len(chan_a))])
        chan_a_ydata.extend(chan_a)

    def anim_update(frame):
        global start_time, chan_a_xdata, chan_a_ydata, chan_b_xdata, chan_b_ydata
        if frame == 0:
            start_time = time.time()
            return [chan_a_line, chan_b_line]

        if not ops_chan_a:
            chan_a_xdata, chan_a_ydata = plot_raw('A')
        elif isinstance(ops_chan_a[0], GPUAcquisition.BoxCarOp):
            plot_boxcar()
        chan_a_line.set_data(chan_a_xdata, chan_a_ydata)

        if not ops_chan_b:
            pass
            # chan_b_xdata, chan_b_ydata = plot_raw('B')
        chan_b_line.set_data(chan_b_xdata, chan_b_ydata)

        return [chan_a_line, chan_b_line]

    print('Starting data acquisition')

    acq.start()
       
    time.sleep(0.100)

    anim = FuncAnimation(fig, anim_update, init_func=anim_init, blit=True, interval=0.01)
    ax.callbacks.connect('xlim_changed', lambda event: anim._blit_cache.clear())
    ax.callbacks.connect('ylim_changed', lambda event: anim._blit_cache.clear())
    #anim.save('gif.gif', writer='imagemagick', fps=50)
    plt.show()

    if input('Press q to quit: ').upper() == 'Q':
        break
