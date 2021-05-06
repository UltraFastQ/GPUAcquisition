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
from matplotlib.animation import FuncAnimation
import GPUAcquisition
from GPUAcquisition import ATS, config, info

fname = 'data.bin'

samples_per_second = 2000000000
ns_per_record = 100
approx_num_records = 100000
approx_num_records = int(input(f'Number of pulses to capture [{approx_num_records}]: ') or approx_num_records)
bytes_per_sample = 3/2 # 12 bits
approx_bytes_per_buffer = 1<<23 # 8 MiB

while True:
    ns_per_record = int(input(f'Time per record (ns) [{ns_per_record}]: ') or ns_per_record)

    samples_per_record = int(ns_per_record * 1e-9 * samples_per_second)
    bytes_per_record = samples_per_record * bytes_per_sample
    records_per_buffer = int(approx_bytes_per_buffer // bytes_per_record)
    num_buffers = int(np.ceil(approx_num_records / records_per_buffer))

    num_records = records_per_buffer * num_buffers
    num_bytes = num_records * samples_per_record * bytes_per_sample
    print(f'Number of pulses that will be acquired: {num_records}')
    print(f'Size of acquisition: {num_bytes/(1<<20):.3f} MiB')

    fname = input(f'Name of file to save data to ["{fname}"]: ') or fname

    acq = GPUAcquisition.Acquisition()
                
    conf = config.AcquisitionConfig({
        'capture_clock' : {
            'source' : ATS.EXTERNAL_CLOCK_10MHZ_REF, # ATS.INTERNAL_CLOCK,
            'sample_rate' : samples_per_second,
            'edge' : ATS.CLOCK_EDGE_RISING,
            'decimation' : 1
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
            'source1' : ATS.TRIG_EXTERNAL,
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
            'mode' : ATS.AUX_OUT_PACER,
            'parameter' : 125
        },
        'acquisition_setup' : {
            'channels' : ATS.CHANNEL_A, #| ATS.CHANNEL_B,
            'transfer_offset' : 0,
            'pre_trigger_samples' : 0,
            'post_trigger_samples' : samples_per_record,
            'records_per_buffer' : records_per_buffer,
            'records_per_acquisition': num_records,
            'adma_flags' : ATS.ADMA_EXTERNAL_STARTCAPTURE | ATS.ADMA_NPT | ATS.ADMA_INTERLEAVE_SAMPLES,
            'gpu_flags' : 0
        },
        'num_gpu_buffers' : 10,
        'data_writing' : {
            'fname' : fname,
            'num_buffs_to_write' : 0
        }
    })

    acq.configure_devices(conf)

    ops_chan_a = []
    ops_chan_b = []
    acq.set_ops(ops_chan_a, ops_chan_b)

    acq.start()

    while True:
        if acq.is_finished():
            break
        time.sleep(0.100)

    time.sleep(0.5)
    print(f'Done acquiring to {fname}')
    user_input = input('Press enter for more acquisitions or q to quit: ')
    if user_input == 'q':
        break
