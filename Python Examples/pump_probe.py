import os
import sys

ATS9373_LIB_PATH = 'C:/Users/femtoQLab/source/repos/GPUAcquisition/x64/Release'
ATS_GPU_LIB_PATH = 'C:/AlazarTech/ATS-GPU/3.7.0/base/library/x64'
if ATS9373_LIB_PATH not in sys.path:
    sys.path.append(ATS9373_LIB_PATH)
if ATS_GPU_LIB_PATH not in os.environ['PATH']:
    os.environ['PATH'] += f';{ATS_GPU_LIB_PATH};'

import time
import numpy as np
import SMC100CC
import serial
import GPUAcquisition
from GPUAcquisition import ATS, config, info
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# Setup the arduino
# arduino = serial.Serial('COM3', 115200)

# Configure the stage
stage = SMC100CC.SMC100(1,'COM4')
stage.reset_and_configure()
stage.home()

testing = (input('Are you testing (y/N)? ') or 'N').upper() == 'Y'

# Grab the stage parameters from the user
max_pos = -32.787
max_pos = float(input(f'Max stage pos [mm] ({max_pos}): ') or max_pos)

min_pos = -34.787
min_pos = float(input(f'Min stage pos [mm] ({min_pos}): ') or min_pos)

# Swap if the order is incorrect
if min_pos > max_pos:
    min_pos, max_pos = max_pos, min_pos

zero_delay_pos = -33.787
zero_delay_pos = float(input(f'Zero delay pos [mm] ({zero_delay_pos}): ') or zero_delay_pos)

step_size = 100
step_size = int(input(f'Step size [µm] ({step_size}): ') or step_size)
step_size /= 1000

if min_pos <= -49.999 or min_pos >= 49.999 or max_pos <= -49.999 or max_pos >= 49.999:
    raise RuntimeError('Either your min pos or max pos is out of the valid range of the stage')

# Calculate the stage parameters
num_steps = int(np.ceil((max_pos - min_pos)/step_size))
theoretical_positions = np.linspace(max_pos, min_pos, num=num_steps+1)
measured_positions = np.zeros(num_steps+1)

# Get stage setup for data acquisition
stage.set_speed(2)
stage.move_absolute_mm(theoretical_positions[0]-0.001)
stage.set_speed(1)

# Acquisition parameters
sample_rate = 2000000000
bytes_per_sample = 3/2
approx_num_records = 100000
approx_bytes_per_buffer = 1<<23
samples_per_record = 2000
bytes_per_record = samples_per_record * bytes_per_sample
records_per_buffer = int(approx_bytes_per_buffer // bytes_per_record)
num_buffers = int(np.ceil(approx_num_records / records_per_buffer))
num_records = records_per_buffer * num_buffers

# How we convert from pulse arrival to wavelength
a, b, c = 0.000814913056, -2.16787083, 1620.87034
def pk_arrival_to_wl(arrival):
    return (-b + np.sqrt(b**2 - 4*a*(c-arrival)))/(2*a)
min_arrival_time = c - b**2/(4*a)

def pos_to_delay(pos):
    return (pos - zero_delay_pos) * 2e12 / (1e3 * 299792458)

plot_data = np.zeros((int(samples_per_record//2), num_steps+1))
plot = plt.imshow(plot_data, cmap='seismic', aspect='auto', extent=[pos_to_delay(min_pos), pos_to_delay(max_pos), 1000, 0])
plt.xlabel('Delay [ps]')
plt.ylabel('Signal time [ns]')

# Start the experiment
pbar = trange(num_steps+1)
for i in pbar:
    # Move the stage and measure its position
    pbar.set_description('Moving stage')
    stage.move_absolute_mm(theoretical_positions[i])
    measured_positions[i] = stage.get_position_mm()

    # Configure the acquisition
    pbar.set_description('Configuring acquisition')
    acq = GPUAcquisition.Acquisition()

    conf = config.AcquisitionConfig({
        'capture_clock' : {
            'source' : ATS.INTERNAL_CLOCK if testing else ATS.EXTERNAL_CLOCK_10MHZ_REF,
            'sample_rate' : ATS.SAMPLE_RATE_2000MSPS if testing else 2000000000,
            'edge' : ATS.CLOCK_EDGE_RISING,
            'decimation' : 0 if testing else 1
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
            'level1' : 128 if testing else 179,
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
            'channels' : ATS.CHANNEL_A | ATS.CHANNEL_B,
            'transfer_offset' : 0,
            'pre_trigger_samples' : 0,
            'post_trigger_samples' : samples_per_record,
            'records_per_buffer' : records_per_buffer,
            'records_per_acquisition': num_records,
            'adma_flags' : ATS.ADMA_EXTERNAL_STARTCAPTURE | ATS.ADMA_NPT | ATS.ADMA_INTERLEAVE_SAMPLES,
            'gpu_flags' : 0
        },
        'data_writing': {
            'fname': f'data/pulses_pos{i}',
            'num_buffs_to_write': num_buffers
        },
        'num_gpu_buffers' : 10
    })
    acq.configure_devices(conf)
    acq.set_ops([], [])

    pbar.set_description('Acquiring data')
    acq.start()
    while not acq.is_finished():
        # Busy loop
        time.sleep(0.100)

    # Make sure the files the acquisition was writing to are closed
    acq.cleanup()

    # Save the stage position in a redundant way (just in case)
    with open(f'data/stage_pos{i}') as out_file:
        out_file.write(str(measured_positions[i]) + '\n')

    pbar.set_description('Plotting')
    data_a = np.fromfile(f'data/pulse_pos{i}_chan_a', dtype=np.float32)
    data_a = data_a.reshape((data_a.size//samples_per_record, samples_per_record))
    data_b = np.fromfile(f'data/pulse_pos{i}_chan_b', dtype=np.float32)
    data_b = data_b.reshape((data_b.size//samples_per_record, samples_per_record))

    onoff_thresh = 0.2 # TODO: Measure this
    mask = data_b.max(axis=1) > onoff_thresh
    on_pulses = data_a[np.where(mask)]
    off_pulses = data_a[np.where(~mask)]

    num_pulses = min(on_pulses.shape[0], off_pulses.shape[0])
    on_pulses = on_pulses[:num_pulses]
    off_pulses = off_pulses[:num_pulses]

    # TODO: Handle 1/0 problem
    # dts = (on_pulses - off_pulses) / off_pulses
    dts = on_pulses - off_pulses
    dt_avg = dts.mean(axis=0)

    plot_data[:, i] = dt_avg[::-1]
    plot.set_data(plot_data)

# Save the positions to disk
np.savez_compressed('data/positions', measured_positions=measured_positions)

# Set stage to some "normal" position
stage.set_speed(5)
stage.move_absolute_mm(0)

# Release the stage
stage.CloseConnection()

