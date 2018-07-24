# Script to calibrate JPC readout:
# - Measure average |g> trajectory
# - Measure average |e> trajectory
# - Set read-out envelope (e(t) - g(t))
# - Measure |g> histogram and determine |g> blob
# - Measure |e> histogram and determine |e> blob
# - if MEAS_GE = True take another histogram with a pi/2 pulse on the qubit

import mclient
from pulseseq import sequencer, pulselib
from scripts.single_qubit import rabi

SHOTS = 50000
TIME_CONSTANT = 750
IF_PERIOD = 20.0
ENV_FILE = 'd:/data/ro_env.npy'
SET_WEIGHT_FUNC = True
SET_IQ_POINTS = True
MEAS_GE = False

qubit_info = mclient.get_qubits()[0]
alz = mclient.instruments['alazar']
alz.set_weight_func('')

# Take average |g> trajectory
rg = rabi.Rabi(qubit_info, [0.0001,], real_signals=False)
rg.play_sequence()
alz.setup_avg_shot(20000)
gbuf = alz.take_avg_shot(timeout=50000)

# Take average |e> trajectory
re = rabi.Rabi(qubit_info, [qubit_info.pi_amp], real_signals=False)
re.play_sequence()
alz.setup_avg_shot(20000)
ebuf = alz.take_avg_shot(timeout=50000)

plt.figure()
plt.plot(np.real(gbuf), np.imag(gbuf), label='|g>')
plt.plot(np.real(ebuf), np.imag(ebuf), label='|e>')
plt.legend(loc=0)
#bla

# Calculate envelope
diff = (ebuf - gbuf)
xs = np.arange(len(diff)) * IF_PERIOD
env = (np.real(diff) + 1j * np.imag(diff)) * np.exp(-xs / TIME_CONSTANT)
env /= np.sum(np.abs(env))

plt.figure()
plt.plot(xs, np.real(env), label='I weight')
plt.plot(xs, np.imag(env), label='Q weight')
plt.legend()
#bla

# Save data
np.save(ENV_FILE, env)
if SET_WEIGHT_FUNC:
    alz.set_weight_func(ENV_FILE)
data = mclient.datafile.create_group('%s_single_shot_cal'%(re._timestamp_str))
gset = data.create_dataset('gbuf', data=gbuf, dtype=np.complex)
eset = data.create_dataset('ebuf', data=ebuf, dtype=np.complex)
data.create_dataset('envelope', data=env, dtype=np.complex)

# Find |g> and |e> histogram points (when using calculated envelope)
alz.set_naverages(SHOTS)
trg = rabi.Rabi(qubit_info, [0.0001,], real_signals=False, histogram=True, title='|g>', plot_seqs=False, keep_data=False)
trg.measure()
iqg = np.average(trg.shot_data[:])
print 'IQ |g>: %s' % (iqg,)
tre = rabi.Rabi(qubit_info, [qubit_info.pi_amp,], real_signals=False, histogram=True, title='|e>', plot_seqs=False, keep_data=False)
tre.measure()
iqe = np.average(tre.shot_data[:])
print 'IQ |e>: %s' % (iqe,)

gset.set_attrs(iqg=iqg)
eset.set_attrs(iqe=iqe)
if SET_IQ_POINTS:
    mclient.instruments['readout'].set_IQg(iqg)
    mclient.instruments['readout'].set_IQe(iqe)

if MEAS_GE:
    trpi2 = rabi.Rabi(qubit_info, [qubit_info.pi_amp/2,], real_signals=False, histogram=True, title='|g+e>, with weighting', plot_seqs=False, keep_data=False)
    trpi2.measure()
