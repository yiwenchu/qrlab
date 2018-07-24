# Optimize mixer leakage using spectrum analyzer called vspec

from lib.math.minimizer import Parameter, Minimizer
import time
import mclient

RFSRC = 'bncq'                                      # Qubit RF source
FREQ = mclient.instruments[RFSRC].get_frequency()   # Minimize power at this frequency
SPEC = mclient.instruments['vspec']
YNG = mclient.instruments['yngwie']
CHAN = 0                                                # FPGA ssb 'mode' (0 or 1)
V1,V2 = YNG.get('offset%d'%CHAN)
VRANGE = 5000

def leakage_func(params, delay=0.2):
    f = getattr(YNG, 'set_offset%d'%CHAN)
    f([int(params['v1'].value),int(params['v2'].value)])
    time.sleep(delay)
    val = SPEC.get_power()
    print 'Measuring at (%.01f, %.01f): %.01f' % (params['v1'].value, params['v2'].value, val)
    return val

SPEC.set_frequency(FREQ)
SPEC.set_rf_on(True)
time.sleep(1)

m = Minimizer(leakage_func,
              n_it=4, n_eval=13, verbose=True)
m.add_parameter(Parameter('v1', value=V1, vrange=VRANGE))
m.add_parameter(Parameter('v2', value=V2, vrange=VRANGE))
m.minimize()

SPEC.set_rf_on(False)
SPEC.set_frequency(FREQ-1e9)
