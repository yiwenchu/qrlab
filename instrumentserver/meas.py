import logging
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import time

import objectsharer as objsh
if objsh.helper.backend is None:
    zbe = objsh.ZMQBackend()
    zbe.start_server(addr='127.0.0.1')
zbe.connect_to('tcp://127.0.0.1:55555')     # Instruments server

instruments = objsh.helper.find_object('instruments')
if 'dsgen3' not in instruments.list_instruments():
    ds = instruments.create('dsgen3', 'dummy_signal_generator')
else:
    ds = instruments['dsgen3']
ds.generate_numpy_array(10000000)

def benchmark():
    start = time.time()
    for i in range(2000):
        w = ds.get_wave()
    end = time.time()
    ar = ds.get_numpy_array()
    end2 = time.time()
    print 'Time: %.03f usec/read' % ((end - start)*1e6/1000, )
    print 'Time: %.03f msec/10MB' % ((end2 - end)*1e3, )

benchmark()

zbe.add_qt_timer()



