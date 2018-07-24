#import numpy as np
import mclient
from time import sleep
#from time import localtime
#import time
#current_milli_time = lambda: int(round(time.time() * 1000))

#lazarus = mclient.instruments['fridge']
#vna = mclient.instruments['VNA']

#datafolder = r'Z:\\_Oxford Dry Fridge Smeagol\\HEMT data\\'
#datafolder = r'Z:\\Luke\\classes\\microwaves\\vna\\'
#datafolder = r'Z:\\_Data\\Si_uMachine\\201502_PatchmonDevice1\\20150420RT\\'
datafolder = r'Z:\ChanU\Data\171101_SG_cooldown\ucavcpl_1\ucav_1st'
atten=0

if 0:
    import mclient
    from mclient import instruments
#    import time
#    lazarus = instruments.create('fridge', 'sauron', host='sauron.central.yale.internal', port=33590, fridge='LZ')
    vna = instruments.create('VNA','Agilent_E5071C', address='GPIB::29')

if 0:
     import VNA_functions

if 1: # collect existing VNA data
#    title = 'mode3S21'
    collect_data = VNA_functions.collect_and_display(vna,fridge=None,atten=atten,
                        folder=datafolder,
                        save=True, fpre='S21',
                        display=True)
    
if 0:  #do a power sweep 
     pows=np.linspace(0, -50, 11)
     avgs_start=1
     avgs_stop=10
     datafolder = r'Z:\\_Data\\Si_uMachine\\201502_PatchmonDevice1\\20150401Cooldown\\peak8739powersweep\\'
     VNA_functions.power_sweep(vna,pows,fridge=None,folder=datafolder,avgs_start=avgs_start,avgs_stop=avgs_stop,
                                IF_BANDWIDTH=100,NUM_POINTS=1601,fpre='',fpost=".dat",
                                PULSE_TUBE='ON',save=True,revert=True,display=True,atten=atten,
                                reverse=False,sweep_type='LIN',plot_type='MAG',fmt='PLOG')
                                

if 0:#sweep voltage
#    yoko1.do_set_voltage_range(10)
#    yoko1.do_set_source_type(0)
    
#    yoko1 = instruments.create('yoko1', 'Yokogawa_7651', address='GPIB::5', output_state=False)
#    mclient.instruments.remove('yoko1')
    start = np.log(0.1)/np.log(10)
    end = np.log(28)/np.log(10)
    voltages = np.logspace(start,end,30)
    for v in voltages:
        yoko1.do_set_voltage(v)
        sleep(60)
        yoko1.do_set_output_state(1)
        collect_data = VNA_functions.collect_and_display(vna,fridge=None,atten=atten,
                        folder=datafolder,
                        save=True, fpre='with_filter_'+str(v)+'_V',
                        display=True)