import mclient
from mclient import instruments
import time

# Example create_instruments file:
# - Address aliases can be set using NI Measurement & Automation explorer


##awg2 = instruments['AWG2']

# vna = instruments.create('VNA','Agilent_E5071C', address='TCPIP0::172.28.143.1::inst0::INSTR')
vna = instruments.create('VNA','Agilent_E5071C', address='TCPIP0::172.28.143.85::inst0::INSTR') #Galahad
#vna = instruments.create('VNA', 'Agilent_E5071C', address='TCPIP0::VNA1.central.yale.internal::INSTR')
# vna_sg = instruments.create('VNA_SG', 'Agilent_E5071C', address='TCPIP0::VNA1.central.yale.internal::INSTR')
# vna = instruments.create('VNA', 'Agilent_E5071C', address='TCPIP0::172.28.143.185::inst0::INSTR')
#vna= instruments.create('VNA', 'Agilent_E5071C', address='TCPIP0::172.28.142.166::inst0::INSTR')

# spec = instruments.create('spec','Vlastakis_Spec',address='COM9',
                           # rfsource='brick3', df0=10.689e6, Navg=10, 
                           # delay=0.04)
# yoko = instruments.create('Yoko','Yokogawa_7651', address='DumbYoko2')



#LZvna = instruments.create('LZVNA', 'Agilent_E5071C', address='TCPIP0::VNA2.central.yale.internal::inst0::INSTR')
# tnvna = instruments.create('tnVNA', 'Agilent_E5071C', address='TCPIP0::172.28.143.185::inst0::INSTR')
# vna = instruments.create('VNA', 'Agilent_E5071C', address='TCPIP0::172.28.143.54::inst0::INSTR') # Merlin

#JPCvna = instruments.create('JPCVNA', 'Agilent_E5071C', address='TCPIP0::172.28.143.46::inst0::INSTR')

## Don't restore instrument values for fridge
#lazarus = instruments.create('fridge', 'sauron', host='sauron.central.yale.internal', port=33590, fridge='LZ')
# jpc = instruments.create('JPCfridge', 'sauron', host='sauron.central.yale.internal', port=33590, fridge='JP')
smeagol = instruments.create('fridge_SG', 'sauron', host='sauron.central.yale.internal', port=33590, fridge='SG')
# charlie = instruments.create('charlie', 'sauron', host='sauron.central.yale.internal', port=33590, fridge='CL')



# tn = instruments.create('tn', 'tn', host='sauron.central.yale.internal', port=33590, fridge='TN')
# yoko = instruments.create('Yoko','Yokogawa_GS200', address='GPIB0::5::INSTR')