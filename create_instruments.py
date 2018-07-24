import mclient
from mclient import instruments
import time

### VNA
#vna = instruments.create('VNA', 'Agilent_E5071C', address='TCPIP0::169.254.45.52::inst0::INSTR')
##vna = instruments.create('VNA', 'Agilent_E5071C', address='TCPIP0::172.28.143.85::inst0::INSTR')
#vna = instruments.create('VNA', 'Agilent_E5071C', address='TCPIP0::172.28.140.252::inst0::INSTR')

# Example create_instruments file:
# - Address aliases can be set using NI Measurement & Automation explorer

# Source to trigger everything
# fg = instruments.create('funcgen', 'BNC_FuncGen645', address='FUNCGEN')
funcgen = instruments.create('funcgen', 'Agilent_33250A', address='GPIB::10', sync_on = True, frequency = 2e3)

#laserfg = instruments.create('laserfg', 'Agilent_FuncGen33250A', address='GPIB::9', output_on=False)


# # Setup Alazar
alz = instruments.create('alazar', 'Alazar_Daemon',
                         ch1_range=0.04, ch2_range=0.05,
                         ch1_coupling='AC', ch2_coupling='AC',
                         clock_source='EXT10M', sample_rate='1GEXT10',
                         nsamples=10240,
                         engJ_trig_src='EXT', engJ_trig_lvl=135,
                         )


# AWGs, waittime=0 is used to not wait on them now.
instruments.create('AWG1', 'Tektronix_AWG5014C', address='AWG_SMEAGOL', waittime=0,
                    clock=1e9, refsrc='EXT', reffreq=10e6,
                    trig_level=0.5, trig_slope='POS', trig_impedance=50, ch1_m1_high = 2, ch2_m1_high = 2, 
                    ch4_m1_high = 2, ch4_m2_high = 2,
                    )             

#instruments.create('spec', 'SA124B', serial = 60050241)


# RF sources
#bnc1 = instruments.create('bnc1', 'BNC_RFsource', address='RFCAVMOD')
ag1 = instruments.create('ag1', 'Agilent_N5183A', address='GPIB::20')
#ag2 = instruments.create('ag2', 'Agilent_N5183A', address='GPIB::17')
#ag2 = instruments.create('ag2', 'Agilent_N5183A', address='GPIB::19')
#brick1 = instruments.create('brick1', 'LabBrick_RFSource', devid=0)
#brick1.set_parameter_bounds('frequency', 5e9, 10e9)

# Read out settings
readout = instruments.create('readout', 'Readout_Info',
                rfsource1='ag1', rfsource2='ag2', pulse_len=1000,
                readout_chan='4m1', acq_chan='4m2')



LO_brick = instruments.create('LO_brick', 'LabBrick_RFSource', serial=5964,
                                pulse_on=False, use_extref=True,)
LO_info = instruments.create('LO_info', 'Qubit_Info',
                               deltaf=0,
                               channels='200,201',
                               sideband_channels='Is,Qs')

spec_brick = instruments.create('spec_brick', 'LabBrick_RFSource', serial=2490,
                                pulse_on=True, use_extref=True,)
spec_info = instruments.create('spec_info', 'Qubit_Info',
                               deltaf=0,
                               channels='100,101',
                               sideband_channels='Is,Qs',
                               marker_channel='1m1')

#va_lo = instruments.create('va_lo', 'LabBrick_RFSource', serial=5963)
#va_lo_2 = instruments.create('va_lo_2', 'LabBrick_RFSource', serial=10389)

#qubit_brick = instruments.create('qubit_brick', 'LabBrick_RFSource', serial=1351)
#cavity_brick = instruments.create('cavity_brick', 'LabBrick_RFSource', serial=10388)
qubit_brick = instruments.create('qubit_brick', 'LabBrick_RFSource', serial=10388)#4-8
qubit_info = instruments.create('qubit_info', 'Qubit_Info',
        deltaf=-40e6,
        pi_area=100,
        rotation='Gaussian',
        sigma=20,
        channels='1,2',
        sideband_channels='Ia,Qa',
        sideband_phase=0,
        marker_ofs=-85,
        marker_bufwidth=5,
        )
        
        
qubit_ef_info = instruments.create('qubit_ef_info', 'Qubit_Info',
        deltaf=-180e6,
        pi_area=100,
        rotation='Gaussian',
        sigma=40,
        channels='1,2',
        sideband_channels='Ix,Qx',
        sideband_phase=0,
        marker_ofs=-85,
        marker_bufwidth=5,
        )        
    
#qubit_brick_2 = instruments.create('qubit_brick_2', 'LabBrick_RFSource', serial=2491)#5-10
#qubit_info_2 = instruments.create('qubit_info_2', 'Qubit_Info',
#        deltaf=-70e6,
#        pi_area=100,
#        rotation='Gaussian',
#        sigma=20,
#        channels='3,4',
#        sideband_channels='Ib,Qb',
#        sideband_phase=0,
#        marker_ofs=-85,
#        marker_bufwidth=5,
#        )
#        
#qubit_ef_info_2 = instruments.create('qubit_ef_info_2', 'Qubit_Info',
#        deltaf=-130e6,
#        pi_area=100,
#        rotation='Gaussian',
#        sigma=40,
#        channels='3,4',
#        sideband_channels='Iy,Qy',
#        sideband_phase=0,
#        marker_ofs=-85,
#        marker_bufwidth=5,
#        )  
        
#        
#qubit_2_info = instruments.create('qubit_2_info', 'Qubit_Info',
#        deltaf=-50e6,
#        pi_area=100,
#        rotation='Gaussian',
#        sigma=20,
#        channels='1,2',
#        sideband_channels='Ib,Qb',
#        sideband_phase=0,
#        marker_ofs=-85,
#        marker_bufwidth=5,
#        )

phonon1_info = instruments.create('phonon1_info', 'Qubit_Info',
        deltaf=-50e6,
        pi_area=100,
        rotation='Gaussian',
        sigma=20,
        channels='1,2',
        sideband_channels='Ib,Qb',
        sideband_phase=0,
        marker_ofs=-85,
        marker_bufwidth=5,
        )
        


#qubit_ef_brick = instruments.create('qubit_ef_brick', 'LabBrick_RFSource', serial=2491)    
#     
#qubit_ef_info = instruments.create('qubit_ef_info', 'Qubit_Info',
#        deltaf=-50e6,
#        pi_area=100,
#        rotation='Gaussian',
#        sigma=20,
#        channels='3,4',
#        sideband_channels='Ia,Qa',
#        sideband_phase=0,
#        marker_ofs=-85,
#        marker_bufwidth=5,
#        )

        
cavity_info = instruments.create('cavity_info', 'Qubit_Info',
            deltaf=-50e6,
            pi_area=82,
            rotation='Gaussian',
            sigma=80,
            channels='1,2',
            sideband_channels='I1,Q1',
            sideband_phase=0,
            )
cavity_brick = instruments.create('cavity_brick', 'LabBrick_RFSource', serial=2491)
        
cav_info = instruments.create('cavity0', '')


# Spectrum analyzer
#vspec = instruments.create('vspec', 'Vlastakis_Spec', address='COM4',
#                           rfsource='va_lo', df0=10.574e6, Navg=10, delay=0.1)


#va_lo = instruments.create('va_lo', 'LabBrick_RFSource', serial= 1351)
#va_lo = instruments.create('va_lo', 'LabBrick_RFSource', serial= 10389)

#Yokogawa current source:
#yoko1 = instruments.create('yoko1', 'Yokogawa_7651', address='GPIB::1', output_state=False)
#yoko2 = instruments.create('yoko2', 'Yokogawa_7651', address='GPIB::5', output_state=False)

#Agilent power supply
AgPS1 = instruments.create('AgPS1', 'Agilent_E3631A', address='GPIB::3')

# # These were created asynchronously, get them now.
awg1 = instruments['AWG1']

# Load latest available settings
mclient.restore_instruments()

#Yokogawa current source:
yoko1 = instruments.create('yoko1', 'Yokogawa_7651', address='GPIB::1', output_state=False, source_type = 'CURR')#)
yoko2 = instruments.create('yoko2', 'Yokogawa_7651', address='GPIB::5', output_state=False, source_type = 'CURR')#)

instruments.create('spec', 'SA124B', serial = 60050241)

'''
Run these separately for time-out errors:

instruments.create('spec', 'SA124B', serial = 60050241)


spec = instruments['spec']

yokoa = instruments.create('yokoa', 'Yokogawa_GS200', address='GPIB::7', output_state=False, source_type = 'CURR')

'''