import plotconfig

datadir = 'c:/Data'
#datafilename = 'c:/Data/20170217/PnC3Biv.h5'
#datafilename = 'C:/Data/20170514/CAll.h5'
datafilename = 'C:/Data/20180508/C10.h5'
#datafilename = 'c:/Data/20170208/PnC3b.h5'
#datafilename = 'c:/Data/20160224/C11Q2.h5'
#datafilename ='c:/Data/20160224/C11Q2.h5'
#datafilename = 'c:/Data/20160331/C10.h5'
tempfilename = 'c:/Data/temp.h5'
ins_store_fn = 'c:/Data/ins_settings.set'

# Use AWG bulk waveform/sequence loading if available.
# Should be faster but apparently isn't always.
awg_bulkload = False
 
awg_fileload = True
#dot_awg_path = r'Z:\_AWG\smeagol2'
dot_awg_path = r'C:\awg_seqs'
dot_awg_path_awg = r'Y:\\'
# Force generation of these channels even if no sequence present
required_channels = (1, 2, 3, 4,)

ENV_FILE = r'C:\labpython\qrlab\scripts\calibration\ro_weight_func.npy'


#added 3/24/16 for patcmon readout
#channel_delays=[['4m1',0],['4m2',100]] #readout pulse, alazar start
1