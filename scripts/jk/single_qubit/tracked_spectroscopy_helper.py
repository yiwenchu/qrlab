#from mclient import datafile
import numpy as np

def collect_tracked_specs(data_group):

    all_tracked = [s for s in data_group.keys() if s.find('Tracked_Spectroscopy') != -1]

    return all_tracked


    '''
    20140511 / u'175022_Tracked_Spectroscopy'
    20140512 / u'014911_Tracked_Spectroscopy'
    20140512 / u'102231_Tracked_Spectroscopy'
    20140512 / u'110818_Tracked_Spectroscopy'
    u'E:\\Experiment\\Stabilizer\\Cooldown13\\Cooldown13.h5/20140603/130049_Tracked_Spectroscopy'


    u'E:\\Experiment\\Stabilizer\\Cooldown13\\Cooldown13.h5/20140603/143548_Tracked_Spectroscopy'

    u'E:\\Experiment\\Stabilizer\\Cooldown13\\Cooldown13.h5/20140603/145810_Tracked_Spectroscopy'

    u'E:\\Experiment\\Stabilizer\\Cooldown13\\Cooldown13.h5/20140604/024535_Tracked_Spectroscopy'

    u'E:\\Experiment\\Stabilizer\\Cooldown13\\Cooldown13.h5/20140604/110302_Tracked_Spectroscopy'

    u'E:\\Experiment\\Stabilizer\\Cooldown13\\Cooldown13.h5/20140604/150030_Tracked_Spectroscopy'

    '''


#all_yoko_voltages = []
#all_center_freqs = []
#all_ro_powers = []
#all_spec_powers = []


def add_data_master_list(data_group, include=[]):

    if len(include) == 0:
        include = np.arange(len(data_group['yoko_voltages'][:]))

    all_yoko_voltages.extend(data_group['yoko_voltages'][:][include])
    all_center_freqs.extend(data_group['center_freqs'][:][include])
    all_ro_powers.extend(data_group['ro_powers'][:][include])
    all_spec_powers.extend(data_group['spec_powers'][:][include])

def del_el_master_list(index):

    del(all_yoko_voltages[index])
    del(all_center_freqs[index])
    del(all_ro_powers[index])
    del(all_spec_powers[index])