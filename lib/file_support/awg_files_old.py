'''
Jacob Blumoff 6/24/2014
'''

import numpy as np
import struct
import types
import time
import hashlib


import glob
import os

import config

def clear_dot_awgs(path=None, delete='all'):#r'Z:\_AWG\Stabilizer'):
    #Todo:  Pull path from config, delete by date/time
    if path == None:
        (path, fn) = os.path.split(config.dot_awg_path)

    key = os.path.join(path,'*.awg')
    dalist = glob.glob(key)
    for idx,item in enumerate(dalist):
        marked_for_death = True

#        if delete == 'all':
#            marked_for_death == True
#
#        elif delete == 'before_today':
#            pass

        if marked_for_death:
            os.remove(item)
            print 'Deleting (%d/%d): %s' % (idx+1,len(dalist),item)
    pass

NULL = chr(0)

'''Make sure the AWG has the path loaded, and that it's mounted as
the same drive letter.'''

class Dot_AWG_Load():
    def __init__(self, seqs, path=None, awg=None):
        '''At a high level this will only know about one AWG.
        Each seqs will already be predivided into the relevant channels
        '''
        self.awg = awg
        self.seqs = seqs
        self.path = path[:-4]+str(int(time.time()*1000))+'.awg'
        self.path = path + '\\' + str(int(time.time()*1000))  + '.awg'
        print self.path

        self.rl = [] #initial record list
        self.loaded_wfs = [] #List of names of loaded wfs
        self.admin_duty = 1 #Triggering, looping, repeating:  only done once

    def load_seqs(self, delay_override = 600e3, verbose = False):
        '''The main routine'''

        MAX_ATTEMPTS = 6
        self.parse_seqs()

        for load_attempt_no in range(MAX_ATTEMPTS):
            try:
                self.pull_stubborn_params()
                self.clear_awg_and_pull_data(delay_override)

                self.write_attempt_no = 0
                self.write_wrapper()

                self.awg.load_dot_awg(self.path)
                if delay_override:
                    self.awg.wait_getID(delay=delay_override, timeout=delay_override)
                self.restore_stubborn_params()

                self.check_seq_length()
                return

            except Exception as e:
                if load_attempt_no != MAX_ATTEMPTS - 1:
                    msg =  'Failed AWG load (%s), big loop, retrying. (%d/%d) ' % (e, load_attempt_no, MAX_ATTEMPTS)
                    print msg
                else:
                    print 'ABORTING, LOAD FAILURE'
                    raise Exception('AWG load failed')

    def clear_awg_and_pull_data(self, delay):
        '''Delete all of the waveform and sequence memory, save a .awg
        file that contains only the device settings.
        The timeout=delay makes ObjectSharer wait long enough for the AWG
        process to respond'''
        self.awg.clear_sequence(timeout=delay)
        self.awg.delete_all_waveforms(wait=delay, timeout=delay) #wait = timeout.  Deleting
                                                  #large expts can take a while.
        self.awg.wait_getID(timeout=delay)
        self.awg.pull_dot_awg(self.path, timeout=delay)

    def pull_stubborn_params(self):
        '''This is where we handle params that aren't maintained by
        restoring the .awg  (there aren't many of them and we're not
        totally sure why this is necessary)'''
        self.offsets = []
        for ch in [1,2,3,4]:
            self.offsets.append(self.awg.do_get_offset(ch))
        #also direct output, filter, DC output - to be added
        #Every other setting I've checked looks fine.

    def restore_stubborn_params(self):
        for ch in [1,2,3,4]:
            self.awg.do_set_offset(float(self.offsets[ch-1]),ch)

    def parse_seqs(self):
        '''Calls add waveform for each sequence element'''
        ch = 0
        for ch in self.seqs:
            if type(ch) != int:
                continue
            for idx,_ in enumerate(self.seqs[ch]):
                self.add_waveform(idx, ch)
            self.admin_duty = 0 #After the first channel

    def add_waveform(self, idx, chan):
        '''Converts a sequence object into a set of .awg records or
        instructions.  Adds both waveform data and sequence data.'''
        if not self.seqs.has_key(chan):
            return

        pulse = self.seqs[chan][idx]
        pulse_name = pulse.get_name()
        pulse_repeat = pulse.repeat
        pulse_len = pulse.get_length() / pulse_repeat
        pulse_data = pulse.get_data()
        pulse_trigger = pulse.get_trigger()

        m1_chan = '%dm1' % chan
        if self.seqs.has_key(m1_chan) and self.seqs[m1_chan] != None:
            m1 = self.seqs[m1_chan][idx]
            m1_name = m1.get_name()
            m1_data = m1.get_data()
        else:
            m1_name = 'delay%d' % pulse_len
            m1_data = np.zeros(pulse_len)

        m2_chan = '%dm2' % chan
        if self.seqs.has_key(m2_chan) and self.seqs[m2_chan] != None:
            m2 = self.seqs[m2_chan][idx]
            m2_name = m2.get_name()
            m2_data = m2.get_data()
        else:
            m2_name = 'delay%d' % pulse_len
            m2_data = np.zeros(pulse_len)

        idx += 1 #AWG indexing starts at 1
        waveform_name = pulse_name+'_'+m1_name+'_'+m2_name
        waveform_name = hashlib.md5(waveform_name).hexdigest() #optional

        if waveform_name not in self.loaded_wfs:
            n_wf = len(self.loaded_wfs)+1 #unique waveform #

            '''each new waveform takes 5 records'''
            self.rl.append(['WAVEFORM_NAME_%d' % n_wf,
                            waveform_name])
            self.rl.append(['WAVEFORM_TYPE_%d' % n_wf,
                            np.array([1], dtype=np.uint16)])
            self.rl.append(['WAVEFORM_LENGTH_%d' % n_wf,
                            np.array([pulse_len], dtype=np.uint32)])
            self.rl.append(['WAVEFORM_TIMESTAMP_%d' % n_wf,
                            self.make_timestamp()])

            '''Combine the analog and marker data into 16 bit words'''
            pulse_data_processed =  self.awg.get_bindata(pulse_data,
                                                     m1=m1_data,
                                                     m2=m2_data)
            self.rl.append(['WAVEFORM_DATA_%d' % n_wf,
                            pulse_data_processed])

            self.loaded_wfs.append(waveform_name)

            if n_wf > 32000:
                raise ValueError('Too many waveforms!  32000 wf limit.')

        '''element by element sequencing'''
        self.rl.append(['SEQUENCE_WAVEFORM_NAME_CH_%d_%d' % (chan,idx),
                        waveform_name])

        if self.admin_duty:
            '''It's important that these are only written once, otherwise
            the awg fails'''
            goto_idx = int(idx == len(self.seqs[chan].seq))
            self.rl.append(['SEQUENCE_GOTO_%d' % (idx),
                            np.array([goto_idx],dtype='<u2')])

            self.rl.append(['SEQUENCE_WAIT_%d' % (idx),
                            np.array([pulse_trigger],dtype='<u2')])

            self.rl.append(['SEQUENCE_JUMP_%d' % (idx),
                            np.array([0],dtype='<u2')])

            self.rl.append(['SEQUENCE_LOOP_%d' % (idx),
                            np.array([pulse_repeat],dtype='<u4')])

    def compose_record(self,name, data):
        '''
        Composes the line-item instructions in the .awg file into the
        proper format.

        name should be an AWG record name, i.e. 'MAGIC' or 'WAVEFORM_NAME_1'
        data is a 1D numpy array or a string.
        '''
        name += NULL #The Record_Name ends in NULL
        name_len = len(name)

        if type(data) is types.StringType:
            data = np.fromstring(data, dtype=np.uint8)
            data = np.concatenate([data,np.array([0,], dtype=np.uint8)])
        data_bytes = data.tostring()
        data_len = len(data_bytes)
        fmt_string = '<II'+'%ds%ds'%(name_len,data_len)

        return struct.pack(fmt_string, name_len, data_len, name, data_bytes)

    def write_dot_awg(self):
        recs = [' ']*len(self.rl)
        for idx,[name,data] in enumerate(self.rl):
            recs[idx] = self.compose_record(name, data)

        try:
            initial_file_size = os.path.getsize(self.path)
            print '.AWG file init: %d bytes' % initial_file_size
        except:
            raise Exception('Failed at reading init filezise')

        with open(self.path,'ab') as f:
            '''Note the 'a' for appending to the pulled .awg'''
            for rec in recs:
                f.write(rec)

        try:
            final_file_size = os.path.getsize(self.path)
            print '.AWG file final: %d bytes' % final_file_size
        except:
            raise Exception('Failed at reading final filezise')

    def write_wrapper(self):
        MAX_ATTEMPTS = 6
        self.write_attempt_no += 1
        try:
            self.write_dot_awg()
#        except IOError:
#            print 'AWG write failed'
#            time.sleep(0.5)
#            try:
#                self.write_dot_awg()
#            except:
#                raise Exception('AWG writing failed twice.')
        except:
            if self.write_attempt_no < MAX_ATTEMPTS:
                print 'Failed AWG write wrapper, retrying. (%d/%d) ' % (self.write_attempt_no, MAX_ATTEMPTS)
                self.write_wrapper()
            else:
                self.write_attempt_no = 0
                raise Exception('AWG writing failed six times.')

    def make_timestamp(self):
        return np.array([
            2014, 6, 3, 24, 16, 8, 11, 0
        ], dtype=np.uint16)

    def check_seq_length(self):
        val = int(self.awg.ask('SEQ:LENG?'))
        print 'NUM LOADED SEQS: %s' % (val,)
        if val == 0:
            raise Exception('AWG is reporting seq len = 0')

#if __name__ == '__main__':
#    import sys
#    sys.path.append(r"C:\pythonLab\pulseseq\pulseseq")
#    import pulselib
#    from sequencer import *
#
#
#    r = pulselib.GaussianRotation(100,200,200,0,1)
#    s = Sequence()
#    for angle in np.linspace(-2*np.pi,2*np.pi,50):
#        s.append(Trigger(250))
#        s.append(Constant(250, 1, chan='2m2'))
#        s.append(Repeat(Constant(250,1,chan=3),5))
#        s.append(r(angle, 0.0))
#    s = Sequencer(s)
#    seqs = s.render()
#
#    a = Dot_AWG_Load(seqs, awg = awg1, path='Z:\\Jacob\\temp.awg')
#    a.load_seqs()


