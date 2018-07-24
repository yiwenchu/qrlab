'''
Jacob Blumoff 6/24/2014
'''

import numpy as np
import struct
import types
import time
import hashlib
import logging

import glob
import os

import config

def clear_dot_awgs(path=None, delete='all'):#r'Z:\_AWG\Stabilizer'):
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
            logging.info('Deleting (%d/%d): %s' % (idx+1,len(dalist),item))
    pass

NULL = chr(0)

'''Make sure the AWG has the path loaded, and that it's mounted as
the same drive letter.'''

class Dot_AWG_Load():
    def __init__(self, seqs, path=None, path_awg=None, awg=None):
        '''At a high level this will only know about one AWG.
        Each seqs will already be predivided into the relevant channels
        '''
        self.awg = awg
        self.seqs = seqs
        td_path = str(int(time.time()*1000)) + '.awg'
#        self.path = path.split('.awg')[0] + td_path
        self.path = os.path.join(path.split('.awg')[0], td_path)
        self.path_awg = os.path.join(path_awg.split('.awg')[0], td_path)

        self.rl = [] #initial record list
        self.loaded_wfs = [] #List of names of loaded wfs
        self.admin_duty = 1 #Triggering, looping, repeating:  only done once

        self.load_attempt_no = 0#This is hacky.

#    def move_file(self, src, destination, timeout=500e3):
#        if not os.path.isfile(src):
#            raise Exception('awg_files: File not found: %s' % (src))
#        file_size = os.path.getsize(self.path)
#        shutil.move(src, destination)
#        start_time = time.time()
#        while (os.path.getsize(destination) != file_size):
#            time.sleep(1)
#            if time.time() - start_time > timeout:
#                raise Exception('awg_file: file move timeout exceeded')
#        os.remove(src)
#        logging.info('awg_files: completed move from %s to %s' % (src, destination))

    def load_seqs(self, delay_override=2400e3, verbose=False):
        MAX_ATTEMPTS = 6
        '''The main routine'''
        self.parse_seqs()

        self.load_attempt_no += 1
        try:
            logging.info('awg_files - %s: starting load_seqs' % (self.awg.get_name()))

#            self.path = self.server_path
            self.pull_stubborn_params()
#            logging.info('awg_files - %s: finished pull_stubborn_params' % (self.awg.get_name()))
            self.clear_awg_and_pull_data(delay_override)
#            self.move_file(self.server_path, self.local_path)
#            self.path = self.local_path
#            logging.info('awg_files - %s: finished clear_awg_etc' % (self.awg.get_name()))
            self.write_attempt_no = 0

#            logging.info('awg_files - %s: starting to write .awg file' % (self.awg.get_name()))
            self.write_wrapper()

            logging.info('awg_files - %s: starting .awg file load' % (self.awg.get_name()))
            self.awg.load_dot_awg(self.path_awg)
            if delay_override:
                self.awg.wait_getID(delay=delay_override)
            self.restore_stubborn_params()

            self.check_seq_length()
        except Exception as e:
            if self.load_attempt_no < MAX_ATTEMPTS:
                self.awg.get_id(timeout=300e3)
                msg = 'Failed AWG load, big loop, retrying. (%d/%d) [%s]' % (self.load_attempt_no, MAX_ATTEMPTS, e)
                logging.info(msg)
                self.load_seqs(delay_override = delay_override, verbose=verbose)
            else:
                time.sleep(2) #Added 4/20/15 jacob
                logging.error('ABORTING, LOAD FAILURE: %s' % e)
                raise Exception('%s load failed' % self.awg.get_name())

    def clear_awg_and_pull_data(self, delay):
        '''Delete all of the waveform and sequence memory, save a .awg
        file that contains only the device settings.'''
        self.awg.clear_sequence()
#        logging.info('clear_awg_and_pull_data -1')
        self.awg.delete_all_waveforms(wait=delay) #wait = timeout.  Deleting
                                                  #large expts can take a while.
#        logging.info('clear_awg_and_pull_data -2')        
        self.awg.wait_done()
#        logging.info('clear_awg_and_pull_data -3')        
        self.awg.pull_dot_awg(self.path_awg)
#        logging.info('clear_awg_and_pull_data -4')
        self.awg.wait_done()
#        logging.info('clear_awg_and_pull_data -5')

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

    def compose_record(self, name, data):
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

    def write_dot_awg(self, write_timeout=500e4):
        recs = [' ']*len(self.rl)
        for idx,[name,data] in enumerate(self.rl):
            recs[idx] = self.compose_record(name, data)
        try:
            logging.info('%s: looking for .AWG file at %s' % (self.awg.get_name(), self.path))
            print self.path, self.path_awg
            initial_file_size = os.path.getsize(self.path)
            logging.info('.AWG file init: %d bytes' % initial_file_size)
        except:
            raise Exception('Failed at reading init filesize - Usually means path disagreement')

        with open(self.path,'ab') as f:
            '''Note the 'a' for appending to the pulled .awg'''
            for rec in recs:
                f.write(rec)   

        # see if the file is done writing
        time_start = time.time()
        while (time.time() < (time_start + write_timeout)):
            try:
                final_file_size0 = os.path.getsize(self.path)
                time.sleep(0.01)
                final_file_size = os.path.getsize(self.path)
            except:
                raise Exception('Failed at reading final filesize')

            if final_file_size0 == final_file_size:
                logging.info('.AWG file final: %d bytes' % final_file_size)
                return
            logging.info('not finished writing file')

#        try:
#            final_file_size = os.path.getsize(self.path)
#            logging.info('.AWG file final: %d bytes' % final_file_size)
#        except:
#            raise Exception('Failed at reading final filezise')

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
        except Exception as e:
            if self.write_attempt_no < MAX_ATTEMPTS:
                logging.info('Failed %s write wrapper, retrying: %s (%d/%d) ' % (self.awg.get_name(), e, self.write_attempt_no, MAX_ATTEMPTS))
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
        logging.info('NUM LOADED SEQS: %s' % (val,))
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


