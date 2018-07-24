# Pulse sequencer
#
# TODO:
# - drop time-dependent pulse compatibility; this makes rendering much harder.
#   althoug it is also nice to have and maybe not terribly bad.
# - Allow adding and multiplying of properly aligned sequences. If they are
#   not aligned they can be by using Combined.generate()

import types
import numpy as np
import copy
import hashlib
import logging
import convolve
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 'small'

#########################################
# Constants
#########################################

CHAN_ANY    = -1

ALIGN_LEFT    = 0
ALIGN_RIGHT   = 1
ALIGN_CENTER  = 2

PAD_LEFT      = 0
PAD_RIGHT     = 1
PAD_BOTH      = 2
PAD_CENTER    = 2

MINLEN        = 250

IGNORE        = 0
WARN          = 1
RAISE         = 2

DEBUG         = False

FLOW_INS_LEN  = 32
FLOW_PARAMS   = set((
    'trigger',
    'jump',
    'label',
    'inslength',
    'callreturn',
    'call',
))

#########################################
# Helper functions
#########################################

def merge_arrays(ar1, ar2):
    for i in ar2:
        if i not in ar1:
            ar1.append(i)
    return ar1

def chars_in_str(chars, s):
    for ch in chars:
        if ch in s:
            return True
    return False

def is_delay(el):
    if isinstance(el, Pulse) and el.name.startswith('delay'):
        return True
    return False

def is_delay1(el):
    if isinstance(el, Pulse) and el.name == 'delay1':
        return True
    return False

class Sequencer:
    '''
    The Sequencer object provides functions to render a sequence properly.
    (the sequence can consist of one or more Sequence objects).
    The steps it takes:
    - resolve timings
    - render pulses (including channel delays)
    - render marker channels
    - perform sideband modulation
    '''

    def __init__(self, s=None, flatten=False, minlen=MINLEN, ch_align=True):
        self.pulse_data = {}
        self.chan_props = {}
        self.minlen = minlen
        self._ssb_list = []
        self._marker_chans = {}
        self._req_chans = []
        self._delays = {}
        self.ch_align = ch_align
        self._master_chans = []
        self._slave_triggers = []
        self._flatten_waveforms = flatten
        self._convolve_channels = []

        self.slist = []
        if type(s) in (types.ListType, types.TupleType):
            for el in s:
                self.add_sequence(el)
        elif s is not None:
            self.add_sequence(s)

        # Generate delay1 pulse
        Delay(1)

    def add_sequence(self, s):
        self.slist.append(s)

    def add_ssb(self, ssb):
        '''
        Add an SSB object that will be used to sideband modulate after
        generating the sequence.
        '''
        if ssb is None:
            return
        if ssb not in self._ssb_list:
            self._ssb_list.append(ssb)

    def add_required_channel(self, chan):
        '''
        Add a channel that we should generate even if there is no activity.
        (which means it will include an empty sequence).
        '''
        if type(chan) in (types.ListType, types.TupleType):
            self._req_chans.extend(chan)
        else:
            self._req_chans.append(chan)

    def add_marker(self, markchan, activechan, pre=8, post=0, ofs=None, bufwidth=None):
        '''
        Add a marker channel <markchan> that is to be generated upon activity
        on <activechan>. <pre> and <post> respectively indicate how many
        samples should be generated before and after channel activity.
        Alternatively <ofs> and <bufwidth> can be specified, which will result
        in a marker that is shifted in time by <ofs> with respect to the
        channel activity and wider by 2*<bufwidth> samples.
        '''

        # Determine width and offset from <pre> and <post>
        if ofs is None:
            bufwidth = round((pre + post) / 2.0)
            ofs = round((post - pre) / 2.0)

        if markchan not in self._marker_chans:
            self._marker_chans[markchan] = dict(channels=[], ofs=ofs, bufwidth=bufwidth)
        if activechan not in self._marker_chans[markchan]:
            self._marker_chans[markchan]['channels'].append(activechan)

    def add_channel_delay(self, chan, delay):
        '''
        Add a delay for a specific channel (can be negative)
        '''
        self._delays[chan] = delay

    def add_master_channel(self, chan):
        '''
        Designate channel <chan> (can be list or tuple as well) as 'master'
        channels, i.e. add a delay at the start of each channel that allows
        to trigger 'slave' channels on another AWG.
        '''
        if type(chan) in (types.ListType, types.TupleType):
            self._master_chans.extend(chan)
        else:
            self._master_chans.append(chan)

    def add_slave_trigger(self, chan, delay):
        '''
        Play a trigger on channel <chan> that gives the slave a time <delay>
        to start playing its sequences
        '''
        self._slave_triggers.append((chan, delay))

    def set_flatten(self, flatten):
        self._flatten_waveforms = flatten

    def add_convolution(self, chan, kernel):
        '''
        Specifies <chan> to be convolved with <kernel> after the sequence is rendered.
        The whole sequence is automatically flattened per trigger prior to convolution.
        '''
        self.set_flatten(True)
        self._convolve_channels.append((chan, kernel))

    def debug_seqs(self, seqs, msg, debug):
        if not debug:
            return
        print 'Sequence %s' % (msg,)
        for seq in seqs.values():
            seq.print_seq()

    def get_channels(self):
        chs = set()
        for s in self.slist:
            chs |= set(s.get_channels())
        chs |= set(self._req_chans)
        return list(chs)

    def split_at_trigger(self):
        '''
        Split the list of sequences to be rendered in sub-sequences that start
        with a trigger.
        '''

        ret = []
        for s in self.slist:
            istart = 0
            while istart < len(s):
                iend = istart + 1
                while iend < len(s) and not s[iend].get_trigger():
                    iend += 1
                seq = Sequence(s[istart:iend])
                ret.append(Combined(seq, ch_align=self.ch_align, ch_delays=self._delays))
                istart = iend
        return ret

    def add_master_slave_triggers(self, seqs):
        '''
        Add master/slave delays and triggers to a rendered sequence.
        '''

        # Nothing to do
        if len(self._slave_triggers) == 0 and len(self._master_chans) == 0:
            return

        # Determine max delay, we add 100 ns because the Tektronix AWGs seem
        # to play garbage if a marker channel is thrown high at the start of
        # a sequence.
        delays = np.array([delay for chan, delay in self._slave_triggers])
        trigchans = [chan for chan, delay in self._slave_triggers]
        maxdelay = np.max(delays) + 100

        # Choose reference channel for aligned empty sequence
        if len(self._master_chans) != 0:
            refchan = self._master_chans[0]
        else:
            refchan = seqs.keys()[0]

        # Add triggers on requested channels
        for chan, delay in self._slave_triggers:
            if chan not in seqs:
                seqs[chan] = seqs[refchan].aligned_empty_sequence(chan=chan)
            data = np.zeros(maxdelay)
            data[maxdelay-delay:] = 1
            p = Pulse('trig(%d,%d)'%(delay,maxdelay), data=data, trigger=True)
            seqs[chan].seq[0].trigger = False
            seqs[chan].prepend(p)

        # Add delay on master channels
        for chan, seq in seqs.items():
            if chan in self._master_chans and chan not in trigchans:
                seqs[chan].seq[0].trigger = False
                seq.prepend(Delay(maxdelay, fixed=True, unroll=False, trigger=True))

    def render_subseq(self, ss, chs, debug=False):
        '''
        Render a sub-sequence.
        '''

        if debug:
            print 'Rendering: %s' % (ss, )

        seqs = {}
        for ch in ss.get_channels():
            seqs[ch] = ss.resolve(0, ch)
        self.debug_seqs(seqs, 'after resolve()', debug)

        seqs = {}
        for ch in chs:
            seqs[ch] = ss.generate(0, ch)
        self.debug_seqs(seqs, 'after generate()', debug)

        # Sanitize delays
        if self.minlen != 0:
            for seq in seqs.values():
                seq.sanitize_delays(self.minlen)
            self.debug_seqs(seqs, 'after sanitizing delays', debug)

        # Make sure each pulse is at least <minlen> long
        for chan, seq in seqs.items():
            seq.join_small_elements(self.minlen)
        self.debug_seqs(seqs, 'after join_small_elements', debug)

        # Perform sideband modulation
        for ssb in self._ssb_list:
            ssb.modulate(seqs)

        # Generate marker channels
        for mchan, info in self._marker_chans.items():
            m = self.generate_marker(mchan, seqs, info)
            if m is not None:
                seqs[mchan] = m
        self.debug_seqs(seqs, 'after generate_marker', debug)

        # Check whether the sequences agree
        if self.ch_align and not self.sequences_aligned(seqs):
            raise Exception('Error: generated sequences are not aligned')

        # Add master/slave triggers and delays, sequences will not be aligned
        self.add_master_slave_triggers(seqs)
        self.debug_seqs(seqs, 'after add_master_slave_triggers', debug)

        return seqs

    #TODO: double-check order of delay / sideband modulation
    def render(self, debug=False):
        chs = self.get_channels()
        ss = self.split_at_trigger()    # Get sub-sequences

        seqs = {}
        for el in ss:
            new_seqs = self.render_subseq(el, chs, debug)
            for ch, seq in new_seqs.items():
                if ch not in seqs:
                    seqs[ch] = seq
                else:
                    seqs[ch].append(seq)

        for ch in seqs.keys():
            seqs[ch].join_sequences()

        # Generate sequence element addresses
        for chan, seq in seqs.items():
            seq.generate_addresses()

        # Add deconvolution with provided kernels
        if len(self._convolve_channels) != 0:
            seqs = self.flatten_sequences(seqs)
            for ch, kernel in self._convolve_channels:
                seqs = self.convolve_channel(ch, seqs, kernel)

        return seqs

    def sequences_aligned(self, seqs):
        '''
        Check whether the sequences in dictionary <seqs> are compatible.
        '''
        chs = seqs.keys()
        for ch in chs[1:]:
            if not seqs[chs[0]].is_aligned(seqs[ch]):
                return False
        return True

    def get_sorted_chans(self, seqs):
        chans = seqs.keys()
        chans.sort(key=str)
        return chans

    def print_seqs(self, seqs):
        chans = self.get_sorted_chans(seqs)
        for i, ch in enumerate(chans):
            seq = seqs[ch]
            seq.print_seq()

    def plot_seqs(self, seqs):
        f = plt.figure()
        chans = self.get_sorted_chans(seqs)
        for i, ch in enumerate(chans):
            seq = seqs[ch]
            seq.plot_seq(fig=f, subplot=(len(seqs),1,i+1))

    def extract_repeat(self, seqs, i_el, side, N=1):
        '''
        Extract an element from a repeated Pulse in all channels in <seqs>.
        Place it to the left when <side> is left, otherwise to the right.
        '''
        for seq in seqs.values():
            seq.extract_repeat(i_el, side, N=N)

    marker_n = 0
    def add_to_marker(self, outseq, i_el, i_pulse, activity, side=None):
        '''
        Add activity to a marker channel element.
        '''
        seqel = outseq.seq[i_el]
        if seqel.name.startswith('delay'):
            data = np.zeros([len(seqel.data),])
            name = '%s_p%d' % (outseq.chan, self.marker_n)
            self.marker_n += 1
            ret = 1
        else:
            data = seqel.data
            name = seqel.name
            ret = 0

        # Add activity
        if side == 'left':
            data[:len(activity)] += activity
        elif side == 'right':
            data[-len(activity):] += activity
        else:
            data += activity
        data = (data > 0).astype(int)

        # Set sequence element
        outseq.seq[i_el] = Pulse(name, data, repeat=seqel.repeat, trigger=seqel.trigger, chan=seqel.chan)

        return ret

    def generate_marker(self, mchan, seqs, info):
        '''
        Generate marker channel <mchan> for sequences <seqs>.
        See add_marker for explanation of parameters.
        '''

        i_pulse = 0
        ofs = info['ofs']
        bufwidth = info['bufwidth']
        outseq = None
        win = np.ones([2*bufwidth,])

        # At which offsets in the convolved data the current pulse starts and
        # ends, and at which side of the current pulse it should be added.
        pulse_start = max(0, bufwidth - ofs)
        pulse_end = min(-1, -bufwidth - ofs + 1)
        pulse_side = 'right'
        if pulse_end == -1:
            pulse_side = 'left'

        for ch in info['channels']:
            if ch not in seqs:
                continue

            if outseq is None:
                outseq = seqs[ch].aligned_empty_sequence(chan=mchan)
                seqs[mchan] = outseq

            i_el = 0
            while i_el < len(seqs[ch]):

                # Determine activity and extend using window
                el = seqs[ch].seq[i_el]
                new_activity = (el.data != 0).astype(np.int8)
                new_activity = np.convolve(new_activity, win)
                new_activity = (new_activity > 0).astype(np.int8)

                prev_pulse = new_activity[:pulse_start]
                cur_pulse = new_activity[pulse_start:pulse_end]
                next_pulse = new_activity[pulse_end:]

                # Put in previous pulse
                if pulse_start != 0 and np.count_nonzero(prev_pulse) > 0:
                    if i_el == 0:
                        logging.warning('Unable to extend marker before sequence start')
                    else:
                        if outseq.seq[i_el-1].repeat > 1:
                            self.extract_repeat(seqs, i_el-1, 'right')
                            i_el += 1
                        i_pulse += self.add_to_marker(outseq, i_el-1, i_pulse, prev_pulse, 'right')

                # Put in current pulse
                if np.count_nonzero(cur_pulse) > 0:
                    i_pulse += self.add_to_marker(outseq, i_el, i_pulse, cur_pulse, pulse_side)

                # Put in next pulse
                if pulse_end != -1 and np.count_nonzero(next_pulse) > 0:
                    if i_el + 1 == len(outseq.seq):
                        logging.warning('Unable to extend marker after sequence end')
                    elif outseq.seq[i_el+1].get_trigger():
                        logging.warning('Unable to extend marker to right due to trigger, add Delay')
                    else:
                        if outseq.seq[i_el+1].repeat > 1:
                            self.extract_repeat(seqs, i_el+1, 'left')
                        i_pulse += self.add_to_marker(outseq, i_el+1, i_pulse, next_pulse, 'left')

                i_el += 1

        return outseq

    def flatten_sequences(self, seqs):
        '''
        Flatten <seqs> in place to one waveform per trigger.
        '''
        for ch, seq in seqs.items():
            seq.flatten()
        return seqs

    def convolve_channel(self, ch, seqs, kernel):
        '''
        Convolves the waveform in <ch> with <kernel> for every element in <seqs>
        '''
        for i, pulse in enumerate(seqs[ch].seq):
            data = convolve.perform_convolution(pulse.data, kernel)
            new_pulse = Pulse('convolved'+str(ch)+'-'+str(i),
                              data, ch=ch, trigger=pulse.get_trigger())
            seqs[ch].seq[i] = new_pulse
        return seqs

class Instruction(object):
    '''
    General class for sequence instructions. In the end, all of them should provide
    a function 'generate' which resolves to a list of instructions that can be mapped
    to hardware, which at the moment is only the Pulse instruction.
    '''

    def __init__(self, **kwargs):
        self.params = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        return 'Instruction %s' % self.__class__

    def __repr__(self):
        return self.__str__()

    def set_params(self, key=None, val=None, **kwargs):
        '''
        Set a keyword argument
        '''
        if key:
            kwargs[key] = val
        for k, v in kwargs.items():
            self.params[k] = v
            setattr(self, k, v)

    def get_flow_params(self):
        params = FLOW_PARAMS & set(self.params.keys())
        return {k: self.params[k] for k in params}

    def has_flow_params(self):
        return len(FLOW_PARAMS & set(self.params.keys())) > 0

    def resolve(self, now, chan):
        '''
        Resolve timing of this instruction
        '''
        return self.generate(now, chan)

    def generate(self, now, chan):
        '''
        Function to go from an abstract Instruction to a Sequence object
        containing elements that can be represented by hardware
        '''
        return Sequence([], chan=chan)

    def get_data(self):
        '''
        Return the data associated with this pulse. Note that this
        does not care about the channel specification. It is mostly
        present to generate for example derivative pulses.
        '''
        return np.array([])

    def get_name(self):
        return 'ins?'

    def get_length(self, now=None):
        '''
        Get the length of this instruction; zero means undefined.
        '''
        return len(self.get_data())

    def get_trigger(self):
        if hasattr(self, 'trigger'):
            return self.trigger
        else:
            return False

    def get_channels(self):
        return []

    def get_used_pulses(self):
        return {}

#######################################
# Core instructions
#######################################

class Sequence(Instruction):
    '''
    The sequence class is an container for a list of Instructions.

    It provides some operations to combine, split or simplify these
    elements such that a representation compatible with the present
    hardware can be obtained.

    Several sequences can be combined in the Sequencer object.

    Sequence inherits from Instruction, which means it has the generate()
    function to reduce it to a Sequence for a particular channel.

    The addition operator is overloaded to concatenate Instructions or
    Sequences.
    The multiplication operator is overloaded to simultaneously play
    Instructions or Sequences by combining them in a MultiChannelPulse.
    '''

    PRINT_EXTENDED = False

    def __init__(self, seq=None, chan=None, join=False, repeat=1, **kwargs):
        self.channel_map = {}

        # Propagate trigger to this sequence
        try:
            kwargs['trigger'] = kwargs.get('trigger', False)
            kwargs['trigger'] |= seq.get_trigger()
        except:
            pass

        if isinstance(seq, Sequence):
            # Is there a problem with repeats here
            self.seq = seq.seq
            join = seq.join
        elif seq is None:
            self.seq = []
        elif type(seq) not in (types.ListType, types.TupleType):
            self.seq = [seq,]
        else:
            self.seq = seq

        self.chan = chan
        self.join = join
        self.repeat = repeat

        super(Sequence, self).__init__(**kwargs)

    def join_sequences(self):
        '''
        Absorb sub-sequences directly in this sequence object (in place)
        '''
        i = 0
        while i < len(self.seq):
            el = self.seq[i]
            if isinstance(el, Sequence):
                el.join_sequences()

            if isinstance(el, Sequence) and (self.join or self.join == el.join or len(el.seq) == 1):
                del self.seq[i]
                for el2 in el.seq:
                    self.seq.insert(i, el2)
                    i += 1
            else:
                i += 1
        self.merge_delays()

    def check_elements(self, seq):
        if type(seq) not in (types.ListType, types.TupleType):
            raise Exception('Sequence should be a tuple or list')
        i = 0
        while i < len(seq):
            if seq[i] == None:
                del seq[i]
                print 'Warning: removing None element from sequence'
            else:
                i += 1

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return self.seq[i]

    def __setitem__(self, i, el):
        self.seq[i] = el

    def __delitem__(self, i):
        del self.seq[i]

    def get_data(self):
        ar = np.array([])
        for el in self.seq:
            eldata = el.get_data()
            if hasattr(el, 'repeat') and el.repeat > 1:
                eldata = np.tile(eldata, el.repeat)
            ar = np.concatenate((ar, eldata))
        return ar

    def get_channels(self):
        chs = []
        for el in self.seq:
            elchs = el.get_channels()
            chs = merge_arrays(chs, elchs)
        return chs

    def get_name(self):
        if len(self.seq) == 1:
            if getattr(self.seq[0], 'repeat', 1) == 1:
                return self.seq[0].get_name()
            else:
                return '%d%s' % (self.seq[0].repeat, self.seq[0].get_name())
        if self.join:
            name = 'join('
        else:
            name = 'seq('
        for i, el in enumerate(self.seq):
            if i != 0:
                name += ','
            if hasattr(el, 'repeat') and el.repeat > 1:
                name += ('%d'%el.repeat) + el.get_name()
            else:
                name += el.get_name()
        name += ')'
        return name

    def __str__(self):
        return 'Sequence(join=%s, chan=%s, name=%s, trigger=%s)' % (self.join, self.chan, self.get_name(), self.trigger)

    def get_trigger(self):
        return self.trigger or (len(self.seq) > 0 and self.seq[0].get_trigger())

    def __iadd__(self, rval):
        self.append(rval)
        return self

    def __add__(self, rval):
        seq = copy.deepcopy(self.seq)
        seq.append(rval)
        return Sequence(seq)

    def append(self, ins):
        if ins is None:
            return
        if type(ins) in (types.ListType, types.TupleType):
            ins = Sequence(ins)
        self.seq.append(ins)

    def prepend(self, ins):
        if ins is None:
            return
        if type(ins) in (types.ListType, types.TupleType):
            ins = Sequence(ins)
        self.trigger |= ins.get_trigger()
        self.seq.insert(0, ins)

    def get_length(self, now=None):
        t = 0
        for i in self.seq:
            t += i.get_length(now)
        return t

    def is_aligned(self, other):
        '''
        Determine whether this sequence is aligned with another.
        The sequences must have been resolved, i.e. only consist of pulses.
        '''
        if not isinstance(other, Sequence):
            return False
        if len(self.seq) != len(other.seq):
            return False
        for i in range(len(self.seq)):
            if self.seq[i].get_length() != other.seq[i].get_length():
                return False
            if getattr(self.seq[i], 'repeat', 1) != getattr(other.seq[i], 'repeat', 1):
                return False
        return True

    def resolve(self, now, chan):
        self.check_elements(self.seq)
        s = Sequence(trigger=self.get_trigger(), join=self.join, chan=chan)
        for p in self.seq:
            rseq = p.resolve(now, chan)
            s.append(rseq)
        s.join_sequences()
        return s

    def generate(self, now, chan):
        '''
        Return a new Sequence object for this particular channel.
        '''
        seq_out = Sequence(chan=chan, join=self.join, **self.get_flow_params())
        bufs = []
        for p in self.seq:
            seq_add = p.generate(now, chan)
            now += seq_add.get_length(now)
            if self.join:
                bufs.append(seq_add.get_data())
            seq_out += seq_add

        if self.join:
            buf_out = np.concatenate(bufs)
            if seq_out.delays_only() or np.count_nonzero(buf_out) == 0:
                p = Delay(len(buf_out), unroll=False, fixed=True, **self.get_flow_params())
            else:
                name = seq_out.get_name()
                p = Pulse(name, buf_out, chan=chan, **self.get_flow_params())
            return p.generate(now, chan)

        seq_out.join_sequences()

        # Propagate label
        if getattr(self, 'label', None):
            seq_out.seq[0].set_params(label=self.label)

        return seq_out

    def delays_only(self):
        '''
        Return whether this Sequence consists only of Delay items.
        '''
        for el in self.seq:
            if isinstance(el, Pulse):
                if el.name.startswith('delay'):
                    continue
            if isinstance(el, Sequence) and el.delays_only():
                continue
            return False
        return True

    def unroll_delays(self):
        '''
        Unroll all delays
        '''
        seqout = []
        for el in self.seq:
            if is_delay(el):
                if len(seqout) == 0 or not is_delay(seqout[-1]) or el.get_trigger() :
                    seqout.append(Delay(el.get_length(), trigger=el.get_trigger()))
                elif not el.get_trigger():
                    seqout[-1].repeat += el.get_length()
            else:
                seqout.append(el)
        self.seq = seqout

    def merge_delays(self):
        '''
        Merge neighboring unrolled delays (in place)
        '''
        i = 1
        while i < len(self.seq):
            el = self.seq[i]
            if is_delay1(el) and is_delay1(self.seq[i-1]) and \
                    not el.has_flow_params():
#                    not (self.seq[i-1].has_flow_params() or el.has_flow_params()):
                del self.seq[i]
                self.seq[i-1].repeat += el.repeat
            else:
                i += 1

    ###################################################
    # Functions to manipulate single channel sequences
    ###################################################

    def time_action_at(self, tgt_t, idx=0):
        '''
        If there is an active pulse at time <tgt_t> from index <idx> return
        how much longer it lasts.
        '''

        # If we're supposed to join this sequence, return length from tgt_t
        if self.join:
            return self.get_length() - tgt_t

        t = 0
        while idx < len(self.seq):
            dt = self.seq[idx].get_length()
            if tgt_t >= t and tgt_t <= t + dt:
                if isinstance(self.seq[idx], Pulse) and self.seq[idx].name == 'delay1':
                    return 0
                else:
                    return (t + dt - tgt_t)
            t += dt
            idx += 1
        return 0

    def consume_up_to(self, tgt_t, unroll=False, fixed=True):
        '''
        Consume the current sequence up to <tgt_t> from the start.
        Return a Sequence with join=True and modify the current object
        in place.
        '''
        ret = Sequence(join=not unroll)

        t = 0
        while len(self.seq) and t < tgt_t:
            plen = self.seq[0].get_length()
            if plen < 0:
                raise ValueError('Negative delay!')
            repeat = getattr(self.seq[0], 'repeat', 1)
            trig = self.get_trigger()
            self.trigger = False
            self.seq[0].trigger = False

            if isinstance(self.seq[0], Pulse) and self.seq[0].name == 'delay1' and t + plen >= tgt_t:
                dt = tgt_t - t
                self.seq[0].repeat -= dt
                if self.seq[0].repeat == 0:
                    del self.seq[0]
                if dt > 0:
                    ret.append(Delay(dt, unroll=unroll, fixed=fixed, trigger=trig))
#                print '  remaining: %s, ret: %s' % (self.seq, ret)
                return ret

            # <tgt_t> should be chosen such that full copies can be made of
            # non-delay pulses
            else:
                t += plen
                if isinstance(self.seq[0], Pulse) and self.seq[0].name == 'delay1':
                    ret.append(Delay(plen, unroll=unroll, fixed=fixed, trigger=self.seq[0].get_trigger()))
                else:
#                    print 'Copying %s, %s' % (self.seq[0], self.seq[0].name)
                    ret.append(self.seq[0])
                del self.seq[0]

#        print '  remaining: %s, ret: %s' % (self.seq, ret)
        return ret

    def print_seq(self):
        print 'Channel %s' % (self.chan,)
        if self.PRINT_EXTENDED:
            layout = '%1s%6s%2s  %6s  %4s  %4s  %-20s  %8s  %-16s  %s'
            print layout % ('', 'start', '', 'end', 'n', 'len', 'name', 'addr', 'label', 'args')
        else:
            layout = '%1s%6s%2s    %6s  %4s   %4s   %s'
            print layout % ('', 'start', '', 'end', 'n', 'len', 'name')
        t = 0
        for i_el, el in enumerate(self.seq):
            mark = ''
            if (i_el % 5) == 0:
                mark = '+'
            flags = ''
            if el.get_trigger():
                flags += 'T'
            if hasattr(el, 'unroll') and not el.unroll:
                flags += 'U'

            if getattr(el, 'name', None) in Pulse.pulse_data:
                plen = len(Pulse.pulse_data[el.name])
            else:
                plen = 0

            tend = max(t, t + el.repeat * plen - 1)
            if self.PRINT_EXTENDED:
                print layout % (mark, t, flags, tend, getattr(el, 'repeat', 0), plen, el.name, getattr(el, 'address', 0), getattr(el, 'label', ''), getattr(el, 'params', ''))
            else:
                print layout % (mark, t, flags, tend, getattr(el, 'repeat', 0), plen, el.name)
            t += el.repeat * plen

        if self.PRINT_EXTENDED:
            print layout % ('', '', '', '', '', '', 'done.', '', '', '')
        else:
            print layout % ('', '', '', '', '', '', 'done.')

    def plot_seq(self, style='ks', fig=None, subplot=(1,1,1)):
        ymax = 1.05
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(*subplot)
        ax.set_title('Channel %s' % self.chan)
        t = 0
        for el in self.seq:
            if el.get_trigger():
                plt.plot([t, t], [0, 1], 'r', linewidth=1)
            if isinstance(el, Pulse) and el.name in Pulse.pulse_data:
                dt = len(Pulse.pulse_data[el.name])
                if not el.name.startswith('delay'):
                    for i in range(el.repeat):
                        plt.plot(np.arange(t+i*dt,t+(i+1)*dt), Pulse.pulse_data[el.name])
                ymax = max(ymax, np.max(np.abs(Pulse.pulse_data[el.name]*1.05)))
            else:
                dt = 0
            t += el.repeat * dt

        ax.set_xlim(-2, t+2)
        ax.set_ylim(-ymax, ymax)
        return ax

    def get_used_pulses(self):
        ret = {}
        for el in self.seq:
            try:
                ret.update(el.get_used_pulses())
            except:
                pass
        return ret

    def sanitize_delays(self, minlen):
        self.merge_delays()
        seq_out = []
        for el in self.seq:
            if el.name == 'delay1':
                # OK to have zero delay items if we don't want to trigger
                if el.repeat == 0 and not el.get_trigger():
                    continue

                trig = el.get_trigger()

                # Number of <minlen> unit delays
                nrep = int(np.floor(el.repeat / minlen))
                nremain = el.repeat % minlen
                if nremain != 0 and nrep > 0:
                    nrep -= 1
                    nremain += minlen

                # Do not repeat a delay with a trigger
                if nrep > 0 and trig:
                    seq_out.append(Delay(minlen, unroll=False, repeat=1, trigger=True))
                    nrep -= 1
                    trig = False

                # Remaining minlen unit delays (should never have to trigger here)
                if nrep > 0:
                    seq_out.append(Delay(minlen, unroll=False, repeat=nrep, trigger=False))

                # Remaining minlen + something delay
                if nremain > 0:
                    seq_out.append(Delay(nremain, unroll=False, trigger=trig))

            else:
                seq_out.append(el)
        self.seq = seq_out

    def unroll_small_repeats(self, minlen):
        '''
        In-place unroll of repeats smaller than <minlen>.
        '''

        self.join_sequences()

        j = 0                           # Number of extra pulses we've inserted
        seqout = copy.copy(self.seq)    # Can't loop over something we modify
        for i, el in enumerate(self.seq):
            if el.repeat > 1 and len(el.data) < minlen:

                # Determine new repeat count
                n_to_unroll = int(np.ceil(float(minlen) / len(el.data)))
                new_repeat_count = (el.repeat / n_to_unroll)
                remainder = (el.repeat % n_to_unroll)
                if remainder != 0:
                    new_repeat_count -= 1
                    remainder += n_to_unroll

                if new_repeat_count > 0:
                    repeatdata = np.tile(el.data, n_to_unroll)
                    if np.count_nonzero(repeatdata) == 0:
                        repeatpulse = Delay(len(repeatdata), repeat=new_repeat_count, unroll=False, fixed=True)
                    else:
                        repeatname = "%dx%s" % (n_to_unroll, el.name)
                        repeatpulse = Pulse(repeatname, repeatdata, repeat=new_repeat_count)
                    seqout[i+j] = repeatpulse

                # Delete original pulse
                else:
                    del seqout[i+j]
                    j -= 1

                if remainder != 0:
                    remaindata = np.tile(el.data, remainder)
                    if np.count_nonzero(remaindata) == 0:
                        remainpulse = Delay(len(remaindata), repeat=1, unroll=False, fixed=True)
                    else:
                        remainname = "%dx%s" % (remainder, el.name)
                        remainpulse = Pulse(remainname, remaindata)
                    j += 1
                    seqout.insert(i+j, remainpulse)

        self.seq = seqout

    def join_small_elements(self, minlen):
        '''
        Join adjacent elements where necessary to obtain pulses longer than
        <minlen>. In-place operation.
        '''
        i = 0
        while i < len(self.seq):
            curlen = self.seq[i].get_length()

            # We're good
            if curlen >= minlen:
                i += 1
                continue

            # Try to find pulses to join in both directions, first try non-
            # repeating ones. Try to pick smallest element to make pulse
            # longer than minlen. Stop at triggers.
            # After this loop we want to join elements [i:j+1]
            j = i
            for allow_unroll in False, True:

                while curlen < minlen:
                    # Try on left (if no trigger in this element)
                    if i > 0 and not self.seq[i].get_trigger() and (allow_unroll or self.seq[i-1].repeat == 1):
                        prevlen = self.seq[i-1].get_length() / self.seq[i-1].repeat
                    else:
                        prevlen = 0

                    # Try on right (if not trigger on next element)
                    if j < len(self.seq)-1 and not self.seq[j+1].get_trigger() and (allow_unroll or self.seq[j+1].repeat == 1):
                        nextlen = self.seq[j+1].get_length() / self.seq[j+1].repeat
                    else:
                        nextlen = 0

                    # Both extensions possible, choose one of them
                    # The next pulse if it takes us over the minimum length
                    # and the previous one doesn't, otherwise the previous one
                    if prevlen != 0 and nextlen != 0:
                        dmin = minlen - curlen
                        if nextlen >= dmin and prevlen < dmin:
                            prevlen = 0
                        else:
                            nextlen = 0

                    # Only possible to extend before
                    if prevlen != 0 and nextlen == 0:
                        Nneeded = int(np.ceil((minlen - curlen) / float(prevlen)))
                        if Nneeded >= self.seq[i-1].repeat:
                            i -= 1
                        else:
                            self.extract_repeat(i-1, 'right', N=Nneeded)
                        curlen += self.seq[i].get_length()

                    # Only possible to extend after
                    elif prevlen == 0 and nextlen != 0:
                        Nneeded = int(np.ceil((minlen - curlen) / float(nextlen)))
                        if Nneeded < self.seq[j+1].repeat:
                            self.extract_repeat(j+1, 'left', N=Nneeded)
                        j += 1
                        curlen += self.seq[j].get_length()

                    # Try next option (with unrolling repeats) or stop
                    else:
                        break

                # We're done
                if curlen >= minlen:
                    break

            # This is a problem
            if curlen < minlen:
                raise Exception("Unable to make each element longer than %d. %d available from index %d - %d" % (minlen, curlen, i, j))

            # Perform the actual join
            self.seq[i] = Join(self.seq[i:j+1], chan=self.chan)
            del self.seq[i+1:j+1]

        self.seq = self.generate(0, self.chan).seq

        # TODO: do this before or after joining?
        self.unroll_small_repeats(minlen)

    def aligned_empty_sequence(self, chan=None):
        '''
        Return an empty sequence that is aligned with this one.
        '''
        s = Sequence(chan=chan)
        for i_el, el in enumerate(self.seq):
            s.append(Delay(el.get_length()/el.repeat, unroll=False, fixed=True, repeat=el.repeat, trigger=el.get_trigger()))
        return s

    def extract_repeat(self, i_el, side, N=1):
        '''
        Extract an element from a repeated Pulse.
        Place it to the left when <side> is left, otherwise to the right.
        '''

        if self.seq[i_el].repeat <= N:
            raise Exception('Unable to extract %d repeats from element %d' % (N, i_el))

        self.seq[i_el] = copy.deepcopy(self.seq[i_el])
        self.seq.insert(i_el, copy.deepcopy(self.seq[i_el]))
        if side == 'left':
            self.seq[i_el].repeat = N
            self.seq[i_el+1].repeat -= N
        else:
            self.seq[i_el+1].repeat = N
            self.seq[i_el].repeat -= N

    def lookup_label(self, table, label, curaddr):
        if label == 'next':
            return curaddr + 1
        if label not in table:
            raise ValueError('Jump target %s not defined' % label)
        return table[label]

    def generate_addresses(self):
        # Generate addresses and store labels
        labels = {}
        for i, el in enumerate(self.seq):
            el.address = i
            if getattr(el, 'label', None):
                labels[el.label] = i

        # Add jump addresses
        for i, el in enumerate(self.seq):
            if hasattr(el, 'jump'):
                if type(el.jump) in (types.ListType, types.TupleType):
                    to_address = [self.lookup_label(labels, l, el.address) for l in el.jump]
                else:
                    to_address = self.lookup_label(labels, el.jump, el.address)
                el.set_params(to_address=to_address)

    def flatten(self):
        '''
        Flatten sequence in place to one waveform per trigger.
        '''

        trig_idx = 0
        seq_out = []
        data = None
        for pulse_idx, pulse in enumerate(self.seq):
            new_pulse_data = np.tile(pulse.data, pulse.repeat)
            if data is None:
                data = new_pulse_data
            else:
                data = np.concatenate((data, new_pulse_data))

            if pulse_idx == (len(self.seq) - 1) or self.seq[pulse_idx+1].trigger:
                pulse_name = 'ch%s_n%d' % (str(self.chan), trig_idx)
                new_pulse = Pulse(pulse_name, data, trigger=True)
                seq_out.append(new_pulse)
                trig_idx += 1
                data = None

        self.seq = seq_out

class Pulse(Instruction):

    # Class-wide pulses data, so that identical pulses get recycled
    pulse_data = {}
    RANGE_ACTION = RAISE

    def __init__(self, name, data, chan=0, overwrite=True, trigger=False, repeat=1, **kwargs):
        self.name = name
        if name in Pulse.pulse_data:
            if data is None:
                data = Pulse.pulse_data[name]
            elif overwrite:
                Pulse.pulse_data[name] = data
            elif not (Pulse.pulse_data == data).all():
                raise ValueError('Pulse %s already defined differently' % name)
        else:
            if data is None:
                logging.warning('Creating empty pulse %s', name)
            Pulse.pulse_data[name] = data

            # Only check range if pulse not defined yet
            self.check_range(data)

        self.data = data
        self.chan = chan
        self.trigger = trigger
        self.repeat = repeat
        super(Pulse, self).__init__(**kwargs)

    def rendered(self):
        return self.resolve(0, self.chan)[0]

    def check_range(self, data):
        if np.count_nonzero(np.abs(data) > 1) != 0:
            val = data[np.argmax(np.abs(data))]
            msg = 'Pulse %s contains value larger than +-1: %.03f' % (self.name, val)
            if Pulse.RANGE_ACTION == WARN:
                print msg
            elif Pulse.RANGE_ACTION == RAISE:
                raise Exception(msg)

    def __str__(self):
        return 'Pulse(%s, chan=%s, repeat=%s, trigger=%s)' % (self.name, self.chan, self.repeat, self.trigger)

    def get_channels(self):
        if self.chan != CHAN_ANY:
            return [self.chan, ]
        else:
            return []

    def get_name(self):
        return self.name

    def get_length(self, now=None):
        return self.repeat * len(self.data)

    def get_data(self):
        return self.data

    def get_area(self):
        return np.sum(self.data)

    def generate(self, now, chan):
        '''
        Return the pulse if it belows to this channel, otherwise a delay.
        '''

        # Create pulse for this channel in particular
        if chan == self.chan or self.chan == CHAN_ANY:
            p = Pulse(self.name, self.data, repeat=self.repeat, chan=self.chan, trigger=self.trigger, **self.params)
            return Sequence([p,], chan=chan)

        else:
            plen = self.get_length()
            d = Delay(plen, unroll=True, trigger=self.trigger, **self.params)
            return d.generate(now, chan)

    @staticmethod
    def get_pulse(name, n=1):
        if n == 1:
            return Pulse.pulse_data[name]
        p = Pulse.pulse_data[name]
        d = np.zeros([n*len(p),])
        for i in range(n):
            d[i*len(p):(i+1)*len(p)] = p
        return d

    @staticmethod
    def clear_pulse_data():
        Pulse.pulse_data = {}

    def get_used_pulses(self):
        return {self.name: self}

def Delay(dt, chan=CHAN_ANY, unroll=True, repeat=1, trigger=False, fixed=False, **kwargs):
    '''
    A fixed length delay.
    Comes in 3 flavours:
    - unrolled (i.e. N times delay1)
    - smart (i.e. N times delay250 + M times delay1)
    - fidex (i.e. one element delayN)
    '''

    try:
        float(dt)
    except:
        raise ValueError('Delay time should be a value, not %s' % dt)
    dt = int(round(dt))
    if dt == 0:
        return None

    if unroll:
        return Pulse('delay1', np.zeros(1), repeat=repeat*dt, chan=chan, trigger=trigger, **kwargs)

    if fixed:
        return Pulse('delay%d'%dt, np.zeros(dt), chan=chan, trigger=trigger, repeat=repeat, **kwargs)

    if dt >= 2 * MINLEN:
        s = Sequence(repeat=repeat)
        name = 'delay%d' % MINLEN
        s.append(Pulse('delay%d'%MINLEN, np.zeros(MINLEN), repeat=int(np.floor(dt/MINLEN)), chan=chan, trigger=trigger, **kwargs))
        remain = dt % MINLEN
        if remain > 0:
            s.append(Pulse('delay%d'%remain, np.zeros(remain), repeat=1, chan=chan, **kwargs))
        return s

    else:
        return Pulse('delay%d'%dt, np.zeros(dt), repeat=repeat, chan=chan, trigger=trigger, **kwargs)

def Trigger(dt=1, **kwargs):
    '''
    A trigger instruction, which is actually a 1 time unit delay with trigger
    wait pulse. If this is not alright use a Pulse(..., trigger=True)
    '''
    kwargs['trigger'] = True
    return Delay(dt, **kwargs)

class Label(Instruction):
    '''
    An instruction to label a particular time in a sequence.
    '''

    LABELS = {}
    def __init__(self, name):
        self.name = name
        if name in Label.LABELS:
            raise ValueError('Can only define label once')
        Label.LABELS[name] = None

    def generate(self, now, chan):
        if Label.LABELS[self.name] != now:
            print 'Setting label %s to %s' % (self.name, now)
            Label.LABELS[self.name] = now
            Label.UPDATED = True
        return Sequence(chan=chan)

class DelayTo(Instruction):
    '''
    Instruction to delay to a particular time or label.
    '''

    def __init__(self, t):
        self.t = t
        super(DelayTo, self).__init__()

    def __str__(self):
        return 'DelayTo(%s)' % self.t

    def get_tgt_time(self, now):
        if type(self.t) is types.StringType:
            if self.t not in Label.LABELS:
                raise ValueError('Label not defined')
            elif Label.LABELS[self.t] is None:
                print 'Warning: Label %s has not been resolved yet.' % self.t
                return now
            return Label.LABELS[self.t]
        else:
            return self.t

    def get_length(self, now=None):
        if now is None:
            return 0
        dt = self.get_tgt_time(now) - now
        if dt < 0:
            print 'Warning: DelayTo does not fit'
            dt = 0
#            raise ValueError('Unable to resolve DelayTo; does not fit')
        return dt

    def generate(self, now, chan):
        dt = self.get_length(now)
        delay = Delay(dt)
        return delay.generate(now, chan)

class CombineChanInfo:
    def __init__(self, seq):
        self.t = 0
        self.idx = 0
        self.out_idx = 0
        self.seq = seq
        self.seq_out = Sequence(chan=seq.chan)
        self.dt = 0
        self.is_delay = False

class Combined(Instruction):
    '''
    An Instruction for pulses that should be played simultaneously.

    The pulses can be either provided as a dictionary of channel --> pulse
    or as a list of instructions.

    <align> specifies the alignment policy:
    - ALIGN_LEFT (align start of pulses)
    - ALIGN_RIGHT (align end of pulses)
    - ALIGN_CENTER (align center of pulses)

    If <ch_align> is True each channel will consist of equal length blocks.
    '''

    def __init__(self, items, align=ALIGN_LEFT, ch_align=False, ch_delays={}, **kwargs):
        self.seqs = {}      # The resolved sequence map for each channel
        self.align = align
        self.ch_align = ch_align
        self.ch_delays = ch_delays
        if type(items) is types.DictType:
            self.items = items.values()
        elif type(items) in (types.ListType, types.TupleType):
            self.items = items
        elif isinstance(items, Instruction):
            self.items = [items]
        else:
            raise ValueError('Invalid items to combine: %s' % (items, ))

        # Number of times this sequence has been resolved
        self.nresolved = 0

        super(Combined, self).__init__(**kwargs)

    def __str__(self):
        return 'Combined(%s)' % (self.items,)

    def get_name(self):
        names = [i.get_name() for i in self.items]
        return 'combine(' + ','.join(names) + ')'

    def add_item(self, item):
        self.items.append(item)

    def get_channels(self):
        chs = set([])
        for el in self.items:
            chs = chs.union(set(el.get_channels()))
        if CHAN_ANY in chs:
            chs.remove(CHAN_ANY)
        return list(chs)

    def get_trigger(self):
        for el in self.items:
            if el.get_trigger():
                return True
        return False

    def get_length(self, now=None, include_delays=True):
        mlen = 0
        item = None
        for el in self.items:
            if hasattr(el, 'name') and el.name.startswith('delay') and not include_delays:
                continue
            mlen = max(mlen, el.get_length(now))
            if mlen == el.get_length(now):
                item = el
        return mlen

    def add_channel_delays(self):
        '''
        Add channel delays to change relative timing of channels.
        '''

        all_delays = np.array(self.ch_delays.values())
        if len(all_delays) == 0:
            return

        maxpos = 0
        if len(all_delays[all_delays>0]) > 0:
            maxpos = np.max(all_delays)

        maxneg = 0
        if len(all_delays[all_delays<0]) > 0:
            maxneg = -np.min(all_delays)

        for ch in self.seqs.keys():
            # Determine delays to add
            d = self.ch_delays.get(ch, 0)
            post = maxpos - d
            pre = maxneg + d

            if pre > 0:
                # Remove spurious triggers
                for el in self.seqs[ch].seq:
                    el.trigger = False
                self.seqs[ch].prepend(Delay(pre))
            if post > 0:
                self.seqs[ch].append(Delay(post))

    def merge_channels(self):
        '''
        Merge channels into aligned pieces.
        '''

        chans = self.seqs.keys()
        if len(chans) == 0:
            return

        chinfo = {}
        for chan in chans:
            chinfo[chan] = CombineChanInfo(self.seqs[chan])
        l_chinfo = chinfo.values()

        # At the start of this loop we can always assume that the sequences
        # are synchronized, i.e. start at this point. That makes future time
        # calculations using time_action_at easy
        while len(l_chinfo[0].seq):
            if DEBUG:
                for ch, info in chinfo.items():
                    print '  ch%s: idx=%d (len %d, trig %s) --> %s' % (ch, info.idx, info.seq.get_length(), info.seq.seq[0].get_trigger(), info.seq)

            # Get some information about the channels
            max_nondelay = 0
            min_delay = 1000000000
            ndel_ch = 0
            for ch, info in chinfo.items():
                info.dt = info.seq[0].get_length()
                info.is_delay = (info.seq[0].get_name() == 'delay1')
                if info.is_delay:
                    ndel_ch += 1
                    min_delay = min(min_delay, info.dt)
                if not info.is_delay and info.dt > max_nondelay:
                    max_nondelay = info.dt

            if DEBUG:
                print '    max_nondelay: %s, mindelay %s, ndel_ch %s' % (max_nondelay, min_delay, ndel_ch)
            if min_delay < 0:
                raise ValueError('Negative delay!')

            # all channels contain a delay, add shortest one as an item
            # to the output and subtract time from others.
            if ndel_ch == len(l_chinfo):
                for ch, info in chinfo.items():
                    add = info.seq.consume_up_to(min_delay, unroll=True)
                    if DEBUG:
                        print '      All delays, adding %s' % (add,)
                    info.seq_out.append(add)

            # At least one channel is active.
            else:

                # Find point up to which to consume
                to_time = max_nondelay
                dt = 1
                while dt > 0:
                    dt = 0
                    for ch, info in chinfo.items():
                        dt = max(dt, info.seq.time_action_at(to_time))
                    to_time += dt

                # Consume sequences up to merge point
                # TODO: we might have to be smarter about repeated elements
                for ch, info in chinfo.items():
                    add = info.seq.consume_up_to(to_time)
                    if DEBUG:
                        print '      Activity, adding to ch %s: %s' % (ch, add,)
                    info.seq_out.append(add)

        for ch, info in chinfo.items():
            self.seqs[ch] = info.seq_out
            if DEBUG:
                print 'Merged channel %s into %s' % (ch, info.seq_out)

    def get_chan_seq(self, chan):
        if chan in self.seqs:
            return self.seqs[chan]
        else:
            return self.aligned_empty_sequence(chan=chan)

    def propagate_label(self):
        if getattr(self, 'label', None):
            for ch in self.seqs.keys():
                self.seqs[ch][0].set_params('label', self.label)

    def resolve(self, now, chan):
        if self.nresolved > 0:
            return self.get_chan_seq(chan)
        self.nresolved += 1

        # Regenerate all combined items
        # TODO: do this until timing doesn't change (i.e. after resolving DelayTo)

        self.seqs = {}
        chans = self.get_channels()
        for ch in chans:
            # See what every item wants to contribute to this channel
            planadd = None
            for el in self.items:
                addseq = el.resolve(now, ch)
                if planadd is None:
                    planadd = addseq
                elif planadd.delays_only():
                    planadd = addseq
                elif addseq.delays_only():
                    pass
                else:
                    raise ValueError('Conflicting channel content: %s and %s (current item %s [%d])' % (planadd, addseq, self.items, len(self.items)))

            self.seqs[ch] = planadd

        self.maxlen = 0
        for ch in self.seqs.keys():
            self.seqs[ch].join_sequences()
            self.maxlen = max(self.maxlen, self.seqs[ch].get_length())

        # If the channels are already aligned there is no need to run through
        # the alignement code.
        aligned = True
        for ch in self.seqs.keys()[1:]:
            if not self.seqs[self.seqs.keys()[0]].is_aligned(self.seqs[ch]):
                aligned = False
        if aligned:
            self.propagate_label()
            if chan in self.seqs:
                return self.seqs[chan]
            else:
                return self.aligned_empty_sequence(chan)

        # Only need to do the rest for channels having content
        chs = self.seqs.keys()

        # Find maximum length and pad sequences appropriately.
        for ch in chs:
            delta = self.maxlen - self.seqs[ch].get_length()
            if delta != 0:
                if self.align == ALIGN_LEFT:
                    self.seqs[ch].append(Delay(delta))
                elif self.align == ALIGN_RIGHT:
                    self.seqs[ch].prepend(Delay(delta))
                else:
                    d1 = int(np.round(delta/2))
                    d2 = delta - d1
                    self.seqs[ch].prepend(Delay(d1))
                    self.seqs[ch].append(Delay(d2))

        # Add requested channel delays
        self.add_channel_delays()

        # Propagate label to first element
        self.propagate_label()

        # We now have the sequences we want to play on each channel, but we
        # still need to process them into aligned pieces.
        if self.ch_align:
            self.merge_channels()

        # Flatten sequences
        for ch in chs:
            self.seqs[ch].join_sequences()

        return self.get_chan_seq(chan)

    def aligned_empty_sequence(self, chan):
        s = Sequence(chan=chan)
        if len(self.seqs) == 0:
            s.append(Delay(self.get_length(), trigger=self.get_trigger()))
            return s

        refseq = self.seqs[self.seqs.keys()[0]]
        for i_el, el in enumerate(refseq.seq):
            repeat = getattr(el, 'repeat', 1)
            kwargs = el.get_flow_params()
            trig = el.get_trigger() | kwargs.pop('trigger', False)
            ellen = el.get_length() / repeat
            if repeat == 1 and not self.ch_align:
                new_el = Delay(ellen, chan=chan, trigger=trig, unroll=True, repeat=repeat, **kwargs)
            else:   # Try to preserve repeats
                new_el = Delay(ellen, chan=chan, trigger=trig, unroll=False, fixed=True, repeat=repeat, **kwargs)
            s.append(new_el)

        if getattr(self, 'label', None):
            s.seq[0].set_params('label', self.label)
            s.seq[0].unroll = False

        if DEBUG:
            print 'Aligned seq ch%s: %s' % (chan, s)

        return s

    # TODO: set trigger only on first item of generated sequence
    # TODO: do twice if a label is present to make sure the other channels
    # agree
    def generate(self, now, chan):
        if chan not in self.seqs:
            return self.aligned_empty_sequence(chan)
        else:
            return self.seqs[chan].generate(now, chan)

def AlignRight(seqs, t=None, tabs=None):
    '''
    Align an instruction or sequence to the right (i.e. end), either
    up to time <tabs> (can be a label) or making this block total length <t>.
    '''
    if type(seqs) is types.TupleType:
        seqs = list(seqs)
    if type(seqs) is not types.ListType:
        seqs = [seqs,]

    m = Combined(seqs, align=ALIGN_RIGHT)
    if tabs:
        m.add_item(DelayTo(tabs))
    elif t is not None:
        m.add_item(Delay(t))

    return m

def Join(els, **kwargs):
    seq = Sequence(els, join=True, trigger=els[0].get_trigger(), **kwargs)
    seq.join_sequences()
    return seq

class Constant(Pulse):
    '''
    Define a constant pulse with width <w> and amplitude <a>.
    '''

    def __init__(self, w, a, **kwargs):
        blockwidth = int(round(w))
        ys = a * np.ones([w,])
        if a == 0:
            name = 'delay%d' % (w,)
        else:
            name = 'const(%03d,%03d)' % (w, round(a*1000))
        super(Constant, self).__init__(name, ys, **kwargs)

def Pad(ins, rlen, pad=PAD_BOTH, err=IGNORE):
    '''
    Pad an instruction to length <rlen> using strategy <pad>
    (PAD_LEFT, PAD_RIGHT or PAD_BOTH).
    <err> specifies what to do if the instruction is longer than <rlen>
    already, and can be IGNORE (default), WARN or RAISE.
    '''

    l = ins.get_length()
    if l > rlen:
        msg = 'Unable to pad, instruction %s too long' % ins
        if err == WARN:
            print msg
        elif err == RAISE:
            raise Exception(msg)
        return ins

    delta = rlen - l
    if pad == PAD_RIGHT:
        return Join([ins, Delay(delta, unroll=False)])
    elif pad == PAD_LEFT:
        return Join([Delay(delta, unroll=False), ins])
    else:
        d1 = int(np.round(delta/2))
        d2 = delta - d1
        return Join([Delay(d1, unroll=False), ins, Delay(d2, unroll=False)])

class Repeat(Instruction):
    '''
    Repeat an instruction <n> times.

    If merge is True and there is a sub-sequence to be repeated, the pulses
    are automatically merged into a single waveform. Otherwise the pulse
    instructions are copied <n> times in a row.
    '''

    def __init__(self, instruction, n, merge=True):
        self.instruction = instruction
        if n == 0:
            raise ValueError('Unable to create 0 repeat pulse')
        self.n = n
        self.merge = merge
        Instruction.__init__(self)

    def get_name(self):
        return '%d%s' % (self.n, self.instruction.get_name())

    def get_length(self, now=None):
        return self.n * self.instruction.get_length(now)

    def get_channels(self):
        return self.instruction.get_channels()

    def get_trigger(self):
        return self.instruction.get_trigger()

    def get_data(self):
        d = self.instruction.get_data()
        return np.tile(d, self.n)

    def _repeat(self, plist):
        # Should we merge into a single waveform?
        if len(plist) != 1 and self.merge:
            j = Join(plist)
            plist = j.generate()

        # If we are dealing with a single element (or just merged) we can use
        # a simple repeat instruction
        if len(plist) == 1:

            # Do not unroll delays for repeat blocks
            if plist.seq[0].name == 'delay1':
                plist.seq[0] = Delay(plist.seq[0].repeat, unroll=False, fixed=True)
                plist.seq[0].repeat = self.n

            # Simple repeat
            else:
                plist.seq[0] = copy.deepcopy(plist.seq[0])
                plist.seq[0].repeat = self.n

            return plist

        # Otherwise repeat sub-sequence.
        else:
            ret = []
            for i in range(self.n):
                ret.extend(plist)
            return ret

    def resolve(self, now, chan):
        plist = self.instruction.resolve(now, chan)
        return self._repeat(plist)

    def generate(self, now, chan):
        plist = self.instruction.generate(now, chan)
        return self._repeat(plist)

def Jump(to, **kwargs):
    kwargs['jump'] = to
    if 'chan' not in kwargs:
        kwargs['chan'] = -1
    data = np.zeros(FLOW_INS_LEN)
    name = 'jump(%s)' % to
    return Pulse(name, data, **kwargs)

#######################################
# Operations
#######################################

class SequenceOperation(object):
    '''
    Base class to perform an operation on generated pulses.
    '''

    def __init__(self, func=None, inplace=False, ttr=True):
        '''
        <inplace>: whether to perform operation in place, currently not
        supported, 'apply' always returns a new sequence.
        <ttr>: trigger time reset, whether to reset time on trigger.
        '''

        self._ext_func = func
        self.inplace = inplace
        self.ttr = ttr

    def func(self, el, now):
        if self._ext_func:
            return self._ext_func(el, now)
        return el

    def apply(self, s):
        '''
        Apply operation to a completely rendered sequence.
        '''
        seq_out = []
        now = 0
        for el in s.seq:
            # Reset time when trigger occurs
            if self.ttr and el.get_trigger():
                now = 0
            if isinstance(el, Pulse):
                el = self.func(el, now)
            seq_out.append(el)
            now += el.get_length()

        if self.inplace:
            s.seq = seq_out
            return self
        else:
            return Sequence(seq_out)

class CleanSeqNames(SequenceOperation):

    def __init__(self):
        super(CleanSeqNames, self).__init__(self)

    def func(self, el, now):
        if len(el.name) > 64 or chars_in_str('(),', el.name):
            newname = hashlib.md5(el.name).hexdigest()
            return Pulse(newname, el.data, repeat=el.repeat, trigger=el.trigger)
        else:
            return el

def clean_names(seqs):
    cleaner = CleanSeqNames()
    seqs_out = {}
    for name, seq in seqs.items():
        seqs_out[name] = cleaner.apply(seq)
    return seqs_out

class ModulateSequence(SequenceOperation):
    '''
    Operation to modulate a sequence, i.e. multiply by a cosine wave with
    period <if_period> and additional phase <phase> (-pi/2 gives a sine wave).
    '''

    def __init__(self, if_period, phase, amp=1.0):
        self.if_period = if_period
        self.phase = phase
        self.amp = amp
        super(ModulateSequence, self).__init__()

    def func(self, el, now):
        data = el.get_data()
        if np.count_nonzero(data) == 0:
            return el
        if el.repeat > 1 and np.abs(len(el.get_data()) % self.if_period) > 1e-4:
            raise ValueError('Unable to modulate repeated blocks not a multiple of if_period')

        phi0 = float(now % self.if_period) / self.if_period * 2 * np.pi + self.phase
        phis = np.linspace(0, 2*np.pi*len(data)/self.if_period, len(data), endpoint=False)
        data = data * self.amp * np.cos(phis + phi0)
        deg = np.rad2deg(phi0 % (2*np.pi))
        name = 'mod(%d,%d,%d,%s)' % (int(round(10000*self.if_period)), int(100*deg), int(10000*self.amp), el.get_name())
        return Pulse(name, data, repeat=el.repeat, chan=el.chan, trigger=el.get_trigger())

class DifferentiateSequence(SequenceOperation):
    '''
    Operation to differentiate a sequence.
    '''

    def __init__(self, amp):
        self.amp = amp
        super(DifferentiateSequence, self).__init__()

    def func(self, el, now):
        data = el.get_data()
        if np.count_nonzero(data) == 0:
            return el
        diff = (data[2:] - data[:-2]) / 2
        data[0] = (data[1] - data[0])
        data[-1] = (data[-1] - data[-2])
        data[1:-1] = diff
        data *= self.amp
        name= 'diff(%.03f,%s)' % (self.amp, el.get_name(),)
        return Pulse(name, data, repeat=el.repeat, chan=el.chan, trigger=el.get_trigger())

class SequencePairOperation(object):
    '''
    Base class to perform an operation on generated pulses.
    '''

    def __init__(self, name='op', func=None, ttr=True):
        '''
        <ttr>: trigger time reset, whether to reset time on trigger.
        '''
        self.name = name
        self._ext_func = func
        self.ttr = ttr

    def func(self, el1, el2, now):
        if self._ext_func:
            return self._ext_func(el1, el2, now)
        return el1, el2

    def apply(self, s1, s2):
        '''
        Apply operation to a completely rendered sequence.
        '''
        if len(s1.seq) != len(s2.seq):
            raise ValueError('Incompatible sequences! Channel %s has %d and channel %s %d items' % (s1.chan, len(s1.seq), s2.chan, len(s2.seq)))

        seq_out = []
        now = 0
        for el1, el2 in zip(s1.seq, s2.seq):
            # Reset time when trigger occurs
            if self.ttr and (el1.get_trigger() or el2.get_trigger()):
                now = 0
            if isinstance(el1, Pulse) and isinstance(el2, Pulse):
                el1 = self.func(el1, el2, now)
            seq_out.append(el1)
            now += el1.get_length()

        s = Sequence(seq_out, chan=self.name)
        return s

class SequenceAdd(SequencePairOperation):
    def __init__(self, name='add'):
        super(SequenceAdd, self).__init__(name=name)

    def func(self, el1, el2, now):
        data1 = el1.get_data()
        data2 = el2.get_data()
        if np.count_nonzero(data1) == 0:
            return el2
        if np.count_nonzero(data2) == 0:
            return el1
        data = data1 + data2
        name = 'add(%s,%s)' % (el1.get_name(), el2.get_name())
        return Pulse(name, data, repeat=el1.repeat, chan=el1.chan, trigger=el1.get_trigger())

class SequenceSub(SequencePairOperation):
    def __init__(self, name='sub'):
        super(SequenceSub, self).__init__(name=name)

    def func(self, el1, el2, now):
        data1 = el1.get_data()
        data2 = el2.get_data()
        if np.count_nonzero(data2) == 0:
            return el1
        if np.count_nonzero(data1) == 0:
            name = 'minus(%s)' % (el2.get_name())
            data = -data2
            return Pulse(name, data, repeat=el1.repeat, chan=el1.chan, trigger=el1.get_trigger())
        data = data1 - data2
        name = 'sub(%s,%s)' % (el1.get_name(), el2.get_name())
        return Pulse(name, data, repeat=el1.repeat, chan=el1.chan, trigger=el1.get_trigger())

def add_sequences(s1, s2, name='add'):
    return SequenceAdd(name).apply(s1, s2)

def sub_sequences(s1, s2, name='sub'):
    return SequenceSub(name).apply(s1, s2)

class SSB:
    '''
    Single Side Band modulation class

    <phi> represents the phase imbalance, i.e. the extra phase to add on the Q
    channel.

    The transformation that SSB performs is the following:

        Iin -> cos(x) on Iout, -sin(x + dphi) on Qout
        Qin -> sin(x) on Iout, cos(x + dphi) on Qout

    Or in matrix form:

        Iout = [cos(x)          sin(x)     ]    Iin
        Qout = [-sin(x+dphi)    cos(x+dphi)]    Qin

    (So the phase correction is on the second output channel).
    '''

    def __init__(self, if_period, chans, dphi, outchans=None, amps=(1.0, 1.0), replace=None):
        self.if_period = if_period
        self.chans = chans
        self.dphi = dphi
        self.outchans = outchans
        self.amps = amps
        if replace is None:
            if outchans is None or outchans == chans:
                replace = True
            else:
                replace = False
        self.replace = replace


    def modulate(self, seqs):
        # From I channel to I/Q outputs
        if self.chans[0] in seqs:
            m0 = ModulateSequence(self.if_period, 0, self.amps[0])
            m1 = ModulateSequence(self.if_period, np.pi/2 + self.dphi, self.amps[1])
            c0m0 = m0.apply(seqs[self.chans[0]])
            c0m1 = m1.apply(seqs[self.chans[0]])

        # From Q channel to I/Q outputs
        if self.chans[1] in seqs:
            m0 = ModulateSequence(self.if_period, -np.pi/2, self.amps[0])
            m1 = ModulateSequence(self.if_period, self.dphi, self.amps[1])
            c1m0 = m0.apply(seqs[self.chans[1]])
            c1m1 = m1.apply(seqs[self.chans[1]])

        if self.outchans:
            outchans = self.outchans
        else:
            outchans = self.chans

        # Both channels used as input present
        if self.chans[0] in seqs and self.chans[1] in seqs:
            tmp0 = add_sequences(c0m0, c1m0, outchans[0])
            tmp1 = add_sequences(c0m1, c1m1, outchans[1])

        # Just one input channel present
        elif self.chans[0] in seqs:
            tmp0 = c0m0
            tmp1 = c0m1
        elif self.chans[1] in seqs:
            tmp0 = c1m0
            tmp1 = c1m1

        # None present
        else:
            return seqs

        tmp0.chan = outchans[0]
        tmp1.chan = outchans[1]

        if outchans[0] in seqs and not self.replace:
            seqs[outchans[0]] = add_sequences(seqs[outchans[0]], tmp0, outchans[0])
        else:
            seqs[outchans[0]] = tmp0
        if outchans[1] in seqs and not self.replace:
            seqs[outchans[1]] = add_sequences(seqs[outchans[1]], tmp1, outchans[1])
        else:
            seqs[outchans[1]] = tmp1

        return seqs

def map_pulse_names(names):
    ret = {}
    for name in names:
        ret[name] = hashlib.md5(name).hexdigest()
    return ret

if __name__ == '__main__':
    print 'Please use testsequencer.py to test'
