# fpgagenerator.py, Reinier Heeres, 2014
# Glue between the pulse sequencer and Nissim's FPGA table generation code.

import numpy as np
from load_fpgadevelop import *
from YngwieEncoding import *
from pulseseq import sequencer, pulselib
import sys
import time
import types
import fpgapulses
import config

def join_files(fns, outfn):
    fout = open(outfn, 'wb')
    for fn in fns:
        fin = open(fn, 'rb')
        while True:
            buf = fin.read(8192)
            if not buf:
                break
            fout.write(buf)
        fin.close()
    fout.close()

def branch_abs_kwargs(curaddr, tgtaddr):
    return dict(addr=curaddr, goto0=tgtaddr, goto1=0,
                branch_type=BranchType.GOTO,
                addressing_mode=0)

def branch_kwargs(ins, ofs):
    branch_type = getattr(ins, 'branch_type', BranchType.GOTO)
    if getattr(ins, 'call', False):
        branch_type = BranchType.GOSUB
    if getattr(ins, 'callreturn', False):
        branch_type = BranchType.RETURN
    if hasattr(ins, 'to_address'):
        tgtaddr = ins.to_address
        if type(tgtaddr) in (types.ListType, types.TupleType):
            tgtaddr = tgtaddr[0]+ofs, tgtaddr[1]+ofs
            tgtstr = ins.jump[0], ins.jump[1]
            if branch_type == BranchType.GOTO:
                branch_type = BranchType.IF
        else:
            tgtaddr = tgtaddr+ofs, 0
            tgtstr = ins.jump, ''
    else:
        tgtaddr = ins.address+1, 0
        tgtstr = 'next', ''
    if branch_type == BranchType.RETURN:
        tgtaddr = tgtaddr[0], 1
    return dict(addr=ins.address, goto0=tgtaddr[0], goto1=tgtaddr[1],
                goto0_label=(tgtstr[0], ''), goto1_label=(tgtstr[1], ''),
                branch_type=branch_type, addressing_mode=0)

def return_kwargs(curaddr):
    return dict(addr=curaddr, goto0=0, goto1=0, branch_type=2, addressing_mode=0)

def get_fpga_kwargs(ins, chan):
    '''
    Get all keyword arguments starting with <chan>_ and copy them without the
    first part.
    '''
    ret = dict()
    for k, v in ins.params.items():
        if chan is not None and k.startswith(chan+'_'):
            ret[k[len(chan)+1:]] = v
    return ret

def map_special(ins, instype, ofs, chan, **kwargs):
    '''
    Map special instructions; only delays actually.
    '''
    if hasattr(ins, 'master_internal_function'):
        setattr(ins, 'master_load_internal_function', True)
    kwargs.update(get_fpga_kwargs(ins, chan))
    if ins.name.startswith('delay'):
        length = ins.params.get('inslength', ins.get_length()/4)
        return instype(length=length,
                       label=(getattr(ins, 'label', ''), ''),
                       branch_params=branch_kwargs(ins, ofs),
                       **kwargs)
    return None

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 7

class FPGAGenerator:
    '''
    Class to generate analog/digital/master sequence tables.
    '''

    def __init__(self, tables_prefix, preamble_len=512, integration=0, nmodes=2, nchannels=4):
        self.tables_prefix = tables_prefix
        self.preamble_len = preamble_len
        # On the small card, don't use efficient memory encoding due to bug
        use_fourchans = (not getattr(config, 'small_fpga', False)) and (nchannels == 4)
        self.wm = WaveMemory(nmodes, four_channels=use_fourchans)
        self.integration = integration
        self.nmodes = nmodes
        if 0:
            self.wave_fig = plt.figure()
            self.wave_ax = self.wave_fig.add_subplot(111)
            self.fft_fig = plt.figure()
            self.fft_ax = self.fft_fig.add_subplot(111)

    def encode_analog_mode_chan(self, A, seq, ofs, pulse_addrs, mode_id, chan):
        if seq is None:
            return
        for i_el, el in enumerate(seq.seq):
            el.address += ofs
            insout = map_special(el, AnalogInstruction, ofs, 'chan%d'%chan, wave_address=0, unique_wave=False)
            if insout is None:
                wname = 'M%d_%s' % (mode_id, el.name)
                ddata = np.round(el.data*0x7FFF).astype(np.int16)

                # Determine whether the pulse is 'unique', i.e. all samples identical
                unique = (np.count_nonzero(ddata - np.average(ddata)) != 0)
                if not unique and ddata[0] == 0:
                    pulse_addrs[wname] = 0
                elif wname not in pulse_addrs:
                    try:
                        # If the pulse is a constant, just add 4 ns worth of data
                        if not unique:
                            pulse_addrs[wname] = self.wm.append(ddata[0:4], mode=mode_id)
                        else:
                            pulse_addrs[wname] = self.wm.append(ddata, mode=mode_id)
                    except Exception, e:
                        print 'Unable to add pulse %s of length %s' % (wname, len(el.get_data()))
                        raise(e)
                length = el.params.get('inslength', el.get_length()/4)

                # Get keyword arguments, set mixer_amplitudes to default values:
                # (65535,0) for I channels, (0, 65535) for Q
                channame = 'chan%d'%chan
                inskwargs = get_fpga_kwargs(el, channame)
                if ('mixer_amplitudes') not in inskwargs:
                    if (chan % 2) == 0:
                        inskwargs['mixer_amplitudes'] = (65535, 0)
                    else:
                        inskwargs['mixer_amplitudes'] = (0, 65535)
                insout = AnalogInstruction(length=length, wave_address=pulse_addrs[wname], unique_wave=unique,
                                           branch_params=branch_kwargs(el, ofs),
                                           **inskwargs)
            A.append(insout)

    def encode_analog_mode(self, seq0, seq1, mode_id, chans):
        # Determine lengths
        totlen = 0
        seq0len = 0
        if seq0 is not None:
            seq0len = len(seq0.seq)
            totlen = seq0.get_length()
        if seq1 is not None:
            seq1len = len(seq1.seq)
            totlen = seq1.get_length()

        # Start the I/Q channels at different places
        A = [
            AnalogInstruction(length=self.preamble_len/4, branch_params=branch_abs_kwargs(0, 2)),
            AnalogInstruction(length=self.preamble_len/4, branch_params=branch_abs_kwargs(1, 2+seq0len)),
        ]

        pulse_addrs = {}
        self.encode_analog_mode_chan(A, seq0, 2, pulse_addrs, mode_id, chans[0])
        self.encode_analog_mode_chan(A, seq1, 2+seq0len, pulse_addrs, mode_id, chans[1])
        return A

    def encode_digital_mode_chan(self, D, seq, ofs, chan):
        if seq is None:
            return
        for i_el, el in enumerate(seq.seq):
            el.address += ofs
            insout = map_special(el, DigitalInstruction, ofs, 'dig%d'%chan, marker_levels=[[0],[0]])
            if insout is None:
                v0 = int(round(el.get_data()[0])) & 0x01
                v1 = (int(round(el.get_data()[0])) & 0x02) >> 1
                length = el.params.get('inslength', el.get_length()/4)
                insout = DigitalInstruction(length=length,
                                            marker_levels=[[v0],[v1]],
                                            branch_params=branch_kwargs(el, ofs))
            D.append(insout)
        return D

    def encode_digital_mode(self, seq0, seq1):
        # Determine lengths
        totlen = 0
        seq0len = 0
        if seq0 is not None:
            seq0len = len(seq0.seq)
            totlen = seq0.get_length()
        if seq1 is not None:
            seq1len = len(seq1.seq)
            totlen = seq1.get_length()

        # Start the I/Q channels at different places
        D = [
            DigitalInstruction(length=self.preamble_len/4, branch_params=branch_abs_kwargs(0, 2)),
            DigitalInstruction(length=self.preamble_len/4, branch_params=branch_abs_kwargs(1, 2+seq0len)),
        ]
        self.encode_digital_mode_chan(D, seq0, 2, 0)
        self.encode_digital_mode_chan(D, seq1, 2+seq0len, 1)
        return D

    def encode_master(self, seq, ofs=1):
        M = [
            MasterInstruction(length=self.preamble_len/4,
                              branch_params=branch_abs_kwargs(0, 1),
                              estimation_params_addr=self.integration,
                              estimation_params_update=True,
            )
        ]
        for i_el, el in enumerate(seq.seq):
            el.address += 1
            insout = map_special(el, MasterInstruction, ofs, 'master')
            if insout is None:
                v0 = int(round(el.get_data()[0])) & 0x01
                v1 = (int(round(el.get_data()[0])) & 0x02) >> 1
                length = el.params.get('inslength', el.get_length()/4)
                insout = MasterInstruction(
                    length=length,
                    label=(getattr(el, 'label', ''), ''),
                    trigger_levels=[[v0],[v1]],
                    branch_params=branch_kwargs(el, 1),
                )
            integrate = getattr(el, 'master_integrate', None)
            if integrate is not None:
                insout.estimation_params_addr = integrate
                insout.estimation_params_update = True
            M.append(insout)

        export_master_sequence(self.tables_prefix, M)
        return M

    def generate(self, seqs, plot=False):
        '''
        Generate the analog/digital/master sequence using seqs:
        '''
        # Determine which modes are used
        chans = seqs.keys()
        mode0 = ((0 in chans) or (1 in chans))
        mode1 = ((2 in chans) or (3 in chans))

        # Encode the tables for the individual analog modes
        As = []
        if mode0:
            As.append(self.encode_analog_mode(seqs.get(0, None), seqs.get(1, None), 0, (0,1)))
        else:
            As.append(None)
        if mode1:
            As.append(self.encode_analog_mode(seqs.get(2, None), seqs.get(3, None), 2, (2,3)))
        else:
            As.append(None)
        export_analog_sequence(self.tables_prefix, mode0=As[0], mode2=As[1])

        # Export wavememory
        self.wm.export(self.tables_prefix)

        # Encode digital markers
        D = self.encode_digital_mode(seqs.get('m0', None), seqs.get('m1', None))
        export_digital_sequence(self.tables_prefix, D)

        # Encode master channel
        if 'master' in seqs:
            M = self.encode_master(seqs['master'])

        if plot:
            self.plot_sequencers(M, As, D)

    def plot_sequence(self, S, y0, elofs=0, label=''):
        '''
        Plot a sequence of elements S: draw the boundaries and the control
        flow arrows.
        Return <tick_ts>, <tick_labels> and <jump_lines>.
        The latter can be compared for each sequence to make sure things are
        aligned properly.
        '''

        plt.text(50, y0+0.1, label)

        tick_ts = []
        tick_labels = []
        jump_lines = []

        # Find times for each instruction
        t = 0
        Sts = []
        for i, el in enumerate(S):
            Sts.append(t)
            ellength = el.length
            if type(ellength) in (list, tuple):
                ellength = length[1]
            t += ellength #* 4

        il = 0
        for i, (el, t) in enumerate(zip(S, Sts)):
            ellength = el.length
            if type(ellength) in (list, tuple):
                ellength = length[1]
            tend = t+ellength #*4
            plt.plot([tend, tend], [y0, y0+0.5])
            if el.label not in ('none', ''):
                tick_ts.append(t)
                tick_labels.append(el.label[0])
            branches = [(el.goto0, 'k')]
            if el.branch_type != BranchType.RETURN:
                branches.append((el.goto1, 'b'))
            for goto, col in branches:
                if goto != 0 and goto != i+elofs+1:
                    il = (il + 1) % 20
                    dx = Sts[goto-elofs] - tend
                    arsize = 200
                    while arsize > abs(dx):
                        arsize /= 2
                    if dx < 0:
                        dx += arsize
                    else:
                        dx -= arsize
                    jump_lines.append((i, (tend, dx)))
                    plt.arrow(tend, y0+(il*0.025), dx, 0, lw=0.05, head_width=0.06, head_length=arsize, fc=col, ec=col)
                    plt.plot(tend, y0+(il*0.025), 'o', color='k', ms=4)

        return tick_ts, tick_labels, jump_lines

    def check_alignment(self, jump1, jump2, label):
        '''
        Check alignment of jumps in two sequences.
        One of them is assumed to be the master sequence.
        '''
        l = min(len(jump1), len(jump2))
        if len(jump1) != len(jump2):
            print 'WARNING: %s contains different number of jumps (%d / %d) than master' % (label, len(jump2)-len(jump1), len(jump1))
        for i in range(l):
            if jump1[i][1] != jump2[i][1]:
                print 'WARNING: %s jump %d @ idx %d (time %d) differs from master' % (label, i, jump1[i][0], jump1[i][1][0])
#                break

    def plot_sequencers(self, M, As, D):
        plt.figure()
        tick_ts, tick_labels, m_lines = self.plot_sequence(M[1:], 0, elofs=1, label='M')
        y0 = 1
        Ss = []
        for i, A in enumerate(As):
            Ss.append(('A%d'%i, A))
        Ss.append(('D', D))
        for name, S in Ss:
            start1 = S[0].goto0
            start2 = S[1].goto0
            seq1 = S[start1:start2]
            ret = self.plot_sequence(seq1, y0, elofs=start1, label='%s/0'%name)
            self.check_alignment(m_lines, ret[2], '%s/0'%name)
            seq2 = S[start2:]
            ret = self.plot_sequence(seq2, y0+1, elofs=start2, label='%s/1'%name)
            self.check_alignment(m_lines, ret[2], '%s/1'%name)
            y0 += 2

        # Set limits and plot labels
        plt.ylim(-0.1, y0+0.1)
        for t, label in zip(tick_ts, tick_labels):
            plt.plot([t,t],[0,y0],'k--')
            plt.text(t+25, y0-0.5, label, rotation=90)

