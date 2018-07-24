# fpga_sequences.py, Reinier Heeres 2014
# Some general sequences that can be included to actively cool a qubit or
# qubit/cavity system.

from pulseseq import sequencer, pulselib
import fpgapulses
import YngwieEncoding as ye
import numpy as np

def cavity_cool(m, NCONVINCE=4, label='qccool', tgtlabel=None):
    '''
    Qubit/cavity |gg> cooling sequence.
    If <tgtlabel> is None, a function return will be generated at the end,
    otherwise the sequence jumps to <tgtlabel> on completion.
    Uses registers R1, R2 and counter0
    '''

    if tgtlabel:
        retlabel = tgtlabel
    else:
        retlabel = label + '_ret'

    s = sequencer.Sequence()

    s.append(sequencer.Delay(1000, label=label, master_integrate=m.integrate_nolog, master_counter0=NCONVINCE))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.NOP(), label='_'))

    # First measurement, store result in R1, 0 = |g>, 1 = |e>
    s.append(fpgapulses.LongMeasurementPulse())
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.NOP(), length=220, master_internal_function=ye.FPGASignals.s0, jump=(label+'_setR1_0', label+'_setR1_1')))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOVI(1, 0), label=label+'_setR1_0', jump=label+'_M2'))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOVI(1, 1), label=label+'_setR1_1'))

    # Second measurement, store result in R2, 0 = |g>, 1 = |e>
    s.append(m.qubit.rotate_selective(np.pi,0,label=label+'_M2'))
    s.append(fpgapulses.LongMeasurementPulse())
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.NOP(), length=220, master_internal_function=ye.FPGASignals.s0, jump=(label+'_setR2_0', label+'_setR2_1')))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOVI(2, 0), label=label+'_setR2_0', jump=label+'_decide'))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOVI(2, 1), label=label+'_setR2_1'))

    # Compare R1 and R2
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.CMP(1, 2), master_internal_function=ye.FPGASignals.r0, label=label+'_decide', jump=(label+'_same', label+'_diff')))
    # If the same: reset counter, move last outcome to R1 and re-measure
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOV(1, 2), master_counter0=NCONVINCE, jump=label+'_M2', label=label+'_same'))

    # If different: decrease counter, move last outcome to R1 and goto remeasure if counter not zero
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOV(1, 2), master_counter0=-1, master_internal_function=ye.FPGASignals.c0, jump=('next', label+'_M2'), label=label+'_diff'))

    # We're almost good, make sure we are actually in |g>, if so jump to start
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.CMPI(1, 0), master_internal_function=ye.FPGASignals.r0, jump=(retlabel, 'next')))
#    s.append(fpgapulses.RegOp(ye.RegisterInstruction.CMPI(1, 0), master_internal_function=ye.FPGASignals.r0, jump=(label+'_qcool', 'next')))
    # Otherwise, set counter to 1 and remeasure
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.NOP(), master_counter0=1, jump=label+'_M2'))
    if tgtlabel is None:
        s.append(fpgapulses.RegOp(ye.RegisterInstruction.NOP(), callreturn=True, label=retlabel))
    return s

def qubit_cool(m, NCONVINCE=4, label='qcool', tgtlabel=None):
    '''
    Qubit cooling sequence: measure |g> <NCONVINCE> times in a row.
    If <tgtlabel> is None, a function return will be generated at the end,
    otherwise the sequence jumps to <tgtlabel> on completion.
    Uses counter0
    '''
    if tgtlabel:
        retlabel = tgtlabel
    else:
        retlabel = label + '_ret'
    s = sequencer.Sequence()
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.NOP(), master_integrate=m.integrate_nolog, master_counter0=NCONVINCE, label=label))
    s.append(fpgapulses.LongMeasurementPulse(label=label+'_meas'))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.NOP(), length=220, master_internal_function=ye.FPGASignals.s0, jump=(label+'_decc0', 'next')))
    # We measured |e>, reset counter and jump back to measure
    s.append(m.qubit.rotate(np.pi,0,jump=label+'_meas',master_counter0=NCONVINCE))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.NOP(), master_counter0=-1, master_internal_function=ye.FPGASignals.c0, jump=(retlabel, label+'_meas'), label=label+'_decc0'))
    if tgtlabel is None:
        s.append(fpgapulses.RegOp(ye.RegisterInstruction., NCONVINCE=4NOP(), callreturn=True, label=retlabel))
    return s

def qubit_cavity_cool(m, NCONVINCE=4, label='qccool', tgtlabel=None):
    '''
    Qubit/cavity |gg> cooling sequence.
    If <tgtlabel> is None, a function return will be generated at the end,
    otherwise the sequence jumps to <tgtlabel> on completion.
    Uses registers R1, R2 and counter0
    '''

    s = cavity_cool(m, NCONVINCE, label=label, tgtlabel=label+'_q')
    s.append(qubit_cool(m, NCONVINCE, label=label+'_q', tgtlabel=tgtlabel))
    return s

