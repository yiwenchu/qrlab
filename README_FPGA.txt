Yngwie FPGA / Python GETTING STARTED.

FPGA INSTALL
First install the FPGA card, ask Nissim for help. Make sure you load logic
A15 or higher, as this contains several features that will be used.
If you're using just one card, you'll need the Texas Instruments board to
provide a 1 GHz clock and trigger signals with a fast enough rise time. It
has some crazy java control software; ask Yehan to set it up. If you're using
2 cards or more, you'd better get an X3 timing card, or better yet, a Black
Mamba housing which contains a trigger card.
I would absolutely recommend to have a 4 channel scope around with a reasonable
bandwidth. Note that the FPGA does not have inverted copies of the signal
output (as the Tek did), so I take resistive splitters (Minicircuits ZFRSC-42+)
to split the signal and plug it in the scope.

INPUTS / OUTPUTS / MARKERS
Now hook up the in- and outputs of the card. If you use just 2 outputs at
1 GHz, you should connect Analog 0 and Analog 2(?) to your mixer, if you're
using 4 channels at 500 MHz, Analog 0/1 and Analog 2/3 form a pair. Note that
the output bandwidth of the FPGA is very high, and you'll absolutely need
low-pass filters to get rid of the steps in the signals. I use the Minicircuits
BLP-300+ (loss < 1dB to 225 MHz, 3dB @ 250 MHz) on all outputs, but maybe a bit
lower would be even better. Use some (10 dB for me) attenuators on the mixer
I/Q inputs, as you want to stay in the linear driving regime.

The reference should be hooked up to Input 0, the signal to Input 1.

There are two automatically generated 'buffer' signals which can go to a fast
RF switch when there is activity on the analog channels. If the card is in your
computer: Analog 0/1 lives on N10, Analog 2/3 on P10. If the card is in the
Black Mamba: ...
Note that the output voltage of the marker lines, about 1.25V in 50 Ohms, is
not quite enough for the Hittite switches we use, so you'll have to get some
amplifier to step the signal up by about a factor 2.
There are 4 other markers that can be used; they are controlled by two
independent 'sequencers' running on the FPGA. In the control software these
sequencers are called 'channels' m0 and m1. The actual markers are controlled
in a bitwise way, so bit0 controls the first marker and bit1 the second. So
playing 1 in m0 sets the first marker high, playing 2 in m0 sets the second
marker high and playing 3 sets both markers high. The markers can currently
only be set high or low for a whole sequence element, so at least ~10 cycles
(= 40 ns).
The following lines are used (i.e. hard-wired in the FPGA):
'marker 0': m01 = P12
'marker 1': m02 = N12
'marker 2': m11 = P13
'marker 3': m12 = N13

CHECKING SIGNAL LEVELS
The FPGA has 12 bit ADCs, which means values from -2048 to +2048. You really
want to use as much of the input range as you can. Unlike the Alazar card, the
FPGA does not have internal amplifiers which allow you to select an input
range, so you might have to add some amplification manually.

First setup the readout parameters in the instrument GUI:
- <pulse_len>: the length of the readout (marker) pulse sent to channel
<readout_chan>. For example: 500 (ns) to m01 (the marker at P12).
- <acq_len>: the acquisition length for the signal
- <ref_len>: the acquisition length for the reference. The FPGA estimates the
phase of the reference every IQ cycle. Because the typical read-out we do sends
a trigger to an Agilent RF generator the reference is only available during the
read-out pulse. You have to make sure that you set <ref_len> such that reading
out the reference stops before the end of the pulse. The FPGA will use the last
estimated phase to correct the rest of the signal. Unless your read-out is
crazy slow this should work totally fine.

Now run fpga_rawreadout.py to acquire some traces and check the signal levels,
as well as whether you have chosen the read-out lengths correctly.

TUNING SINGLE SIDEBAND MODULATION
If you want to be able to automatically tune-up carrier leakage and sideband
modulation, you'll need a 'Vlastaktrum' analyzer (these are better than the
BBN version because the BBN one has an internal mixer which is limiting it's
frequency range). See leakage_opt_fpga.py and sideband_opt_fpga.py.
Otherwise, you can do the tune-up manually.

SEQUENCE / INSTRUCTION NOTES
- After each register operation:
    * if the result is 0, the r0 flag is HIGH
    * if the result is negative, the r1 flag is HIGH
- For the state estimator:
    * if the estimator value is above the threshold, s0 is HIGH
- If counter0 (counter1) is zero, c0 (c1) is HIGH

Special registers:
- R0/R1/R2/R3 represent the 2x2 matrix that can be loaded into the mixer
amplitudes (using FPGAMeasurement.qubit.loaduse_mixer_kwargs). This allows
to programmatically control the amplitude of a pulse.
See fpgarabi.py for an example.

- R12 is the register that can be used to programmatically change the sideband
modulation frequency. It contains the ssb frequency in kHz. You can use
FPGAMeasurement.qubit.load_ssb_kwargs to apply the ssb freq to a mode. Takes
a while to actually be available; I typically wait a microsecond.
See fpgassbspec.py for an example.

- R13, R14 and R15 can be used to encode a dynamic instruction length. Use
inslength=('R13', 10) as a pulse keyword argument to make it of length R13 +
10 cycles.
See fpgaT1.py as an example.
