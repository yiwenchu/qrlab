Alazar / Tek / Python GETTING STARTED.

INSTALL
Install the Alazar in your computer and get a Tektronix AWG (5014C preferably),
check with Kevin if the software on the AWG is up to date.

INPUTS / OUTPUTS / MARKERS
The Alazar should get a 10 MHz clock and the reference should be hooked up to
Input 0, the signal to Input 1.

The Python code expects that you have a function generator (called 'funcgen',
which should have the SYNC OUTPUT (!) connected to your (primary) AWG trigger.
The reason is that the function generator's signal output has horrible
transients if you turn the signal on.

CHECKING SIGNAL LEVELS
You can use alazar DSO to check your signal input levels. Select the range that
seems reasonable; 40mV is the smallest range that the hardware does.

TUNING SINGLE SIDEBAND MODULATION
If you want to be able to automatically tune-up carrier leakage and sideband
modulation, you'll need a 'Vlastaktrum' analyzer (these are better than the
BBN version because the BBN one has an internal mixer which is limiting it's
frequency range). Otherwise, you can do the tune-up manually using a spectrum
analyzer.
