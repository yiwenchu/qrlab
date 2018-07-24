qrlab
=====

Python scripts for Measurement

Installation
------------

### Python Installation

- Grab [64-bit (x86-64) python 2.7](https://www.python.org/downloads/windows/)
as of writing the most current version is 2.7.8 (not yet compatible with Python 3.x)
- Install additional libraries from [this wonderful resource](http://www.lfd.uci.edu/~gohlke/pythonlibs/), making sure to get 64bit / py2.7 versions
- Make sure you get at least
  - [Scipy-stack](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy-stack)
  - [docutils](http://www.lfd.uci.edu/~gohlke/pythonlibs/#docutils)
  - [Spyder](http://www.lfd.uci.edu/~gohlke/pythonlibs/#spyder)
  - [QuTiP](http://www.lfd.uci.edu/~gohlke/pythonlibs/#qutip)
  - [PyQt4](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4)
  - [h5py](http://www.lfd.uci.edu/~gohlke/pythonlibs/#h5py)
  - [PyVISA](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyvisa)
  - [PyZMQ](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyzmq)
  	- Soon to be phased out, but not there yet!
- Also, Make sure you have [VISA](http://www.ni.com/visa/) separately installed

### qrlab setup

If you haven't already, install [git](http://git-scm.com/download/win).  Open
up "Git Bash" (resist the temptation to use git GUI). Change directory to where
you want qrlab to end up. I recommend C:\. Since bash uses 
[unix style paths](http://en.wikipedia.org/wiki/Path_%28computing%29#Unix_style)
this would be `cd /c`. Your desktop would be `cd /c/Users/<username>/Desktop`,
etc. Once there, run `git clone https://git.yale.edu/rsl/qrlab.git`. Which will
prompt you for authentication. This will be your yale net id and password,
assuming you have already logged in once.

**don't need to do this part anymore. These repositories are included in qrlab now

Run `install.bat` from the qrlab directory. This will clone several
additional repositories from various places. The repositories from
git.yale.edu will require your authentication.  You should
end up with these five repositories

- [objectsharer](http://github.com/heeres/objectsharer)
- [dataserver](http://github.com/heeres/dataserver)
- [H5Plot](http://github.com/philreinhold/h5plot)
- [instrumentserver](http://git.yale.edu/rsl/instrumentserver)
- [pulseseq](http://git.yale.edu/rsl/pulseseq)
**

In the `config.py` file, specify your preferred data directory. In the
`create_instruments.py` file, setup your instrument classes following the 
patterns established therein.

Test your installation out by running `start.bat`. A bunch of windows should
pop up. If they all close immediately, run start.bat from an already open
shell and debug the output. If the instrument gui appears (a small blank window)
run `create_instruments.py` which should populate the window with instruments which can be communicated with.
