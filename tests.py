# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:53:32 2015

@author: Smeagol
"""

import mclient
qubit_info = mclient.get_qubit_info('qubit_info')

from scripts.single_qubit import rabi
r = rabi.Rabi(qubit_info, np.linspace(0, 1, 21))
r.measure()
