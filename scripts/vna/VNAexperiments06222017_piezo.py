### LOTS OF USEFUL FUNCTIONS IN THIS ONE

import numpy as np
# import mclient
# from mclient import instruments

from time import sleep
from time import localtime
import time
from scripts.vna import VNA_functions
# from scripts import VNA_functions
current_milli_time = lambda: int(round(time.time() * 1000))
import os
from matplotlib import pyplot as plt
from datetime import datetime, time, timedelta
import mclient
from mclient import instruments

vna = instruments['VNA']
set1 = VNA_functions.collect_and_display(vna,fridge=None,folder=r'C:\Data\vna\\', save=True,display=True,fpost=".dat",fpre='KLfilter_12GHz',)