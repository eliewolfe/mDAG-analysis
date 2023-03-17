# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:02:21 2021

@author: ansan
"""

import time
import progressbar as p

for i in p.progressbar(range(100)):
    time.sleep(0.02)