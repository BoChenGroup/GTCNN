# -*- coding: utf-8 -*-
# Created by Chaos 2020/4/28

import numpy as np

real_min = 2.2e-16
def log_max(x):
    return np.log(np.maximum(x, 0.000001))

class Empty():
    pass