

import functools
import os
import scipy.io as sio
import numpy as np


data = sio.loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat') # load dataset
#print(data)

inputs = data['PAMsymRx'].flatten().tolist()
Targets = data['PAMsymTx'].flatten().tolist()
#print(Tx)
#print(Rx)


  