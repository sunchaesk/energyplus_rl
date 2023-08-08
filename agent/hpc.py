
import sys
import os
import subprocess as sp
import time
import shutil

import numpy as np

try:
    sp.Popen(['mkdir', 'results'])
except:
    print('results/ dir already exists')

EPISODE = 2

for lmbda in np.arange(0.5, (20 + 0.5), 0.5):
    print('lmbda', lmbda)
    output_name = 'lambda=' + str(lmbda)
    try:
        sp.Popen(['mkdir', 'results/' + output_name])
    except:
        print('dir already exists:', output_name)

    #print('HIT?')

    process = sp.Popen(['python3', 'sac.py',
              '-lmbda', str(lmbda),
              '-output', 'results/' + output_name,
              '-ep', str(EPISODE)])
    process.communicate()

    # for testing, lmbda is not used but, required for initializing the env
    # TODO: change the env so that it doesn't require lmbda value
    process = sp.Popen(['python3', 'testing.py', '-lmbda', str(lmbda), '-output', 'results/' + output_name])
    process.communicate()
