
import sys
import os
import time
import subprocess as sp

script = 'main.py'
while True:
    print('Running script file', script)
    time.sleep(5)
    progress = sp.Popen(['python3', script])
    progress.wait()
