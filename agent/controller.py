
import subprocess as sp
import time
import psutil
import os
import shutil

script = 'cp_agent_ac.py'

if __name__ == "__main__":
    print('Running Script File:', script)
    time.sleep(5)
    sp.Popen(['mkdir', 'model/'])
    sp.Popen(['mkdir', 'logs/'])
    run = sp.Popen(['python3', script])
    while True:
        print("PSUTIL:MEMORY:", psutil.virtual_memory().percent)
        if psutil.virtual_memory().percent >= 75:
            sp.Popen.terminate(run)
            # dir_name = '/home/ck/Downloads/energyplus_rl/agent/output'
            # dir_name_1 = '/home/ck/Downloads/energyplus_rl/agent/runs'
            # for files in os.listdir(dir_name):
            #     path = os.path.join(dir_name, files)
            #     try:
            #         shutil.rmtree(path)
            #     except:
            #         os.remove(path)

            # for files in os.listdir(dir_name_1):
            #     path = os.path.join(dir_name, files)
            #     try:
            #         shutil.rmtree(path)
            #     except:
            #         os.remove(path)


            run = sp.Popen(['python3', script])

        time.sleep(5)
