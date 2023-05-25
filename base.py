# pip installed ray, tabulate, tree
# pip uninstalled ''
import argparse
import sys
import os
import threading
from tempfile import TemporaryDirectory
from typing import Dict, Any, Tuple, Optional, List
from queue import Queue, Empty, Full
from datetime import datetime

import random

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

sys.path.insert(0, '/home/ck/Downloads/EnergyPlus-23.1.0-87ed9199d4-Linux-CentOS7.9.2009-x86_64/')
from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.datatransfer import DataExchange

'''
NOTE:
- variables:
    - outdoor_temp
    - indoor_temp_living
    - indoor_temp_attic
- meter:
    - elec (Electricity:HVAC)
- actuators:
    - sat_spt: system temperature node setpoint
'''

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idf",
        help="Path to .idf file",
        required=True
    )
    parser.add_argument(
        "--epw",
        help="Path to weather file",
        required=True
    )
    parser.add_argument(
        "--csv",
        help="Generate eplusout.csv at end of simulation",
        required=False,
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--output",
        help="EnergyPlus output directory. Default is a generated one in /tmp/",
        required=False,
        default=TemporaryDirectory().name
    )
    parser.add_argument(
        "--timesteps", "-t",
        help="Number of timesteps to train",
        required=False,
        default=1e6
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="The number of workers to use",
    )
    parser.add_argument(
        "--alg",
        default="PPO",
        choices=["APEX", "DQN", "IMPALA", "PPO", "R2D2"],
        help="The algorithm to use",
    )
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="tf",
        help="The deep learning framework specifier",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Whether to auto-wrap the model with an LSTM. Only valid option for "
             "--run=[IMPALA|PPO|R2D2]",
    )
    built_args = parser.parse_args()
    #print(f"Running with following CLI args: {built_args}")
    return built_args


class EnergyPlusRunner:

    def __init__(self, episode: int, env_config: Dict[str, Any], obs_queue: Queue, act_queue: Queue) -> None:
        self.episode = episode
        self.env_config = env_config
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.curr = None # current time of the simulation run for output directory naming

        self.energyplus_api = EnergyPlusAPI()
        self.x: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized = False
        self.init_queue = Queue()
        self.progress_value: int = 0
        self.simulation_complete = False
        self.request_variable_complete = False


        # self.energyplus_api.exchange.request_variable(self.state, "SITE OUTDOOR AIR DRYBULB TEMPERATURE", "ENVIRONMENT")
        # self.energyplus_api.exchange.request_variable(self.state, "ZONE AIR TEMPERATURE", "LIVING_UNIT1")
        # self.energyplus_api.exchange.request_variable(self.state, "ZONE AIR TEMPERATURE", "ATTIC_UNIT1")

        # below is declaration of variables, meters and actuators
        # this simulation will interact with
        self.variables = {
            # °C
            "outdoor_temp" : ("Site Outdoor Air Drybulb Temperature", "Environment"),
            # °C
            "indoor_temp_living" : ("Zone Air Temperature", 'living_unit1'),
            # °C
            "indoor_temp_attic": ("Zone Air Temperature", 'attic_unit1'),


            # # °C
            # "oat": ("Site Outdoor Air DryBulb Temperature", "Environment"),
            # # °C
            # "iat": ("Zone Mean Air Temperature", "TZ_Amphitheater"),
            # # ppm
            # "co2": ("Zone Air CO2 Concentration", "TZ_Amphitheater"),
            # # heating setpoint (°C)
            # "htg_spt": ("Schedule Value", "HTG HVAC 1 ADJUSTED BY 1.1 F"),
            # # cooling setpoint (°C)
            # "clg_spt": ("Schedule Value", "CLG HVAC 1 ADJUSTED BY 0 F"),
        }
        self.var_handles: Dict[str, int] = {}

        self.meters = {
            # HVAC elec (J)
            "elec_hvac": "Electricity:HVAC",
            # Heating
            "elec_heating": "Heating:Electricity",
            # Cooling
            "elec_cooling": "Cooling:Electricity",

            'elec_facility': "Electricity:Facility",

            # probably not need based on html output
            'gas_heating': 'NaturalGas:HVAC'

            # District heating (J)
            # "dh": "Heating:DistrictHeating"
        }
        self.meter_handles: Dict[str, int] = {}

        self.actuators = {
            # NOTE: ZoneControl:Thermostat
            # supply air temperature setpoint (°C)
            # "sat_spt": (
            #     "System Node Setpoint",
            #     "Temperature Setpoint",
            #     "zone node_unit1"
            # ),
            "cooling_actuator_living" : (
                "Zone Temperature Control",
                "Heating Setpoint",
                "living_unit1"
            ),

            "heating_actuator_living" : (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "living_unit1"
            )

            # "test" : (
            #     "Zone"
            # )
        }
        self.actuator_handles: Dict[str, int] = {}

    def start(self) -> None:
        self.energyplus_state = self.energyplus_api.state_manager.new_state()
        runtime = self.energyplus_api.runtime
        if not self.request_variable_complete:
            for key, var in self.variables.items():
                self.x.request_variable(self.energyplus_state, var[0], var[1])
                self.request_variable_complete = True
                # (below) old way to request var
                # self.energyplus_api.exchange.request_variable(self.energyplus_state, "SITE OUTDOOR AIR DRYBULB TEMPERATURE", "ENVIRONMENT")
                # self.energyplus_api.exchange.request_variable(self.energyplus_state, "ZONE AIR TEMPERATURE", "LIVING_UNIT1")
                # self.energyplus_api.exchange.request_variable(self.energyplus_state, "ZONE AIR TEMPERATURE", "ATTIC_UNIT1")

        # register callback used to track simulation progress
        def report_progress(progress: int) -> None:
            self.progress_value = progress

        runtime.callback_progress(self.energyplus_state, report_progress)

        # register callback used to collect observations
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)

        # register callback used to send actions
        runtime.callback_after_predictor_after_hvac_managers(self.energyplus_state, self._send_actions)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(runtime, cmd_args, state, results):
            print(f"running EnergyPlus with args: {cmd_args}")

            # start simulation
            results["exit_code"] = runtime.run_energyplus(state, cmd_args)

        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(
                self.energyplus_api.runtime,
                self.make_eplus_args(),
                self.energyplus_state,
                self.sim_results
            )
        )
        self.energyplus_exec_thread.start()

    def stop(self) -> None:
        if self.energyplus_exec_thread:
            self.simulation_complete = True
            self._flush_queues()
            self.energyplus_exec_thread.join()
            self.energyplus_exec_thread = None
            self.energyplus_api.runtime.clear_callbacks()
            self.energyplus_api.state_manager.delete_state(self.energyplus_state)

    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        """
        make command line arguments to pass to EnergyPlus
        """
        self.curr = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        eplus_args = ["-r"] if self.env_config.get("csv", False) else []
        eplus_args += ['-x']
        eplus_args += [
            "-w",
            self.env_config["epw"],
            "-d",
            # change below for output directory name formatting
            f"{self.env_config['output']}/episode-{self.episode}-{datetime.now()}",
            # f"{self.env_config['output']}/episode-{self.episode:08}-{os.getpid():05}",
            self.env_config["idf"]
        ]
        print(eplus_args)
        return eplus_args

    def _collect_obs(self, state_argument) -> None:
        """
        EnergyPlus callback that collects output variables/meters
        values and enqueue them
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            # print('HIT COLLECT OBS')
            return

        # print("# OBS living",self.x.get_variable_value(state_argument, self.var_handles['indoor_temp_living']))
        # print("# OBS attic",self.x.get_variable_value(state_argument, self.var_handles['indoor_temp_attic']))

        self.next_obs = {
            **{
                key: self.x.get_variable_value(state_argument, handle)
                for key, handle
                in self.var_handles.items()
            },
            **{
                key: self.x.get_meter_value(state_argument, handle)
                for key, handle
                in self.meter_handles.items()
            }
        }
        self.obs_queue.put(self.next_obs)

    def _send_actions(self, state_argument):
        """
        EnergyPlus callback that sets actuator value from last decided action
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            # print('HIT SEND ACTIONS')
            return

        if self.act_queue.empty():
            return
        next_action = self.act_queue.get()
        assert isinstance(next_action[0], float)
        assert isinstance(next_action[1], float)

        #print(next_action)
        # self.x.set_actuator_value(
        #     state=state_argument,
        #     actuator_handle=self.actuator_handles["sat_spt"],
        #     actuator_value=next_action
        # )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['cooling_actuator_living'],
            actuator_value=next_action[0]
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['heating_actuator_living'],
            actuator_value=next_action[1]
        )
        #SCORES:  [20538820133.84012, 20538820133.84012]
        temp1 = self.x.get_actuator_value(state_argument,self.actuator_handles['cooling_actuator_living'])
        temp2 = self.x.get_actuator_value(state_argument, self.actuator_handles['heating_actuator_living'])
        indoor = self.x.get_variable_value(state_argument, self.var_handles['indoor_temp_living'])
        print('##', temp1, temp2)
        print('#####', indoor)
        #print('## ACTUATOR VAL:', temp)

    def _init_callback(self, state_argument) -> bool:
        """initialize EnergyPlus handles and checks if simulation runtime is ready"""
        self.initialized = self._init_handles(state_argument) and not self.x.warmup_flag(state_argument)
        # print('INIT HANDLES: ', self._init_handles(state_argument))
        # print('WARMUP FLAGS: ', self.x.warmup_flag(state_argument))
        # print('BOTH???', self._init_handles(state_argument) and self.x.warmup_flag(state_argument))
        # print(self.initialized)
        return self.initialized

    # TODO NOTE: some error with multiple request of handles -> WARNINGS for now but good to fix
    def _init_handles(self, state_argument):
        """initialize sensors/actuators handles to interact with during simulation"""
        # print(self.initialized)
        if not self.initialized:
            # print('WHY')
            if not self.x.api_data_fully_ready(state_argument):
                return False

            # requires requesting variables to energyplus
            # self.x.request_variable(state_argument, "Site Outdoor Air Drybulb Temperature", "Environment")
            # self.x.request_variable(state_argument, "Zone Air Temperature", "living_unit1")
            # self.x.request_variable(state_argument, "Zone Air Temperature", "attic_unit1")

            # print("###### HIT ######")
            # print(self.initialized == True)
            # print("HIIIIITTTT")

            self.var_handles = {
                key: self.x.get_variable_handle(state_argument, *var)
                for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: self.x.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: self.x.get_actuator_handle(state_argument, *actuator)
                for key, actuator in self.actuators.items()
            }

            for handles in [
                    self.var_handles,
                    self.meter_handles,
                    self.actuator_handles
            ]:
                if any([v == -1 for v in handles.values()]):
                    available_data = self.x.list_available_api_data_csv(state_argument).decode('utf-8')
                    print(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}"
                        # NOTE: commented out for now
                    )
                    exit(1)

            self.init_queue.put("")
            self.initialized = True
            # print('THIS SHOULD BE TRUE:', self.initialized)

        return True

    def _flush_queues(self):
        for q in [self.obs_queue, self.act_queue]:
            while not q.empty():
                q.get()


class EnergyPlusEnv(gym.Env):

    '''
    # OAT, IAT, CO2, cooling setpoint, heating setpoint, fans elec, district heating
    '''
    def __init__(self, env_config: Dict[str, Any]):
        self.env_config = env_config
        self.episode = -1
        self.timestep = 0

        # observation space:
        # outdoor_temp, indoor_temp_living, indoor_temp_attic
        # NOTE: I am unsure about the actual bound -> set as larger than expected values
        # TODO update this stuff
        low_obs = np.array(
            [-100.0, -100.0, -100.0]
        )
        hig_obs = np.array(
            [100.0, 100.0, 100.0]
        )
        self.observation_space = gym.spaces.Box(
            low=low_obs, high=hig_obs, dtype=np.float64
            # dtype was originally set to float32
        )
        self.last_obs = {}

        # action space: supply air temperature (100 possible values)
        # 20 - 24 degrees
        self.action_space: Discrete = Discrete(820)

        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None

    def reset(
            self, *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ):
        self.episode += 1
        self.last_obs = self.observation_space.sample()

        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

        # observations and actions queues for flow control
        # queues have a default max size of 1
        # as only 1 E+ timestep is processed at a time
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)

        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            env_config=self.env_config,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue
        )
        self.energyplus_runner.start()

        # wait for E+ warmup to complete
        if not self.energyplus_runner.initialized:
            self.energyplus_runner.init_queue.get()

        try:
            obs = self.obs_queue.get()
        except Empty:
            obs = self.last_obs

        return np.array(list(obs.values())), {}

    def step(self, action):
        self.timestep += 1
        done = False

        # check for simulation errors
        if self.energyplus_runner.failed():
            print(f"EnergyPlus failed with {self.energyplus_runner.sim_results['exit_code']}")
            sys.exit(1)

        # rescale agent decision to actuator range
        # sat_spt_value = self._rescale(
        #     n=int(action),  # noqa
        #     range1=(0, self.action_space.n),
        #     range2=(15, 30)
        # )
        #print('ACTION VAL:', sat_spt_value)
        sat_spt_value = self._rescale(int(action)) # maybe need int(action)

        # set the system temperature actuator value to sat_spt_value

        # api.exchange.set_actuator_value(state, outdoor_dew_point_actuator , 10000)
        # self.

        # enqueue action (received by EnergyPlus through dedicated callback)
        # then wait to get next observation.
        # timeout is set to 2s to handle start and end of simulation cases, which happens async
        # and materializes by worker thread waiting on this queue (EnergyPlus callback
        # not consuming yet/anymore)
        # timeout value can be increased if E+ warmup period is longer or if step takes longer
        timeout = 2
        try:
            self.act_queue.put(sat_spt_value, timeout=timeout)
            self.last_obs = obs = self.obs_queue.get(timeout=timeout)
        except (Full, Empty):
            done = True
            obs = self.last_obs

        #print('ACTION_Q: ', self.act_queue)

        # this won't always work (reason for queue timeout), as simulation
        # sometimes ends with last reported progress at 99%.
        if self.energyplus_runner.progress_value == 100:
            print("reached end of simulation")
            done = True

        # compute reward
        reward = self._compute_reward(obs)

        obs_vec = np.array(list(obs.values()))
        return obs_vec, reward, done, False, {}

    def render(self, mode="human"):
        pass

    @staticmethod
    def _compute_reward(obs: Dict[str, float]) -> float:
        """compute reward scalar"""
        # if obs["htg_spt"] > 0 and obs["clg_spt"] > 0:
        #     tmp_rew = np.diff([
        #         [obs["htg_spt"], obs["iat"]],
        #         [obs["iat"], obs["clg_spt"]]
        #     ])
        #     tmp_rew = tmp_rew[tmp_rew < 0]
        #     tmp_rew = np.max(np.abs(tmp_rew)) if tmp_rew.size > 0 else 0
        # else:
        #     tmp_rew = 0
        # reward = -(1e-7 * (obs["elec"] + obs["dh"])) - tmp_rew - (1e-3 * obs["co2"])

        #oa_temp = api.exchange.get_variable_value(state, outdoor_temp_sensor)

        #print('Heating', obs['heating_elec'], 'Cooling', obs['cooling_elec'])
        # reward = obs['elec']
        reward = obs['elec_heating'] + obs['elec_cooling']
        print('REWARD:', reward)
        # below is reward testing
        # reward = 10
        # print("#########################")
        # print(reward)
        # print("#########################")
        return reward

    @staticmethod
    def _rescale(
            n: int
    ) -> float:
        tuples = []
        for i in range(200,240):
            first_num = i/ 10.0
            for j in range (i + 1, 241):
                second_num = j / 10.0
                tuples.append(tuple([first_num, second_num]))

        #print(len(tuples))
        return tuples[n]

        # def _rescale(
        #     n: int,
        #     range1: Tuple[float, float],
        #     range2: Tuple[float, float]
        # ) -> float:
        #     delta1 = range1[1] - range1[0]
        #     delta2 = range2[1] - range2[0]
        #     return (delta2 * (n - range1[0]) / delta1) + range2[0]

#####################################################
#################          RL STUFF (DQN)     #######
#####################################################


#####################################################
#################      RL STUFF (DQN) end     #######
#####################################################



# TODO: have to give in -x flag
default_args = {'idf': '/home/ck/Downloads/Files/in.idf',
                'epw': '/home/ck/Downloads/Files/weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2
                }

# SCORES:  [81884676878.09312, 81884676878.09312]
#
#SCORES:  [76613073663.50632, 76613073663.50632]
if __name__ == "__main__":
    env = EnergyPlusEnv(default_args)
    print('action_space:', end='')
    print(env.action_space)
    scores = []
    for episode in range(2):
        state = env.reset()
        done = False
        score = 0

        while not done:
            #env.render()
            # action = env.action_space.sample()
            #action = 22.0
            ret = n_state, reward, done, info, _ = env.step(env.action_space.sample())
            #print('RET STUFF:', ret)
            score+=reward
            # print('DONE?:', done)
            print('Episode:{} Reward:{} Score:{}'.format(episode, reward, score))

        scores.append(score)
        print("SCORES: ", scores)
        print("TRULY DONE?") # YES, but program doesn't terminate due to threading stuff?
