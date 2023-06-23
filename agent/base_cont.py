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
import math
import time

import random

import scipy
import pickle

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

sys.path.insert(0, '/home/ck/Downloads/EnergyPlus-23.1.0-87ed9199d4-Linux-CentOS7.9.2009-x86_64/')
from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.datatransfer import DataExchange

'''

TODO ask
- diffuse solar radiation, wall or window?

TODO Testing Stuff:
- action values
- observation stuff

CURR TODO/DONE for this week:
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

    built_args = parser.parse_args()
    #print(f"Running with following CLI args: {built_args}")
    return built_args

#zone mean air temperature
# mean radiant temperature
# relative humidity
# radiant
# diffuse


# outdoor 32.2 8.9
# indoor 41.223511372068636 21.618788496151314
# mean_radiant_temp 42.37128185378897 22.155167975081476
# relative_humidity 59.65230658819031 24.260989719707105
# sky_diff_ldf 210.82257131667535 0.0
# sky_diff_sdr 230.72313848472825 0.0
# site_direct_solar 854.0 0.0
# site_horz 435.0 290.0

class EnergyPlusRunner:

    def __init__(self, episode: int, env_config: Dict[str, Any], obs_queue: Queue, act_queue: Queue, meter_queue: Queue, normalized_obs_queue: Queue) -> None:
        self.episode = episode
        self.env_config = env_config
        self.meter_queue = meter_queue
        self.obs_queue = obs_queue
        self.normalized_obs_queue = normalized_obs_queue
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
        # self.energyplus_api.exchange.request_variable(self.energyplus_state, "Surface Outside Face Solar Radiation Heat Gain Rate per Area", "living_unit1")
        # self.energyplus_api.exchange.request_variable(self.energyplus_state, "Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "living_unit1")

        # below is declaration of variables, meters and actuators
        # this simulation will interact with
        # handle_name : var_name, var_env, norm_method, OPTIONAL: bound
        self.variables = {
            # °C
            "outdoor_temp" : ("Site Outdoor Air Drybulb Temperature", "Environment", (8.9, 32.2)),
            # solar radiation
            # beam radiant
            # diffused radiant
            # °C
            "indoor_temp_living" : ("Zone Air Temperature", 'living_unit1', (15, 42)),
            # °C
            # "indoor_temp_attic": ("Zone Air Temperature", 'attic_unit1'), # NOTE: air temperature already have? NOTE: attic temp not needed?

            # °C, surface area times emissivity
            "mean_radiant_temperature_living": ("Zone Mean Radiant Temperature", "living_unit1", (22.155, 42.371)),

            # %, air relative humidity after the correct step for each zone
            "relative_humidity_living": ("Zone Air Relative Humidity", "living_unit1", (24.26, 59.652)),

            # m/s air velocity
            # "air_velocity_living": ("", ""),

            #  Diffuse Radiation [W]
            #'interior_diffuse_radiation_living' : ('Zone Interior Windows Total Transmitted Diffuse Solar Radiation Rate', 'living_unit1'), #NOTE, interior window is not present in the model
            #########'exterior_diffuse_radiation_living' : ('Zone Exterior Windows Total Transmitted Diffuse Solar Radiation Rate', 'living_unit1'),
            # "solar_radiation_ldf1": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Wall_ldf_1.unit1"),
            # "solar_radiation_ldf2": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Wall_ldf_2.unit1"),
            # "solar_radiation_ldb1": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Wall_ldb_1.unit1"),
            # "solar_radiation_ldb2": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Wall_ldb_2.unit1"),

            # "solar_radiation_ldf1": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Window_ldf_1.unit1"),
            # "solar_radiation_ldf2": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Window_ldf_2.unit1"),
            # "solar_radiation_ldb1": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Window_ldb_1.unit1"),
            # "solar_radiation_ldb2": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Window_ldb_2.unit1"),
            # "solar_radiation_floor": ("Surface Outside Face Solar Radiation Heat Gain Rate per Area", "Inter zone floor 1"),
            # "solar_radiation_ldr1": ("Surface Inside Face Solar Radiation Heat Gain Rate per Area", "Wall_ldr_1.unit1"),
            # "solar_radiation_ldr2": ("Surface Inside Face Solar Radiation Heat Gain Rate per Area", "Wall_ldr_2.unit1"),
            # "solar_radiation_ldl1": ("Surface Inside Face Solar Radiation Heat Gain Rate per Area", "Wall_ldl_1.unit1"),
            # "solar_radiation_ldl2": ("Surface Inside Face Solar Radiation Heat Gain Rate per Area", "Wall_ldl_2.unit1"),

            # Beam Radiation [W]
            #'interior_beam_radiation_living': ('Zone Interior Windows Total Transmitted Beam Solar Radiation Rate', 'living_unit1'), # NOTE, interior window not present
            #####'exterior_beam_radiation_living': ('Zone Exterior Windows Total Transmitted Beam Solar Radiation Rate', 'living_unit1')
            # "test_ldf1": ("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "Window_ldf_1.unit1"),
            # "test_ldf2": ("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "Window_ldf_2.unit1"),
            # "test_ldb1": ("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "Window_ldb_1.unit1"),
            # "test_ldb2": ("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "Window_ldb_2.unit1"),
            # "beam_radiation_ldf1": ("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "Wall_ldf_1.unit1"),
            # "beam_radiation_ldf2": ("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "Wall_ldf_2.unit1"),
            # "beam_radiation_ldb1": ("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "Wall_ldb_1.unit1"),
            # "beam_radiation_ldb2": ("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area", "Wall_ldb_2.unit1"),

            # Direct Solar Radiation Rate per Area [W/m^2]
            # Diffuse Solar Radiation Rate per Area [W/m^2]

            # Sky Diffuse Solar Radiation [W/m^2]
            # NOTE value of ldf1 == value of ldf2
            # NOTE value of ldb1 == value of ldb2
            # 'sky_diffuse_solar_ldf1': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", "Window_ldf_1.unit1"),
            # 'sky_diffuse_solar_ldf2': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", "Window_ldf_2.unit1"),
            # 'sky_diffuse_solar_ldb1': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", "Window_ldb_1.unit1"),
            # 'sky_diffuse_solar_ldb2': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", "Window_ldb_2.unit1"),
            # DONE: since they are same reduce the # of varialbes to:
            'sky_diffuse_solar_ldf': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_ldf_1.unit1', (0, 210.822)),
            #'sky_diffuse_solar_ldb': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_ldb_1.unit1'),
            # DONE
            'sky_diffuse_solar_sdr': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_sdr_1.unit1', (0, 230.723)),
            #'sky_diffuse_solar_sdl': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_sdl_1.unit1'),

            # Ground Diffuse Solar Radiation [W/m^2]
            # NOTE value of ldf1 == ldf2 == ldb1 == ldb2
            # 'ground_diffuse_solar_ldf1': ("Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area", "Window_ldf_1.unit1"),
            # 'ground_diffuse_solar_ldf2': ("Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area", "Window_ldf_2.unit1"),
            # 'ground_diffuse_solar_ldb1': ("Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area", "Window_ldb_1.unit1"),
            # 'ground_diffuse_solar_ldb2': ("Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area", "Window_ldb_2.unit1"),
            # NOTE: they all yield same value so narrow them to single var:
            # 'ground_diffuse_solar': ("Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area", 'Window_ldf_1.unit1'),
            # 'ground_diffuse_solar_2': ("Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area", 'Window_sdr_1.unit1'),
            # 'ground_diffuse_solar_3': ("Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area", 'Window_sdl_1.unit1'),

            # DONE DONE
            'site_direct_solar': ("Site Direct Solar Radiation Rate per Area", "Environment", (0, 854)),
            'site_horizontal_infrared': ("Site Horizontal Infrared Radiation Rate per Area", "Environment", (290, 435)),

            #'test_ldl': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_ldl_1.unit1'),
            # 'diffuse_solar_ldf2': ("", ""),
            # 'diffuse_solar_ldb1': ("", ""),
            # 'diffuse_solar_ldb2': ("", "")
            # Horizontal Infrared Radiation Rate per Area [W/m^2]
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

            # probably not need based on html output NOTE: meter handle not found
            # 'gas_heating': 'NaturalGas:HVAC',

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
                "Cooling Setpoint",
                "living_unit1"
            ),

            "heating_actuator_living" : (
                "Zone Temperature Control",
                "Heating Setpoint",
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

        # requesting variables
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
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_meter)

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
        # eplus_args += ['-a'] # NOTE: enforces simulation to be annual (runtime start = Jan1)

        eplus_args += ["-a"] if self.env_config.get('annual', False) else []
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

    def _collect_meter(self, state_argument) -> None:
        '''
        For addressing -> values used to calculate rewards also seen in observation of the agent
        '''
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        self.next_meter = {
            **{
                key: self.x.get_meter_value(state_argument, handle)
                for key, handle
                in self.meter_handles.items()
            }
        }
        self.meter_queue.put(self.next_meter)

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
        }

        # normalize each of the observations to range of [-1, 1] using linear interpolation
        temp = dict()
        for key, item in self.var_handles.items():
            #print('key', key, 'item', item)
            temp[key] = np.interp(self.next_obs[key], [self.variables[key][2][0], self.variables[key][2][1]], [-1, 1])
        self.normalized_next_obs = temp

        # post process obs state
        self._process_obs(state_argument)

        self.normalized_obs_queue.put(self.normalized_next_obs)
        self.obs_queue.put(self.next_obs)

    def _process_obs(self, state_argument) -> None:
        '''
        Called after _collect_obs, and get_variable_value at state
        use this function to add post-processing to the obs:
        --- such as adding more states, etc

        NOTE: process_obs is called after the inputs have been normalized
        in place manipulation

        eg:
        self.next_obs  = {
        'outdoor_temp': 16.95,
        'indoor_temp_living': 28.936114602861167,
        'mean_radiant_temperature_living': 30.106807344102126,
        'relative_humidity_living': 37.53592428294955,
        'sky_diffuse_solar_ldf': 0.0,
        'sky_diffuse_solar_sdr': 0.0,
        'site_direct_solar': 0.0,
        'site_horizontal_infrared': 328.0
        }
        '''

        # Day Of Week / Hour for Demand Response
        day_of_week = self.x.day_of_week(self.energyplus_state) # in the range (1-7 where 1: sunday)
        hour = self.x.hour(self.energyplus_state) # in the range (0-24)
        #print('day_of_week', day_of_week, 'hour', hour)
        hour_of_week = (24 * (day_of_week - 1)) + hour
        normalized_hour_of_week = np.interp(hour_of_week, [0, 167], [-1, 1])
        self.normalized_next_obs['hour_of_week'] = normalized_hour_of_week
        self.next_obs['hour_of_week'] = hour_of_week

        # Cost Rate Signal (not price and only signal)
        cost_rate_signal = self._compute_cost_rate_signal()
        normalized_cost_rate_signal = np.interp(cost_rate_signal, [1, 4], [-1, 1])
        self.normalized_next_obs['cost_rate_signal'] = normalized_cost_rate_signal
        self.next_obs['cost_rate_signal'] = cost_rate_signal
        return None


    def _compute_cost_rate_signal(self) -> float:
        '''returns the cost rate at current timestep.
        NOTE?: Use demand signal (eg. 1, 2, 3, 4)
        '''
        hour = self.x.hour(self.energyplus_state)
        minute = self.x.minutes(self.energyplus_state)
        day_of_week = self.x.day_of_week(self.energyplus_state)
        if day_of_week in [1, 7]:
            # weekend pricing
            if hour in range(0, 7) or hour in range(23, 24 + 1): # plus one is to include 7
                #cost_rate = 2.4
                return 1
            elif hour in range(7, 23):
                #cost_rate = 7.4
                return 2
        else:
            if hour in range(0, 7) or hour in range(23, 24 + 1):
                #cost_rate = 2.4
                return 1
            elif hour in range(7, 16) or hour in range(21, 23):
                #cost_rate = 10.2
                return 3
            elif hour in range(16, 21):
                #cost_rate = 24.0
                return 4

    def _rescale(self, action, old_range_min, old_range_max, new_range_min, new_range_max):
        '''
        _rescale method can be used for larger range to smaller range
        '''
        old_range = old_range_max - old_range_min
        new_range = new_range_max - new_range_min
        return (((action - old_range_min) * new_range) / old_range) + new_range_min

    def _send_actions(self, state_argument):
        """
        EnergyPlus callback that sets actuator value from last decided action
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            # print('HIT SEND ACTIONS')
            return

        if self.act_queue.empty():
            return
        next_action = self.act_queue.get()[0]
        next_action = self._rescale(next_action, -1, 1, 15, 30)

        assert isinstance(next_action, float) or isinstance(next_action, np.float32) # for Box action space, next_action dtype will be float32
        assert next_action >= 15

        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['cooling_actuator_living'],
            actuator_value=next_action
            # actuator_value=20.0
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles['heating_actuator_living'],
            actuator_value=0 # NOTE: set it to a extreme low temp so it's never triggered
            # actuator_value=15.0
        )
        temp1 = self.x.get_actuator_value(state_argument,self.actuator_handles['cooling_actuator_living'])
        temp2 = self.x.get_actuator_value(state_argument, self.actuator_handles['heating_actuator_living'])
        indoor = self.x.get_variable_value(state_argument, self.var_handles['indoor_temp_living'])

    def _init_callback(self, state_argument) -> bool:
        """initialize EnergyPlus handles and checks if simulation runtime is ready"""
        self.initialized = self._init_handles(state_argument) and not self.x.warmup_flag(state_argument)
        # print('INIT HANDLES: ', self._init_handles(state_argument))
        # print('WARMUP FLAGS: ', self.x.warmup_flag(state_argument))
        # print('BOTH???', self._init_handles(state_argument) and self.x.warmup_flag(state_argument))
        # print(self.initialized)
        return self.initialized

    #  DONE NOTE: some error with multiple request of handles -> WARNINGS for now but good to fix
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
                key: self.x.get_variable_handle(state_argument, var[0], var[1])
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
        # self.start_date = env_config['start_date']
        # self.end_date = env_config['end_date']
        self.start_date = datetime(2000, env_config['start_date'][0], env_config['start_date'][1])
        self.end_date = datetime(2000, env_config['end_date'][0], env_config['end_date'][1])


        self.acceptable_pmv = 0.1

        # Caching PMV values to accelerate
        # NOTE: key: (tr, rh), val : (low, high)
        self.using_pmv_cache = env_config['pmv_pickle_available']
        self.PMV_CACHE = dict()
        self.PMV_CACHE_PATH = env_config['pmv_pickle_path']
        if self.using_pmv_cache:
            self.pickle_load_pmv_cache()

        # observation space:
        # outdoor_temp, indoor_temp_living, mean_radiant_temperature_living, relative_humidity_living, exterior_diffuse_radiation_living, exterior_beam_radiation_living
        # NOTE: I am unsure about the actual bound -> set as larger than expected values
        # TODO update this stuff
        # low_obs = np.array(
        #     [-100.0, -100.0, -100.0, 0, 0]
        # )
        # high_obs = np.array(
        #     [100.0, 100.0, 100.0, 100.0, 100000000.0]
        # )
        # low_obs = np.array(
        #     [-100.0, -100.0, -100.0, 0, 0, 0, 0, 0]
        # )
        # high_obs = np.array(
        #     [100.0, 100.0, 100.0, 100.0, 100000000.0, 100000000.0, 100000000.0, 100000000.0]
        # )

        low_obs = np.array(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1] #10
        )
        high_obs = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
        self.observation_space = gym.spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float64
            # dtype was originally set to float32
        )
        self.last_obs = {}
        self.normalized_last_obs = {}

        # last obs saves it in dictionary format, but last_next_state saves the numpy vec
        self.last_next_state = None
        self.normalized_last_next_state = None

        # action space: np.linspace(15,30,0.1)
        self.action_space: Box = Box(np.array([15]), np.array([30]), dtype=np.float32)

        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.meter_queue: Optional[Queue] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None
        self.normalized_obs_queue: Optional[Queue] = None

    def _rescale(self, action, old_range_min, old_range_max, new_range_min, new_range_max):
        '''
        _rescale already implemented for EnergyPlusRunner class, but for convenience, implemented
        for EnergyPlusEnv
        '''
        old_range = old_range_max - old_range_min
        new_range = new_range_max - new_range_min
        return (((action - old_range_min) * new_range) / old_range) + new_range_min

    def pickle_save_pmv_cache(self):
        '''
        if pmv_pickle_avaiable is True -> self.using_pmv_cache is True
        then even is pickle_save_pmv_cache() is called, it won't save
        '''
        if not self.using_pmv_cache:
            with open(self.PMV_CACHE_PATH, 'wb') as handle:
                pickle.dump(self.PMV_CACHE, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load_pmv_cache(self):
        with open(self.PMV_CACHE_PATH, 'rb') as handle:
            p = pickle.load(handle)
            self.PMV_CACHE = p

    def masking_valid_actions(self, scale:tuple =(-1, 1)) -> tuple:
        '''
        for Policy Gradient methods, find valid action values of the indoor air temperature
        NOTE: valid action value will be
        NOTE: make sure this function is called after the obs_vec has been updated
        to the current time step

        has caching feature to self.PMV_CACHE. Cache is statically saved to self.PMV_CACHE_PATH

        NOTE: note that the function returns the scaled values
        '''
        #print("HITTTTT")
        def f(x):
            tr = self.last_next_state[2]
            rh = self.last_next_state[3]
            return abs(self._compute_reward_thermal_comfort(x, tr, 0.1, rh)) - self.acceptable_pmv

        # try fetch PMV_CACHE
        tr = self.last_next_state[2]
        rh = self.last_next_state[3]
        cache = self.PMV_CACHE.get((round(tr, 3), round(rh, 3)), False)
        if cache:
            # print('using cache!') # NOTE: checked that cache works
            return cache

        pivot = None
        xs = (x * 0.5 for x in range(0,31))
        for x in xs:
            #print('x', x+15, 'f(x)', f(x + 15))
            if f(x + 15) < 0:
                pivot = x + 15
        if pivot == None:
            self.PMV_CACHE[(round(tr, 3), round(rh, 3))] = scale
            #return (15, 30)
            #return scale
            # return (-1.0, -0.7333333333333334)
            ret = scipy.optimize.minimize(f, 20, method="Powell").x[0]
            print(ret - 0.1, ret + 0.1)
            return (ret - 1e-5, ret + 1e-5)
        else:
            root1 = scipy.optimize.brentq(f, pivot, pivot + 20)
            root2 = scipy.optimize.brentq(f, pivot, pivot - 20)
            root1_scaled = self._rescale(root1, self.action_space.low[0], self.action_space.high[0], scale[0], scale[1])
            root2_scaled = self._rescale(root2, self.action_space.low[0], self.action_space.high[0], scale[0], scale[1])
            self.PMV_CACHE[(round(tr, 3), round(rh, 3))] = (root2_scaled, root1_scaled)
            return (root2_scaled, root1_scaled) # tuple([lower root, higher root])


    def retrieve_actuators(self):
        #temp1 = self.x.get_actuator_value(state_argument,self.actuator_handles['cooling_actuator_living'])
        #temp2 = self.x.get_actuator_value(state_argument, self.actuator_handles['heating_actuator_living'])
        runner = self.energyplus_runner
        cooling_actuator_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['cooling_actuator_living'])
        heating_actuator_value = runner.x.get_actuator_value(runner.energyplus_state, runner.actuator_handles['heating_actuator_living'])
        return (cooling_actuator_value, heating_actuator_value)

    def reset(
            self, *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ):
        self.episode += 1
        self.last_obs = self.observation_space.sample()

        if self.energyplus_runner is not None:
            # print('EEEEEEEEEEEEEEEEEEEEEEEE')
            # print('STOPPPP')
            # print('EEEEEEEEEEEEEEEEEEEEEEEE')
            self.energyplus_runner.stop()

        # observations and actions queues for flow control
        # queues have a default max size of 1
        # as only 1 E+ timestep is processed at a time
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)
        self.meter_queue = Queue(maxsize=1)
        self.normalized_obs_queue = Queue(maxsize=1)

        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            env_config=self.env_config,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
            meter_queue=self.meter_queue,
            normalized_obs_queue = self.normalized_obs_queue
        )
        self.energyplus_runner.start()

        # wait for E+ warmup to complete
        if not self.energyplus_runner.initialized:
            self.energyplus_runner.init_queue.get()

        try:
            obs = self.obs_queue.get()
            meter = self.meter_queue.get()
            normalized_obs = self.normalized_obs_queue.get()
        except Empty:
            meter = self.last_meter
            obs = self.last_obs
            normalized_obs = self.normalized_last_obs

        obs_vec = np.array(list(obs.values()))
        normalized_obs_vec = np.array(list(normalized_obs.values()))

        # update the self.last_next_state
        self.last_next_state = obs_vec
        self.normalized_last_next_state = normalized_obs_vec

        return_vec = obs_vec

        return return_vec
    #return np.array(list(obs.values())), np.array(list(meter.values())) #

    def step(self, action):
        '''
        @params: action -> numpy.ndarray w/ 1 element
        '''
        # simulation time values
        #current_time = self.energyplus_runner.x.current_sim_time(self.energyplus_runner.energyplus_state)
        # current_date = self.energyplus_runner.x.day_of_month()
        # current_month = self.energyplus_runner.x.day_of_year()
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
        #sat_spt_value = self._rescale(int(action)) # maybe need int(action)
        #sat_spt_value = action[0]
        sat_spt_value = np.float32(action)

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
            self.last_meter = meter = self.meter_queue.get(timeout=timeout)
            self.normalized_last_obs = normalized_obs = self.normalized_obs_queue.get(timeout=timeout)
        except (Full, Empty):
            done = True
            obs = self.last_obs
            meter = self.last_meter
            normalized_obs = self.normalized_last_obs
            # NOTE: from this point below, obs is updated
            # noticed cases where obs are same as prev (some bottleneck with simulation)
        obs_vec = np.array(list(obs.values()))
        normalized_obs_vec = np.array(list(normalized_obs.values()))

        # NOTE: set observation type
        ret_obs_vec = normalized_obs_vec

        # update the self.last_next_state
        self.last_next_state = obs_vec
        self.normalized_last_next_state = normalized_obs_vec

        # compute energy reward
        reward_energy = self._compute_reward_energy(meter)
        # compute thermal comfort reward
        reward_thermal_comfort = self._compute_reward_thermal_comfort(
            obs_vec[1],
            obs_vec[2],
            0.1, #NOTE: for now set as 0.1, but find if E+ can generate specific values
            obs_vec[3]
        )
        # compute reward cost
        reward_cost = self._compute_reward_cost(meter) # watts * cost (cents/kWh)

        reward_watts = self._compute_reward_energy_watts(meter)

        current_cost_rate = self._compute_cost_rate_signal()

        reward_energy_times_cost_rate = reward_energy * current_cost_rate

        reward = reward_energy_times_cost_rate
        # reward = reward_energy
        print('reward', reward)

        # NOTE: HARD-spiking penalty
        # PENALTY = None
        # if abs(reward_thermal_comfort) > self.acceptable_pmv:
        #     PENALTY = -1e20
        # else:
        #     PENALTY = 0

        #NOTE: soft-penalty sigmoid with adjustable penalty param
        # PENALTY_COEFF = -7000
        # def penalty_sigmoid(x):
        #     return PENALTY_COEFF * (1 / (1 + math.exp(-(x-0))))
        # PENALTY = None
        # if abs(reward_thermal_comfort) > self.acceptable_pmv:
        #     penalty_factor = abs(reward_thermal_comfort) - self.acceptable_pmv
        #     PENALTY = penalty_sigmoid(penalty_factor)
        #     print('pen', PENALTY)
        # else:
        #     PENALTY = 0

        # NOTE: soft-penalty linear
        # PENALTY = None
        # PENALTY_COEFF = -22000
        # def penalty_linear(pmv_diff):
        #     return PENALTY_COEFF * pmv_diff
        # if abs(reward_thermal_comfort) > self.acceptable_pmv:
        #     penalty_factor = abs(reward_thermal_comfort) - self.acceptable_pmv
        #     PENALTY = penalty_linear(penalty_factor)
        #     #print('pen', PENALTY, penalty_factor)
        # else:
        #     PENALTY = 0

        PENALTY = 0

        year = self.energyplus_runner.x.year(self.energyplus_runner.energyplus_state)
        month = self.energyplus_runner.x.month(self.energyplus_runner.energyplus_state)
        day = self.energyplus_runner.x.day_of_month(self.energyplus_runner.energyplus_state)
        hour = self.energyplus_runner.x.hour(self.energyplus_runner.energyplus_state)
        minute = self.energyplus_runner.x.minutes(self.energyplus_runner.energyplus_state)

        # NOTE: -a flag is required therefore, manually alter the runtime
        #print('DATE', month, day)
        # curr_date = datetime(2000, month, day)
        # if curr_date < self.start_date:
        #     # if before simulation start date -> return 0 as reward
        #     return obs_vec, 0, False, False, {'date': (month, day),
        #                                       'actuators': self.retrieve_actuators(),
        #                                       'energy_reward': reward_energy,
        #                                       'comfort_reward': reward_thermal_comfort}
        # if curr_date > self.end_date:
        #     # if past simulation end date -> done = True
        #     # actuators[0] -> cooling, actuators[1] -> heating
        #     return obs_vec, (reward_energy + PENALTY), True, False, {'date': (month, day),
        #                                                              'actuators' : self.retrieve_actuators(),
        #                                                              'energy_reward': reward_energy,
        #                                                              'comfort_reward': reward_thermal_comfort}


        # this won't always work (reason for queue timeout), as simulation
        # sometimes ends with last reported progress at 99%.
        # NOTE: changed this to 99
        #print("PROGRESS: ", self.energyplus_runner.progress_value)
        if self.energyplus_runner.progress_value == 99:
            print("reached end of simulation")
            done = True

        # print('THERMAL COMFORT:', thermal_comfort)

        #print('ACTION VAL:',action, sat_spt_value, "OBS: ", obs_vec[:])
        return ret_obs_vec, (reward + PENALTY), done, False, {'date': (month, day),
                                                                 'actuators' : self.retrieve_actuators(),
                                                                 'energy_reward': reward_energy,
                                                                 'comfort_reward': reward_thermal_comfort,
                                                                 'cost_reward': reward_cost,
                                                                 'year': year,
                                                                 'month': month,
                                                                 'day': day,
                                                                 'hour': hour,
                                                                 'minute': minute,
                                                                 'obs_vec': obs_vec
                                                          }

    def b_during_sim(self):
        '''
        ret boolean value of whether sim is running/not
        '''
        month = self.energyplus_runner.x.month(self.energyplus_runner.energyplus_state)
        day = self.energyplus_runner.x.day_of_month(self.energyplus_runner.energyplus_state)
        curr_date = datetime(2000, month, day)
        if curr_date < self.start_date or curr_date > self.end_date:
            return False
        else:
            return True

    def render(self, mode="human"):
        # TODO? : maybe add IDF visualization option
        pass

    @staticmethod
    def _compute_reward_thermal_comfort(tdb, tr, v, rh) -> float:
        '''
        @params
        tdb: dry bulb air temperature
        tr: mean radiant temperature
        v: used to calculate v_relative: air velocity
        rh: relative humidity
        met: set as a constant value of 1.4
        clo: set as a constant value of 0.5
        -> clo_relative is pre-computed ->

        @return PPD
        '''
        def pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme):
            pa = rh * 10 * math.exp(16.6536 - 4030.183 / (tdb + 235))

            icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
            m = met * 58.15  # metabolic rate in W/M2
            w = wme * 58.15  # external work in W/M2
            mw = m - w  # internal heat production in the human body
            # calculation of the clothing area factor
            if icl <= 0.078:
                f_cl = 1 + (1.29 * icl)  # ratio of surface clothed body over nude body
            else:
                f_cl = 1.05 + (0.645 * icl)

            # heat transfer coefficient by forced convection
            hcf = 12.1 * math.sqrt(vr)
            hc = hcf  # initialize variable
            taa = tdb + 273
            tra = tr + 273
            t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

            p1 = icl * f_cl
            p2 = p1 * 3.96
            p3 = p1 * 100
            p4 = p1 * taa
            p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
            xn = t_cla / 100
            xf = t_cla / 50
            eps = 0.00015

            n = 0
            while abs(xn - xf) > eps:
                xf = (xf + xn) / 2
                hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
                if hcf > hcn:
                    hc = hcf
                else:
                    hc = hcn
                    xn = (p5 + p4 * hc - p2 * xf**4) / (100 + p3 * hc)
                    n += 1
                if n > 150:
                    raise StopIteration("Max iterations exceeded")

            tcl = 100 * xn - 273

            # heat loss diff. through skin
            hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
            # heat loss by sweating
            if mw > 58.15:
                hl2 = 0.42 * (mw - 58.15)
            else:
                hl2 = 0
                # latent respiration heat loss
            hl3 = 1.7 * 0.00001 * m * (5867 - pa)
            # dry respiration heat loss
            hl4 = 0.0014 * m * (34 - tdb)
            # heat loss by radiation
            hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
            # heat loss by convection
            hl6 = f_cl * hc * (tcl - tdb)

            ts = 0.303 * math.exp(-0.036 * m) + 0.028
            _pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

            return _pmv


        def v_relative(v, met):
            """Estimates the relative air speed which combines the average air speed of
            the space plus the relative air speed caused by the body movement. Vag is assumed to
            be 0 for metabolic rates equal and lower than 1 met and otherwise equal to
            Vag = 0.3 (M – 1) (m/s)

            Parameters
            ----------
            v : float or array-like
            air speed measured by the sensor, [m/s]
            met : float
            metabolic rate, [met]

            Returns
            -------
            vr  : float or array-like
            relative air speed, [m/s]
            """
            return np.where(met > 1, np.around(v + 0.3 * (met - 1), 3), v)

        def clo_dynamic(clo, met, standard="ASHRAE"):
            """Estimates the dynamic clothing insulation of a moving occupant. The activity as
            well as the air speed modify the insulation characteristics of the clothing and the
            adjacent air layer. Consequently, the ISO 7730 states that the clothing insulation
            shall be corrected [2]_. The ASHRAE 55 Standard corrects for the effect
            of the body movement for met equal or higher than 1.2 met using the equation
            clo = Icl × (0.6 + 0.4/met)

            Parameters
            ----------
            clo : float or array-like
            clothing insulation, [clo]
            met : float or array-like
            metabolic rate, [met]
            standard: str (default="ASHRAE")
            - If "ASHRAE", uses Equation provided in Section 5.2.2.2 of ASHRAE 55 2020

            Returns
            -------
            clo : float or array-like
            dynamic clothing insulation, [clo]
            """
            standard = standard.lower()
            if standard not in ["ashrae", "iso"]:
                raise ValueError(
                    "only the ISO 7730 and ASHRAE 55 2020 models have been implemented"
                )

            if standard == "ashrae":
                return np.where(met > 1.2, np.around(clo * (0.6 + 0.4 / met), 3), clo)
            else:
                return np.where(met > 1, np.around(clo * (0.6 + 0.4 / met), 3), clo)


        #
        clo_dynamic = 0.443 # precomputed with the clo value of 0.5 (clo_dynamic(0.5, 1.4))
        v_rel = v_relative(v, 1.4)
        #print('V_REL', v_rel)
        pmv = pmv_ppd_optimized(tdb, tr, 0.1, rh, 1.4, clo_dynamic, 0)
        # now calc and return ppd
        return pmv
    #return 100.0 - 95.0 * np.exp(-0.03353 * np.power(pmv, 4.0) - 0.2179 * np.power(pmv, 2.0))

    @staticmethod
    def _compute_reward_energy(meter: Dict[str, float]) -> float:
        """compute reward scalar"""
        reward = -1 * meter['elec_cooling']
        return reward

    @staticmethod
    def _compute_reward_energy_watts(meter: Dict[str, float]) -> float:
        '''
        doesn't work with base watts, works if miliwatts (* 1000)
        '''
        reward = -1 * meter['elec_cooling']
        reward_watt = reward / (10 * 60)
        #reward_kilowatt = reward_watt / 1000
        reward_kilowatt = reward_watt
        return reward_kilowatt # * 1000

    # NOTE: moved to EnergyPlusRunner class for _process_obs
    # NOTE NEW: used for computing the cost rate at timestep t
    def _compute_cost_rate_signal(self) -> float:
        '''returns the cost rate at current timestep.
        NOTE?: Use demand signal (eg. 1, 2, 3, 4)
        '''
        hour = self.energyplus_runner.x.hour(self.energyplus_runner.energyplus_state)
        minute = self.energyplus_runner.x.minutes(self.energyplus_runner.energyplus_state)
        day_of_week = self.energyplus_runner.x.day_of_week(self.energyplus_runner.energyplus_state)
        if day_of_week in [1, 7]:
            # weekend pricing
            if hour in range(0, 7) or hour in range(23, 24 + 1): # plus one is to include 7
                return 2.4
                # return 1
            elif hour in range(7, 23):
                return 7.4
        else:
            if hour in range(0, 7) or hour in range(23, 24 + 1):
                return 2.4
            elif hour in range(7, 16) or hour in range(21, 23):
                return 10.2
            elif hour in range(16, 21):
                return 24.0


    #@staticmethod
    def _compute_reward_cost(self, meter: Dict[str, float]) -> float:
        '''
        NOTE: peak hours and corresponding cost
        ultra-low overnight: Everyday from 11pm to 7am :2.4 C
        weekend off-peak: Weekends and holidays from 7am to 11pm: 7.4 C
        mid-peak: weekdays from 7am to 4pm, 9pm to 11pm
        on-peak: weekdays from 4pm to 9pm
        https://www.torontohydro.com/for-home/rates

        @param: tuple(hour, min)
        hour: 0 -> 12am
        @return: cents
        '''
        cost_rate = None
        hour = self.energyplus_runner.x.hour(self.energyplus_runner.energyplus_state)
        minute = self.energyplus_runner.x.minutes(self.energyplus_runner.energyplus_state)
        day_of_week = self.energyplus_runner.x.day_of_week(self.energyplus_runner.energyplus_state)
        if day_of_week in [1, 7]:
            # weekend pricing
            if hour in range(0, 7) or hour in range(23, 24 + 1): # plus one is to include 7
                cost_rate = 2.4
            elif hour in range(7, 23):
                cost_rate = 7.4
        else:
            if hour in range(0, 7) or hour in range(23, 24 + 1):
                cost_rate = 2.4
            elif hour in range(7, 16) or hour in range(21, 23):
                cost_rate = 10.2
            elif hour in range(16, 21):
                cost_rate = 24.0

        watt_usage = self._compute_reward_energy_watts(meter)
        return (watt_usage / 1000) * cost_rate



# NOTE: have to give in -x flag
# default_args = {'idf': '/home/ck/Downloads/Files/in.idf',
#                 'epw': '/home/ck/Downloads/Files/weather.epw',
#                 'csv': True,
#                 'output': './output',
#                 'timesteps': 1000000.0,
#                 'num_workers': 2
#                 }

default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                'start_date': (6,21), # DEPRECATED -> fixed the idf running problem
                'end_date': (8,21),
                'pmv_pickle_available': True,
                'pmv_pickle_path': './pmv_cache.pickle'
                }
#
#SCORES:  [-343068118.4928892, -343058929.74458027, -343034573.5644406, -343063839.9638236, -343081534.0729704, -343076762.9154123, -343055059.71841764, -343033258.9391935, -343036122.53581744, -343047720.2466282]
if __name__ == "__main__":
    env = EnergyPlusEnv(default_args)
    print('action_space:', end='')
    print(env.action_space)
    print("OBS SHAPE:", env.observation_space.shape)
    scores = []

    for episode in range(10):
        state = env.reset()
        done = False
        score = 0

        while not done:
            temp = env.masking_valid_actions()
            #print(temp)
            action = env.action_space.sample()
            #print(action)
            action = [temp[1]]
            ret = n_state, reward, done, truncated, info = env.step(action)

            #print('n_state', n_state, len(n_state))
            # print('DATE', info['date'][0], info['date'][1], 'REWARD:', reward, 'ACTION:', action[0])
            score+=info['energy_reward']


        env.pickle_save_pmv_cache()
        scores.append(score)
        print("SCORES: ", scores)
    print("TRULY DONE?") # YES, but program doesn't terminate due to threading stuff?
