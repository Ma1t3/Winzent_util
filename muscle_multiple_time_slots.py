import asyncio
import logging
import math
import time
from typing import Optional, Dict, List, Tuple

import nest_asyncio
from palaestrai.agent import Muscle, SensorInformation, ActuatorInformation

from .winzent_mas import WinzentMAS
from .winzent_util import WinzentSensorActuatorUtil
from mango_library.negotiation.winzent.winzent_base_agent import WinzentBaseAgent

logger = logging.getLogger(__name__)


class WinzentMuscle(Muscle):
    def __init__(self, broker_uri, brain_uri, uid, brain_id, path, **params):
        super().__init__(broker_uri, brain_uri, uid, brain_id, path)
        self.step_size = params.get("step_size", 900)
        self.end = eval(params.get("end", 24 * 60 * 60))
        self.ttl = params.get("ttl", 80)
        self.time_to_sleep = params.get("time_to_sleep", 10)
        self.factor_mw = params.get("factor_mw", 1000000)
        self.number_of_restartable_negotiations = params.get(
            "number_of_restartable_negotiations", 40
        )
        self.ethics_score_config = params.get("ethics_score_config", None)
        self.send_message_paths = params.get("send_message_paths", True)
        self.request_processing_waiting_time = float(params.get("request_processing_waiting_time", 0.4))
        self.reply_processing_waiting_time = params.get("reply_processing_waiting_time", 0.4)
        self.use_producer_ethics_score = params.get("use_producer_ethics_score", True)
        self.use_consumer_ethics_score = params.get("use_consumer_ethics_score", True)

        self.decay_rate = 0
        self.sub_tier_size = 0
        self.ethics_score_list = {}
        self.calc_ethics_score_params()

        self.initialized = False
        self.time = 0
        self.winzent_mas: Optional[WinzentMAS] = None
        # mapping: sensor list to (type, agent)
        self.sensor_mapping: List[
            Tuple[str, Optional[WinzentBaseAgent]]
        ] = []
        # mapping: actuator list to (type, agent)
        self.actuator_mapping: List[
            Tuple[str, Optional[WinzentBaseAgent]]
        ] = []

        # To get solution to the actuators
        self.initial_generator_values: Dict[str:int] = {}
        self.initial_grid_json = None
        self.rounded_load_values: Dict[str:int] = {}
        self.final_solution = {}

        self.messages_sent_in_step = 0

    def calc_ethics_score_params(self):
        total_amount_of_steps = self.end / self.step_size
        self.sub_tier_size = 1.0 / total_amount_of_steps
        self.decay_rate = self.sub_tier_size / total_amount_of_steps
        self.reset_ethics_score_list()

    def reset_ethics_score_list(self):
        for key in self.ethics_score_config.keys():
            self.ethics_score_list[key] = [0.0, 0, 0]

    def create_sensor_and_actuator_mapping(
            self,
            sensors: List[SensorInformation],
            actuators: List[ActuatorInformation],
    ):
        for sensor in sensors:
            (
                elem_type,
                index,
            ) = WinzentSensorActuatorUtil.get_element_type_and_index(
                sensor.sensor_id
            )
            agent = self.winzent_mas.get_agent(
                elem_type, index
            )  # returns None if no corresponding agent exists
            sensor_type = (
                WinzentSensorActuatorUtil.get_sensor_or_actuator_type(
                    sensor.sensor_id
                )
            )
            self.sensor_mapping.append((sensor_type, agent))

        for actuator in actuators:
            (
                elem_type,
                index,
            ) = WinzentSensorActuatorUtil.get_element_type_and_index(
                actuator.actuator_id
            )
            agent = self.winzent_mas.get_agent(
                elem_type, index
            )  # returns None if no corresponding agent exists
            actuator_type = (
                WinzentSensorActuatorUtil.get_sensor_or_actuator_type(
                    actuator.actuator_id
                )
            )
            self.actuator_mapping.append((actuator_type, agent))

    def update_flexibilities(self, sensors):
        # example sensor_id: env.Powergrid-0.0-load-0-15.p_mw
        # example sensor_id: env.Powergrid-0.0-load-0-15.q_mvar

        for sensor, (sensor_type, agent) in zip(sensors, self.sensor_mapping):
            if agent is not None:
                if agent.elem_type == "sgen" and sensor_type == "p_mw_flex":
                    flexibility = sensor.sensor_value * self.factor_mw
                    self.initial_generator_values[agent.aid] = flexibility
                    agent.update_flexibility(
                        t_start=self.time,
                        min_p=0,
                        max_p=math.floor(flexibility),
                    )
                elif agent.elem_type == "load" and sensor_type == "p_mw":
                    # print(f"{agent.aid}: {sensor.sensor_value}")
                    self.rounded_load_values[agent.aid] = math.ceil(
                        sensor.sensor_value * self.factor_mw
                    )
                    agent.update_flexibility(
                        t_start=self.time, min_p=0, max_p=0
                    )

        logger.debug(
            f"initial generator values: {self.initial_generator_values}"
        )

    async def run_negotiations(self, sensors):
        self.messages_sent_in_step = 0
        agents_with_started_negotiation = []
        time_span = [self.time]
        # start a negotiation for every load with the new value
        for sensor, (sensor_type, agent) in zip(sensors, self.sensor_mapping):
            if sensor_type == "p_mw" and agent is not None:
                if agent.elem_type == "load":
                    if sensor.sensor_value > 0:
                        # only start negotiation if value is larger than 0
                        logger.debug(
                            f"Start negotiation for {agent.aid} with value {math.ceil(sensor.sensor_value * self.factor_mw)}"
                        )
                        await agent.start_negotiation(
                            start_dates=time_span,
                            values=[math.ceil(sensor.sensor_value * self.factor_mw)],
                        )
                        agents_with_started_negotiation.append(agent)

        number_of_restarted_negotiations = (
            self.number_of_restartable_negotiations
        )  # restriction of restarted negotiations

        logger.debug(
            "All initial negotiations started; waiting for negotiations to be done and restarting "
            "unsuccessful negotiations"
        )
        # waiting for all negotiations to finish
        while len(agents_with_started_negotiation) > 0:
            agent = agents_with_started_negotiation.pop(0)
            try:
                await asyncio.wait_for(agent.negotiation_done, timeout=agent.time_to_sleep * 3)
                logger.debug(f"{agent.aid} negotiation done")
                # restart unsuccessful negotiations
                # only allow a restricted number of restarts
                agent_result_sum = []
                for time_slot in time_span:
                    agent_result_sum.append(0)
                for key in agent.final:
                    it = 0
                    for sub_key in agent.final[key]:
                        agent_result_sum[it] += abs(agent.final[key][sub_key][0])
                        it += 1
                # check if negotiation fulfills requirements
                negotiation_successful = agent_result_sum >= self.rounded_load_values[agent.aid]
                if not negotiation_successful:
                    if number_of_restarted_negotiations > 0:
                        # get sum of already negotiated values for this agent
                        # negotiation was not fully successful, therefore restart
                        agents_with_started_negotiation.append(agent)
                        # restart the negotiation with the missing value
                        await agent.start_negotiation(
                            start_dates=time_span,
                            values=[self.rounded_load_values[agent.aid]
                                  - agent_result_sum],
                        )
                        logger.debug(
                            f"{agent.aid} restarted negotiation for value "
                            f"of {self.rounded_load_values[agent.aid] - agent_result_sum}"
                        )
                        number_of_restarted_negotiations -= 1
                    elif agent_result_sum > self.rounded_load_values[agent.aid]:
                        logger.error(
                            f"Too much power: {agent.aid} has with a sum of {agent_result_sum} instead of "
                            f"{self.rounded_load_values[agent.aid]} from the result:{agent.result} "
                            f"not a feasible solution"
                        )
                    else:
                        agent.ethics_score = self.calculate_new_ethics_score(negotiation_successful, agent.ethics_score)
                        self.save_ethics_score_development(self.ethics_score_list, agent, negotiation_successful)
                else:
                    agent.ethics_score = self.calculate_new_ethics_score(negotiation_successful, agent.ethics_score)
                    self.save_ethics_score_development(self.ethics_score_list, agent, negotiation_successful)
            except asyncio.TimeoutError:
                logger.error(f"{agent.aid} could not finish its negotiation in time. No restart permission can be given.")
                agent.ethics_score = self.calculate_new_ethics_score(False, agent.ethics_score)
                self.save_ethics_score_development(self.ethics_score_list, agent, False)
        logger.info(f"ethics_scores -->{self.ethics_score_list}")
        self.reset_ethics_score_list()

    def save_negotiated_solution_by_load(self):
        self.final_solution = {}
        for agent in self.winzent_mas.winzent_agents["load"].values():
            logger.debug(f"muscle: LOAD {agent.aid} result:{agent.result}")
            for sgen in agent.result.keys():
                if sgen not in self.final_solution.keys():
                    self.final_solution[sgen] = 0
                self.final_solution[sgen] += agent.result[sgen]
            # reset result for next step
            agent.result = {}

    def save_number_of_sent_msg(self):
        """
        Saves the number of sent messages in every
        :return:
        """
        for agent_type in self.winzent_mas.winzent_agents.keys():
            for agent in self.winzent_mas.winzent_agents[agent_type].values():
                self.messages_sent_in_step += agent.messages_sent
                agent.messages_sent = 0

    def set_actuator_setpoints(self, actuators_available):
        # after negotiation fetch the results and give them to the actuators
        # for later winzent versions
        logger.debug("final solution of what winzent has negotiated")
        logger.debug(self.final_solution)
        logger.info(f"agent types: {self.winzent_mas.agent_types}")
        for actuator, (actuator_type, agent) in zip(
                actuators_available, self.actuator_mapping
        ):
            if actuator_type == "scaling" and agent is not None:
                logger.debug(self.final_solution)
                if (
                        agent.aid in self.final_solution
                        and self.initial_generator_values[agent.aid] > 0
                ):

                    value = (
                            self.final_solution[agent.aid]
                            / self.initial_generator_values[agent.aid]
                    )
                    logger.debug(
                        f"final solution: {self.final_solution[agent.aid]} and initial generator values {self.initial_generator_values[agent.aid]}; actuator value = {value}"
                    )
                    if value > 1:
                        logger.info(f"final solution: {self.final_solution[agent.aid]} and initial generator values {self.initial_generator_values[agent.aid]}; actuator value = {value}")
                        value = 1
                        logger.info(
                            "WARNING: Invalid Winzent result detected."
                            "Winzent negotiation results are altered to"
                            "avoid the experiment from crashing."
                        )
                    actuator(value)
                    for key, value_list in self.winzent_mas.agent_types.items():
                        if agent.aid in value_list:
                            print(f"PRODUCED {self.final_solution[agent.aid]} {key} {value} {agent.aid}")
                else:
                    logger.debug("actuator set to zero")
                    actuator(0)

    async def run_step(self, sensors, actuators):
        self.time += self.step_size
        self.initial_generator_values = {}
        self.final_solution = {}
        self.update_flexibilities(sensors)
        await self.run_negotiations(sensors)
        self.save_negotiated_solution_by_load()
        self.save_number_of_sent_msg()
        self.set_actuator_setpoints(actuators)

    async def run_winzent(
            self, sensors, actuators_available, is_terminal=False
    ):
        logger.info("Winzent next step running")
        grid_json = WinzentSensorActuatorUtil.get_grid_json_from_sensors(
            sensors
        )

        # initialize Winzent
        if not self.initialized:
            if grid_json:
                logger.info("Winzent is initializing")
                self.winzent_mas = WinzentMAS(
                    ttl=self.ttl,
                    time_to_sleep=self.time_to_sleep,
                    grid_json=grid_json,
                    send_message_paths=self.send_message_paths,
                    ethics_score_config = self.ethics_score_config,
                    use_consumer_ethics_score=self.use_consumer_ethics_score,
                    use_producer_ethics_score=self.use_producer_ethics_score,
                    request_processing_waiting_time=self.request_processing_waiting_time,
                    reply_processing_waiting_time=self.reply_processing_waiting_time,
                )
                await self.winzent_mas.create_winzent_agents()
                self.winzent_mas.build_topology()
                self.create_sensor_and_actuator_mapping(
                    sensors, actuators_available
                )
                self.initialized = True
                logger.info("Winzent is initialized")
                self.initial_grid_json = grid_json
            else:
                logger.error(
                    "Could not initialize Winzent. Please add grid_json to sensor list!"
                )
                return

        # In every new episode, palaestrai returns the integer 0 instead of a string containing the grid json
        # if this is the case: use the grid json saved in the first step
        if grid_json == 0:
            logger.info("Grid json is 0, using initial grid json")
            grid_json = self.initial_grid_json

        if grid_json:
            self.winzent_mas.check_changes_and_update_topolgy(grid_json)
            logger.debug("Topology was updated")
        else:
            logger.info("No grid json received, don't update topology")

        # logging
        for agents in self.winzent_mas.winzent_agents.values():
            for agent in agents.values():
                logger.debug(
                    f"{agent.aid} {agent.elem_type} index: {agent.index} with {agent.neighbors}"
                )

        start_time = time.time()
        # run a Winzent step
        await self.run_step(sensors, actuators_available)
        runtime = time.time() - start_time

        # logging information
        network_flexibility = 0
        for i in self.initial_generator_values.values():
            network_flexibility += i

        needed_load = 0
        for sensor in sensors:
            if "load" in sensor.id:
                needed_load += sensor.sensor_value * self.factor_mw

        actual_value = 0
        for i in self.final_solution.values():
            actual_value += i
        logger.info(
            f"Flexibility of the network: (0, {network_flexibility}) [{network_flexibility / self.factor_mw}] \n"
            f"Needed Loads: {needed_load} [{needed_load / self.factor_mw}] \n "
            f"Actual negotiated value: {actual_value} [{actual_value / self.factor_mw}] \n"
            f"Messages sent: {self.messages_sent_in_step} \n"
            f"Runtime: {runtime}"
        )
        logger.info(
            "network flexibility, needed loads, negotiated value, number of sent messages, runtime"
        )
        logger.info(
            f"{network_flexibility / self.factor_mw}, {needed_load / self.factor_mw}, {actual_value / self.factor_mw}, "
            f"{self.messages_sent_in_step}, {runtime}"
        )

        if is_terminal:
            await self.winzent_mas.shutdown()
            logger.info("Winzent has shut down all agents")
        logger.info(f"Winzent step {self.time} finished")

    def propose_actions(
            self, sensors, actuators_available, is_terminal=False
    ) -> tuple:
        """The state of the environment (sensor inputs) is given to winzent, which calculates
        a solution and after a successful run in winzent the solution is sent to the environment
        through the actuators"""

        # sensor list: SensorInformation(value=1, observation_space=Discrete(2), sensor_id=myenv.0),
        # with all the sensors that the agent is given in the erd
        nest_asyncio.apply()
        asyncio.run(
            self.run_winzent(sensors, actuators_available, is_terminal)
        )

        # only first list will be sent to environment, rest is for brain
        return (
            actuators_available,
            actuators_available,
            [1 for _ in actuators_available],
            {},
        )

    def save_ethics_score_development(self, ethics_score_list, agent, success):
        print(f"save ethics score from {agent.aid}")
        ethics_score_tiers = list(ethics_score_list.keys())
        if success == False and agent.ethics_score >= ethics_score_tiers[0]:
            logger.info(f"{agent.aid}: High priority target not supplied.\n Solution is {agent.result_sum} and target supply is {self.rounded_load_values[agent.aid]}")
        for tier in ethics_score_tiers:
            if tier <= agent.ethics_score < tier + 1.0:
                ethics_score_list[tier][0] = ethics_score_list[tier][0] + agent.ethics_score
                ethics_score_list[tier][2] += 1
                if not success:
                    ethics_score_list[tier][1] += 1

    def calculate_new_ethics_score(self, success, ethics_score):
        max_len_of_ethics_score = "{:." + str(len(str(self.decay_rate).replace('.', ''))) + "f}"
        initial_ethics_score = float(math.floor(ethics_score))
        str_eth_score = list(str(ethics_score))
        str_eth_score[0] = "0"
        str_eth_score = float("".join(str_eth_score))
        amount_of_outages = int(str_eth_score / self.sub_tier_size)
        current_tier_low = max(float(str(ethics_score)[0]) + (amount_of_outages * (self.sub_tier_size)),
                               initial_ethics_score)
        current_tier_high = max(float(str(ethics_score)[0]) + ((amount_of_outages + 1) * self.sub_tier_size),
                                initial_ethics_score)
        if not success:
            temp = math.floor(ethics_score * 10) / 10
            if (math.floor(float(temp)) + 1) > (float(temp) + self.sub_tier_size):
                if ethics_score == initial_ethics_score:
                    return float(
                        max_len_of_ethics_score.format(initial_ethics_score + self.sub_tier_size - self.decay_rate))
                return float(max_len_of_ethics_score.format(current_tier_high + self.sub_tier_size - self.decay_rate))
            else:
                return float(max_len_of_ethics_score.format((math.floor(float(ethics_score)) + 1) - self.decay_rate))
        else:
            temp_ethics_score = float(max_len_of_ethics_score.format(ethics_score - self.decay_rate))
            if temp_ethics_score <= current_tier_low:
                return current_tier_low
            else:
                return temp_ethics_score

    def setup(self):
        pass

    def update(self, update):
        """Emtpy because Winzent does not have brain and therefore does not change its strategy"""
        pass

    def __repr__(self):
        pass

    def prepare_model(self):
        """Emtpy because Winzent does not have a trained model"""
        pass
