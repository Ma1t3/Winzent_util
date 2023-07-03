import asyncio
import csv
import logging

from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.sensor_information import SensorInformation
from palaestrai.types import Box

from pgasc.agents.winzent_agent_system.muscle import WinzentMuscle

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
logging.getLogger("mango.core").setLevel(logging.INFO)

"""Winzent script for testing integration of Winzent with the Palaestrai framework."""


def load_grid_json():
    with open("grid.json", "r", encoding="utf-8") as file:
        sensor_value = file.read()
    return sensor_value


def load_sensor_information_list():
    sensor_information_list = {}
    with open("sensor_values.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] not in sensor_information_list.keys():
                sensor_information_list[row[0]] = [row[1]]
            else:
                sensor_information_list[row[0]].append(row[1])
    return sensor_information_list


def load_actuator_information_list():
    actuator_information_list = []
    with open("actuator_values.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            actuator_information_list.append(
                ActuatorInformation(
                    setpoint=0,
                    action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
                    actuator_id=str(row),
                )
            )
    return actuator_information_list


async def winzent_test():
    max_steps = 10
    grid_json = load_grid_json()
    params = {
        "step_size": 900,
        "ttl": 120,
        "time_to_sleep": 2,
        "factor_mw": 1000000,
        "number_of_restartable_negotiations": 40,
        "send_message_paths": True,
    }
    muscle = WinzentMuscle("", "", "", "", "", **params)
    actuator_information_list = load_actuator_information_list()
    sensor_information_list = []
    sensor_information_list_all_steps = load_sensor_information_list()
    for current_step in range(max_steps):
        for sensor in sensor_information_list_all_steps.keys():
            sensor_information_list.append(
                SensorInformation(
                    sensor_value=float(
                        sensor_information_list_all_steps[sensor][current_step]
                    ),
                    observation_space=None,  # type: ignore
                    sensor_id=sensor,
                ),
            )
        sensor_information_list.append(
            SensorInformation(
                sensor_value=grid_json,
                observation_space=None,  # type: ignore
                sensor_id="env.Powergrid-0.Grid-0.grid_json",
            ),
        )

        muscle.propose_actions(
            sensors=sensor_information_list,
            actuators_available=actuator_information_list,
        )


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(winzent_test())


if __name__ == "__main__":
    main()
