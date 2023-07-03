import asyncio
import logging

from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.sensor_information import SensorInformation
from palaestrai.types import Box

from pgasc.agents.winzent_agent_system.muscle import WinzentMuscle

"""Winzent script for testing integration of Winzent with the Palaestrai framework."""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
logging.getLogger("mango.core").setLevel(logging.CRITICAL)


def load_grid_json():
    with open("grid.json", "r", encoding="utf-8") as file:
        sensor_value = file.read()
    return sensor_value


def winzent_test_2_preparation():
    grid_json = load_grid_json()

    sensor_information_list = [
        SensorInformation(
            sensor_value=5,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-1-17.p_mw",
        ),
        SensorInformation(
            sensor_value=5,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-1-17.p_mw",
        ),
        SensorInformation(
            sensor_value=grid_json,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.Grid-0.grid_json",
        ),
    ]

    actuator_information_list = [
        ActuatorInformation(
            setpoint=0,
            action_space=Box(0, 1, shape=(1,)),
            actuator_id="env.Powergrid-0.0-sgen-1-17.scaling",
        ),
    ]

    muscle = WinzentMuscle("", "", "", "", "")
    return sensor_information_list, actuator_information_list, muscle


async def winzent_test_2():
    (
        sensor_information_list,
        actuator_information_list,
        muscle,
    ) = winzent_test_2_preparation()

    actuator_outputs, _, _, _ = muscle.propose_actions(
        sensors=sensor_information_list,
        actuators_available=actuator_information_list,
        is_terminal=True,
    )
    print("outputs: ", actuator_outputs[0])
    assert actuator_outputs[0].setpoint == 1


async def winzent_test_2_2x():
    (
        sensor_information_list,
        actuator_information_list,
        muscle,
    ) = winzent_test_2_preparation()

    actuator_outputs, _, _, _ = muscle.propose_actions(
        sensors=sensor_information_list,
        actuators_available=actuator_information_list,
        is_terminal=False,
    )
    print("outputs: ", actuator_outputs[0])
    assert actuator_outputs[0].setpoint == 1

    actuator_outputs, _, _, _ = muscle.propose_actions(
        sensors=sensor_information_list,
        actuators_available=actuator_information_list,
        is_terminal=True,
    )
    print("outputs: ", actuator_outputs[0])
    assert actuator_outputs[0].setpoint == 1


async def winzent_initialisation_test_3_negotiations():
    grid_json = load_grid_json()

    sensor_information_list = [
        SensorInformation(
            sensor_value=10,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-0-15.p_mw",  # agent25
        ),
        SensorInformation(
            sensor_value=5,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-1-17.p_mw",  # agent26
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-4-24.p_mw",  # agent29
        ),
        SensorInformation(
            sensor_value=1,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-0-16.p_mw",  # agent0
        ),
        SensorInformation(
            sensor_value=10,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-1-17.p_mw",  # agent1
        ),
        SensorInformation(
            sensor_value=20,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-2-20.p_mw",  # agent2
        ),
        SensorInformation(
            sensor_value=grid_json,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.Grid-0.grid_json",
        ),
    ]

    actuator_information_list = [
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-0-16.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-1-17.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-2-20.scaling",
        ),
    ]

    muscle = WinzentMuscle("", "", "", "", "")

    muscle.propose_actions(
        sensors=sensor_information_list,
        actuators_available=actuator_information_list,
    )


async def winzent_initialisation_test_2_negotiations():
    grid_json = load_grid_json()

    sensor_information_list = [
        SensorInformation(
            sensor_value=10,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-0-15.p_mw",  # agent25
        ),
        SensorInformation(
            sensor_value=5,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-1-17.p_mw",  # agent26
        ),
        SensorInformation(
            sensor_value=1,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-0-16.p_mw",  # agent0
        ),
        SensorInformation(
            sensor_value=5,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-1-17.p_mw",  # agent1
        ),
        SensorInformation(
            sensor_value=12,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-2-20.p_mw",  # agent2
        ),
        SensorInformation(
            sensor_value=grid_json,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.Grid-0.grid_json",
        ),
    ]

    actuator_information_list = [
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-0-16.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-1-17.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-2-20.scaling",
        ),
    ]

    muscle = WinzentMuscle("", "", "", "", "")

    muscle.propose_actions(
        sensors=sensor_information_list,
        actuators_available=actuator_information_list,
    )


async def winzent_initialisation_test():
    grid_json = load_grid_json()

    sensor_information_list = [
        SensorInformation(
            sensor_value=10,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-0-15.p_mw",
        ),
        SensorInformation(
            sensor_value=1,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-0-16.p_mw",
        ),
        SensorInformation(
            sensor_value=5,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-1-17.p_mw",
        ),
        SensorInformation(
            sensor_value=12,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-2-20.p_mw",
        ),
        SensorInformation(
            sensor_value=grid_json,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.Grid-0.grid_json",
        ),
    ]

    actuator_information_list = [
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-0-16.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-1-17.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-2-20.scaling",
        ),
    ]

    muscle = WinzentMuscle("", "", "", "", "")
    actuator_solution, x, y, z = muscle.propose_actions(
        sensors=sensor_information_list,
        actuators_available=actuator_information_list,
    )
    print(f"Final solution from the actuators {actuator_solution}")


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(winzent_initialisation_test_3_negotiations())


if __name__ == "__main__":
    main()
