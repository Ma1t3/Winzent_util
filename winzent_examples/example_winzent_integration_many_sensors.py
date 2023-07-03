import asyncio
import logging

from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.sensor_information import SensorInformation
from palaestrai.types import Box

from pgasc.agents.winzent_agent_system.muscle import WinzentMuscle

"""Winzent script for testing integration of Winzent with the Palaestrai framework."""

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
logging.getLogger("mango.core").setLevel(logging.CRITICAL)


def load_grid_json():
    with open("grid.json", "r", encoding="utf-8") as file:
        sensor_value = file.read()
    return sensor_value


async def winzent_test():
    grid_json = load_grid_json()

    sensor_information_list = [
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-0-15.p_mw",  # agent25
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-1-17.p_mw",  # agent26
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-2-19.p_mw",  # agent27
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-3-20.p_mw",  # agent28
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-4-24.p_mw",  # agent29
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-5-25.p_mw",  # agent30
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-6-28.p_mw",  # agent31
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-7-30.p_mw",  # agent32
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-8-32.p_mw",  # agent33
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-9-33.p_mw",  # agent34
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-10-35.p_mw",  # agent35
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-11-37.p_mw",  # agent36
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-12-40.p_mw",  # agent37
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-13-44.p_mw",  # agent38
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-14-45.p_mw",  # agent39
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-15-47.p_mw",  # agent40
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-16-48.p_mw",  # agent41
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-17-49.p_mw",  # agent42
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-18-50.p_mw",  # agent43
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-19-52.p_mw",  # agent44
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-20-53.p_mw",  # agent45
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-21-56.p_mw",  # agent46
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-22-57.p_mw",  # agent47
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-23-59.p_mw",  # agent48
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-24-60.p_mw",  # agent49
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-25-61.p_mw",  # agent50
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-26-62.p_mw",  # agent51
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-27-63.p_mw",  # agent52
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-28-65.p_mw",  # agent53
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-29-67.p_mw",  # agent54
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-30-69.p_mw",  # agent55
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-31-70.p_mw",  # agent56
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-32-71.p_mw",  # agent57
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-33-73.p_mw",  # agent58
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-34-74.p_mw",  # agent59
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-35-75.p_mw",  # agent60
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-36-76.p_mw",  # agent61
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-37-77.p_mw",  # agent62
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-38-78.p_mw",  # agent63
        ),
        SensorInformation(
            sensor_value=3,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-load-39-80.p_mw",  # agent64
        ),
        # SGEN
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-0-16.p_mw_flex",  # agent0
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-1-17.p_mw_flex",  # agent1
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-2-20.p_mw_flex",  # agent2
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-3-22.p_mw_flex",  # agent3
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-4-23.p_mw_flex",  # agent4
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-5-25.p_mw_flex",  # agent5
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-6-28.p_mw_flex",  # agent6
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-7-30.p_mw_flex",  # agent7
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-8-32.p_mw_flex",  # agent8
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-9-33.p_mw_flex",  # agent9
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-10-35.p_mw_flex",  # agent10
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-11-37.p_mw_flex",  # agent11
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-12-39.p_mw_flex",  # agent12
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-13-40.p_mw_flex",  # agent13
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-14-43.p_mw_flex",  # agent14
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-15-45.p_mw_flex",  # agent15
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-16-50.p_mw_flex",  # agent16
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-17-53.p_mw_flex",  # agent17
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-18-55.p_mw_flex",  # agent18
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-19-57.p_mw_flex",  # agent19
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-20-63.p_mw_flex",  # agent20
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-21-65.p_mw_flex",  # agent21
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-22-67.p_mw_flex",  # agent22
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-23-71.p_mw_flex",  # agent23
        ),
        SensorInformation(
            sensor_value=4,
            observation_space=None,  # type: ignore
            sensor_id="env.Powergrid-0.0-sgen-24-80.p_mw_flex",  # agent24
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
            actuator_id="env.Powergrid-0.0-sgen-3-22.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-4-23.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-5-25.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-6-28.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-7-30.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-8-32.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-9-33.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-10-35.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-11-37.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-12-39.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-13-40.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-14-43.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-15-45.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-16-50.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-17-53.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-18-55.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-19-57.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-20-63.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-21-65.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-22-67.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-23-71.scaling",
        ),
        ActuatorInformation(
            setpoint=0,
            action_space=Box(low=0, high=1, shape=(1,)),  # type: ignore
            actuator_id="env.Powergrid-0.0-sgen-24-80.scaling",
        ),
    ]
    params = {
        "step_size": 900,
        "ttl": 120,
        "time_to_sleep": 2,
        "factor_mw": 1000000,
        "number_of_restartable_negotiations": 40,
        "send_message_paths": True,
    }
    muscle = WinzentMuscle("", "", "", "", "", **params)

    muscle.propose_actions(
        sensors=sensor_information_list,
        actuators_available=actuator_information_list,
    )


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(winzent_test())


if __name__ == "__main__":
    main()
