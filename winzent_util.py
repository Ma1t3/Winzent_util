from typing import Tuple, Optional, List

from palaestrai.agent import SensorInformation


class WinzentSensorActuatorUtil:
    """Utility class for Winzent-palaestrAI-connection."""

    SENSOR_ID_GRID_JSON = "env.Powergrid-0.Grid-0.grid_json"

    @staticmethod
    def get_grid_json_from_sensors(
        sensors: List[SensorInformation],
    ) -> Optional[str]:
        """returns the grid_json as string from the corresponding sensor"""
        for sensor in sensors:
            if (
                sensor.sensor_id
                == WinzentSensorActuatorUtil.SENSOR_ID_GRID_JSON
            ):
                return sensor.sensor_value

        return None

    @staticmethod
    def get_element_type_and_index(
        sensor_or_actuator_id: str,
    ) -> Tuple[str, int]:
        """
        returns the type and the index of the element associated with the sensor or actuator
        example sensor id: env.Powergrid-0.0-load-0-15.p_mw
          -> elem_type = "load"
          -> index = 0
        """
        id_parts = sensor_or_actuator_id.split(".")
        grid_element = id_parts[2]
        element_id_parts = grid_element.split("-")
        if len(element_id_parts) >= 3:
            # e.g. 0-load-0-15
            elem_type = element_id_parts[1]
            index = int(element_id_parts[2])
        else:
            # e.g. Grid-0
            elem_type = element_id_parts[0]
            index = int(element_id_parts[1])

        return elem_type, index

    @staticmethod
    def get_sensor_or_actuator_type(sensor_or_actuator_id: str) -> str:
        """
        returns the type of the sensor or actuator
        example sensor id: env.Powergrid-0.0-load-0-15.p_mw
         -> type of sensor = "p_mw"
        """
        id_parts = sensor_or_actuator_id.split(".")
        sensor_or_actuator_type = id_parts[3]
        return sensor_or_actuator_type
