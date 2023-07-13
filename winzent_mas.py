import io
import logging
import warnings
from typing import Optional, Dict

import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp
from mango.core.container import Container
from mango_library.negotiation.winzent.winzent_classic_agent import WinzentClassicAgent
from mango_library.negotiation.winzent.winzent_simple_ethical_agent import WinzentSimpleEthicalAgent

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class WinzentAgentWithInfo(WinzentSimpleEthicalAgent):
    """WinzentAgent with additional information about its corresponding element in the power grid"""

    def __init__(
        self,
        container,
        elem_type,
        index,
        ttl,
        time_to_sleep,
        send_message_paths,
        ethics_score
    ):
        super().__init__(
            container,
            ttl=ttl,
            time_to_sleep=time_to_sleep,
            send_message_paths=send_message_paths,
            ethics_score = ethics_score
        )
        self.elem_type = elem_type
        self.index = index


class WinzentMAS:
    """Class to represent a Multi-Agent System (MAS) of WinzentAgents. The class provides methods to
    build and update the topology of the MAS based on the topology of the power grid."""

    ELEMENT_TYPES_WITH_AGENTS = ["sgen", "load", "ext_grid", "bus"]
    CONTAINER_ADDR = ("0.0.0.0", 5555)

    def __init__(
        self, ttl, time_to_sleep, grid_json: str, send_message_paths: bool, ethics_score_config
    ) -> None:
        """Creates a new WinzentMAS based on the given grid_json."""
        self.send_message_paths = send_message_paths
        self._container = None
        self._net = pp.from_json(io.StringIO(grid_json))
        self.ethics_score_config = ethics_score_config
        # print(type(self._net["load"]))
        # print(self._net["sgen"])
        # all winzent agents as dictionary (e.g. self.winzent_agents["bus"][34] returns bus with index 35)
        self.winzent_agents = {
            elem_type: {} for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS
        }
        self.ttl = ttl
        self.time_to_sleep = time_to_sleep
        # agent_id: winzent_agent
        self.aid_agent_mapping: Dict[str, WinzentAgentWithInfo] = {}
        self.graph = nx.DiGraph()
        self.most_ethical_agents = []

    async def create_winzent_agents(self):
        """Creates all WinzentAgents for the elements in the power grid."""
        self._container = await Container.factory(
            addr=WinzentMAS.CONTAINER_ADDR
        )
        for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS:
            for index in self._net[elem_type].index:
                winzent_agent = self._create_agent(elem_type, index)
                self.winzent_agents[elem_type][index] = winzent_agent
                self.aid_agent_mapping[winzent_agent.aid] = winzent_agent
                self.graph.add_node(winzent_agent.aid)

    def build_topology(self):
        """Builds the topology of the MAS based on the topology of the power grid."""
        for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS:
            for index, agent in self.winzent_agents[elem_type].items():
                connected_bus_indices = self._get_connected_buses(
                    self._net, elem_type, index
                )
                for bus_index in connected_bus_indices:
                    bus_agent = self.get_agent("bus", bus_index)
                    if bus_agent is None:
                        logger.critical("Could not create topology")
                    self._add_neighbors(agent, bus_agent)

    def check_changes_and_update_topolgy(self, grid_json: str):
        """Checks if the topology of the power grid has changed and updates the topology of the MAS accordingly."""
        new_net = pp.from_json(io.StringIO(grid_json))
        for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS:
            for index, agent in self.winzent_agents[elem_type].items():
                connected_bus_indices = self._get_connected_buses(
                    new_net, elem_type, index
                )
                old_connected_bus_indices = self._get_connected_buses(
                    self._net, elem_type, index
                )

                disconnected_bus_indices = (
                    old_connected_bus_indices.difference(connected_bus_indices)
                )
                new_connected_bus_indices = connected_bus_indices.difference(
                    old_connected_bus_indices
                )
                self._update_neighborhoods(
                    agent, disconnected_bus_indices, new_connected_bus_indices
                )

        self._net = new_net

    async def shutdown(self):
        """Shutdown all agents and the container."""
        for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS:
            for agent in self.winzent_agents[elem_type].values():
                await agent.stop_agent()
                await agent.shutdown()
        await self._container.shutdown()

    def get_agent(self, elem_type, index) -> Optional[WinzentAgentWithInfo]:
        """Returns the WinzentAgent corresponding to the given element (with elem_type and index) in the power grid."""
        if elem_type in self.winzent_agents:
            if index in self.winzent_agents[elem_type]:
                return self.winzent_agents[elem_type][index]
        return None

    def save_plot(self, filename):
        """Saves a plot of the topology of the MAS to the given filename."""
        labels = {aid: aid[5:] for aid in self.graph.nodes}
        plt.title("WinzentMAS topology (labels are agent ids)")
        nx.draw(self.graph, node_size=100, font_size=7, labels=labels)
        plt.savefig(filename)

    def _create_agent(self, elem_type, index):
        return WinzentAgentWithInfo(
            container=self._container,
            elem_type=elem_type,
            index=index,
            ttl=self.ttl,
            time_to_sleep=self.time_to_sleep,
            send_message_paths=self.send_message_paths,
            ethics_score=self._assign_ethics_score(self._net[elem_type].at[index,"name"])
        )

    def _get_connected_buses(self, net, elem_type, index):
        if elem_type == "bus":
            return pp.get_connected_buses(
                net,
                buses=index,
                respect_switches=True,
                respect_in_service=True,
            )
        else:
            return {net[elem_type].at[index, "bus"]}

    def _add_neighbors(self, agent_1, agent_2):
        # winzent's add neighbor method adds or replaces the agent (no duplicates in neighborhood)
        agent_1.add_neighbor(aid=agent_2.aid, addr=WinzentMAS.CONTAINER_ADDR)
        agent_2.add_neighbor(aid=agent_1.aid, addr=WinzentMAS.CONTAINER_ADDR)
        self.graph.add_edge(agent_1.aid, agent_2.aid)

    def _delete_neighbors(self, agent_1, agent_2):
        agent_1.delete_neighbor(aid=agent_2.aid)
        agent_2.delete_neighbor(aid=agent_1.aid)

    def _update_neighborhoods(
        self, agent, disconnected_bus_indices, new_connected_bus_indices
    ):
        for bus_index in disconnected_bus_indices:
            bus_agent = self.get_agent("bus", bus_index)
            if bus_agent is None:
                logger.critical("Could not create topology")
            self._delete_neighbors(agent, bus_agent)

        for bus_index in new_connected_bus_indices:
            bus_agent = self.get_agent("bus", bus_index)
            if bus_agent is None:
                logger.critical("Could not create topology")
            self._add_neighbors(agent, bus_agent)

    def _assign_ethics_score_old(self, name):
        if "Klinikum" in name or "PV" in name or "Wind" in name:
            return 3
        elif "Households" in name or "Abfall" in name:
            return 1
        else:
            return 2

    def _assign_ethics_score(self, name, index):
        ethics_values = list(self.ethics_score_config.keys())
        for value in ethics_values:
            if any(string in name for string in self.ethics_score_config[value]):
                if value == max(ethics_values):
                    self.most_ethical_agents.append("agent" + str(index))
                return value
        return min(ethics_values)
