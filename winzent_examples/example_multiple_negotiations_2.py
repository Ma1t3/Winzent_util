import asyncio
import logging

from agents.winzent_agent_system.winzent.winzent_agent import (
    WinzentAgent,
)
from mango.core.container import Container

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
logging.getLogger("mango.core").setLevel(logging.CRITICAL)

"""Winzent example script for multiple negotiations."""


async def run_negotiations():
    """
    Run multiple negotiations.
    """
    (
        agent_a,
        agent_b,
        agent_c,
        agent_d,
        container,
    ) = await four_agents_another_topology()

    print("\n Start new negotiation with enough flexibility. ")
    await winzent_test(agent_a, agent_b, agent_c, agent_d)

    # print('\n Start new negotiaion with not enough flexibility. ')
    # await not_enough_flexibility(agent_a, agent_b, agent_c)
    # await agent_b.negotiation_done

    await shutdown(agent_a, agent_b, agent_c, agent_d, container)


async def successful_negotiation(agent_a, agent_b, agent_c, agent_d):
    """
    In this case, the agents receive enough flexibility to solve the
    negotiation request.
    """
    # first, hand flexibility to agents
    agent_a.update_flexibility(t_start=900, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=900, min_p=0, max_p=20)
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=900, min_p=0, max_p=10)

    await agent_a.start_negotiation(ts=[900, 1800], value=40)
    await agent_a.negotiation_done
    print(agent_a.final)

    # await agent_c.start_negotiation(ts=[900, 1800], value=40)
    # await agent_c.negotiation_done


async def winzent_test(agent_a, agent_b, agent_c, agent_d):
    """
    In this case, the agents receive enough flexibility to solve the
    negotiation request.
    """
    # first, hand flexibility to agents
    agent_a.update_flexibility(t_start=900, min_p=0, max_p=0)  # agent0
    agent_b.update_flexibility(t_start=900, min_p=0, max_p=0)  # agent1
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=20)  # agent2
    agent_d.update_flexibility(t_start=900, min_p=0, max_p=10)  # agent3

    # 0 -> 1
    # 1 -> 2
    # 1 -> 3

    await agent_a.start_negotiation(ts=[900, 1800], value=30)
    await agent_a.negotiation_done
    print("agent0:", agent_a.final)


async def not_enough_flexibility(agent_a, agent_b, agent_c):
    """
    In this case, the agents do not receive enough flexibility to solve
    the problem and therefore a solution after timeout is taken.
    """
    agent_a.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=10)

    await agent_b.start_negotiation(ts=[0, 900], value=150)


async def simple_agents_setup():
    """
    Creates 3 simple agents. all living in one container.
    """
    # container addr
    addr = ("127.0.0.1", 5555)

    # multiple container are possible, here just one is taken
    container = await Container.factory(addr=addr)

    # create agents
    agent_a = WinzentAgent(container=container)
    agent_b = WinzentAgent(container=container)
    agent_c = WinzentAgent(container=container)
    agent_d = WinzentAgent(container=container)

    # add neighbors for agents
    agent_a.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_a.add_neighbor(aid=agent_c.aid, addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_a.aid, addr=addr)

    agent_b.add_neighbor(aid=agent_d.aid, addr=addr)
    agent_d.add_neighbor(aid=agent_b.aid, addr=addr)

    return agent_a, agent_b, agent_c, agent_d, container


async def four_agents_in_line_topology():
    """
    Creates 3 simple agents. all living in one container.
    """
    # container addr
    addr = ("127.0.0.1", 5555)

    # multiple container are possible, here just one is taken
    container = await Container.factory(addr=addr)

    # create agents
    agent_a = WinzentAgent(container=container)
    agent_b = WinzentAgent(container=container)
    agent_c = WinzentAgent(container=container)
    agent_d = WinzentAgent(container=container)

    #  a - b - c - d
    # add neighbors for agents
    agent_a.add_neighbor(aid=agent_b.aid, addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_c.aid, addr=addr)

    agent_c.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_d.aid, addr=addr)

    agent_d.add_neighbor(aid=agent_b.aid, addr=addr)

    return agent_a, agent_b, agent_c, agent_d, container


async def four_agents_another_topology():
    """
    Creates 3 simple agents. all living in one container.
    """
    # container addr
    addr = ("127.0.0.1", 5555)

    # multiple container are possible, here just one is taken
    container = await Container.factory(addr=addr)

    # create agents
    agent_a = WinzentAgent(container=container)
    agent_b = WinzentAgent(container=container)
    agent_c = WinzentAgent(container=container)
    agent_d = WinzentAgent(container=container)

    #  a - b - c
    #        - d
    # add neighbors for agents
    agent_a.add_neighbor(aid=agent_b.aid, addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_d.aid, addr=addr)

    agent_c.add_neighbor(aid=agent_b.aid, addr=addr)

    agent_d.add_neighbor(aid=agent_b.aid, addr=addr)

    return agent_a, agent_b, agent_c, agent_d, container


async def shutdown(agent_a, agent_b, agent_c, agent_d, container):
    """
    Shutdown all agents.
    """
    await agent_a.stop_agent()
    await agent_b.stop_agent()
    await agent_c.stop_agent()
    await agent_d.stop_agent()
    await container.shutdown()


loop = asyncio.get_event_loop()
loop.run_until_complete(run_negotiations())
