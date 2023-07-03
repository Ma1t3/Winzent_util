import asyncio

from mango.core.container import Container
from pgasc.agents.winzent_agent_system.winzent.winzent_agent import (
    WinzentAgent,
)

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
        agent_e,
        container,
    ) = await simple_agents_setup()

    print("\n Start new negotiation with enough flexibility. ")
    await unsuccessful_negotiation(agent_a, agent_b, agent_c, agent_d, agent_e)

    await shutdown(agent_a, agent_b, agent_c, agent_d, agent_e, container)


async def successful_negotiation(agent_a, agent_b, agent_c, agent_d, agent_e):
    """
    In this case, the agents receive enough flexibility to solve the
    negotiation request.
    """
    # first, hand flexibility to agents
    agent_a.update_flexibility(t_start=900, min_p=0, max_p=0)  # 0
    agent_b.update_flexibility(t_start=900, min_p=0, max_p=0)  # 1
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=1)  # 2
    agent_d.update_flexibility(t_start=900, min_p=0, max_p=5)  # 3
    agent_e.update_flexibility(t_start=900, min_p=0, max_p=9)  # 4

    await agent_a.start_negotiation(ts=[900, 1800], value=10)  # 0
    await agent_a.negotiation_done
    # sgens updaten (caution: external grids need different update)
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=0)  # 2
    agent_e.update_flexibility(t_start=900, min_p=0, max_p=0)  # 4
    print("NEXT")
    await agent_b.start_negotiation(ts=[900, 1800], value=5)  # 1
    await agent_b.negotiation_done

    # after all negotiation: if a negotiation has not enough flexibility -> complete fail


async def unsuccessful_negotiation(
    agent_a, agent_b, agent_c, agent_d, agent_e
):
    """
    In this case, the agents receive enough flexibility to solve the
    negotiation request.
    """
    # first, hand flexibility to agents
    agent_a.update_flexibility(t_start=900, min_p=0, max_p=0)  # 0
    agent_b.update_flexibility(t_start=900, min_p=0, max_p=0)  # 1
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=0)  # 2
    agent_d.update_flexibility(t_start=900, min_p=0, max_p=9)  # 3
    agent_e.update_flexibility(t_start=900, min_p=0, max_p=2)  # 4

    await agent_a.start_negotiation(ts=[900, 1800], value=6)  # 0
    await agent_a.negotiation_done
    print("a done")
    # sgens updaten (caution: external grids need different update)
    agent_d.update_flexibility(t_start=900, min_p=0, max_p=5)  # 3
    agent_e.update_flexibility(t_start=900, min_p=0, max_p=0)  # 4
    print("START B")
    await agent_b.start_negotiation(ts=[900, 1800], value=4)  # 1
    await agent_b.negotiation_done
    print("b done")

    agent_d.update_flexibility(t_start=900, min_p=0, max_p=1)  # 3
    print("START C")
    await agent_c.start_negotiation(ts=[900, 1800], value=1)  # 2
    await agent_c.negotiation_done

    # after all negotiation: if a negotiation has not enough flexibility -> complete fail


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
    agent_e = WinzentAgent(container=container)

    # add neighbors for agents
    agent_a.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_a.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_c.aid, addr=addr)

    agent_c.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_d.aid, addr=addr)

    agent_d.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_d.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_e.add_neighbor(aid=agent_d.aid, addr=addr)
    agent_e.add_neighbor(aid=agent_a.aid, addr=addr)

    # agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    # agent_c.add_neighbor(aid=agent_a.aid, addr=addr)

    return agent_a, agent_b, agent_c, agent_d, agent_e, container


async def shutdown(agent_a, agent_b, agent_c, agent_d, agent_e, container):
    """
    Shutdown all agents.
    """
    await agent_a.stop_agent()
    await agent_b.stop_agent()
    await agent_c.stop_agent()
    await agent_d.stop_agent()
    await agent_e.stop_agent()
    await container.shutdown()


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_negotiations())


if __name__ == "__main__":
    main()
