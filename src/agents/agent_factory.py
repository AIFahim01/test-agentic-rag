from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

def create_agent(tool_retriever, llm, system_prompt: str, verbose: bool = True):
    """
    Create a single agent given a retriever, an LLM, and a system prompt.
    """
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=tool_retriever,
        llm=llm,
        system_prompt=system_prompt,
        verbose=verbose
    )
    agent = AgentRunner(agent_worker)
    return agent

def create_agents(agent_configs):
    """
    Create multiple agents based on a list of configurations.
    Each configuration could be a dict like:
    {
        'name': <agent_name_string>,
        'retriever': <retriever_function>,
        'llm': <llm_object>,
        'system_prompt': <string>,
        'verbose': True/False (optional)
    }

    Returns a dict of agents keyed by the 'name' field.
    """
    agents = {}
    for config in agent_configs:
        name = config.get('name', 'unnamed_agent')
        agent = create_agent(
            tool_retriever=config['retriever'],
            llm=config['llm'],
            system_prompt=config['system_prompt'],
            verbose=config.get('verbose', True)
        )
        agents[name] = agent
    return agents
