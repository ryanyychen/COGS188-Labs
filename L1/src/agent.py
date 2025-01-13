"""
Implement Agents and Environments. (Chapters 1-2)
"""

from typing import Tuple
import random

loc_A, loc_B = (0, 0), (1, 0)  # The two locations for the Vacuum world


class Agent:
    """
    This is the base class for all agents. It has a location and a performance score.
    """

    def __init__(self, location: Tuple):
        self.location = location
        self.performance = 0


class AgentMemory(Agent):
    """
    This agent has a memory of the locations it has visited and whether they were dirty or clean.
    """

    def __init__(self, location: Tuple):
        super().__init__(location)
        self.visited = {}


class TrivialVacuumEnvironment:
    """This environment has two locations, A and B. Each can be Dirty
    or Clean. The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self, agent: Agent | AgentMemory):
        self.status = {loc_A: random.choice(["Clean", "Dirty"]), loc_B: random.choice(["Clean", "Dirty"])}
        self.agent = agent
        self.action_space = ["Right", "Left", "Suck", "Stay"]

    def execute_action(self, agent: Agent, action):
        """
        Change the environment based on the agent's action.
        moving from left to right costs 1 point from the agent's performance
        sucking a dirty location costs 3 points from the agent's performance
        if the agent sucks a dirty location, it gets 10 points. if the agent sucks a clean location, it does not get point.
        if the agent stays, it does not get any points or lose any points.

        # DOCTEST
        >>> agent = Agent(loc_A)
        >>> env = TrivialVacuumEnvironment(agent)
        >>> env.status = {loc_A: 'Dirty', loc_B: 'Clean'}
        >>> env.execute_action(agent, 'Suck')
        >>> assert env.status == {loc_A: 'Clean', loc_B: 'Clean'}
        >>> assert agent.performance == 7
        >>> env.execute_action(agent, 'Right')
        >>> assert agent.location == loc_B
        >>> assert agent.performance == 6
        >>> env.execute_action(agent, 'Suck')
        >>> assert env.status == {loc_A: 'Clean', loc_B: 'Clean'}
        >>> assert agent.performance == 3
        """

        # Check for valid action
        assert action in self.action_space, "Invalid Action"

        # Add/Deduct points accordingly
        if (action == "Right" and agent.location == loc_A):
            agent.location = loc_B
            agent.performance -= 1
        elif (action == "Left" and agent.location == loc_B):
            agent.location = loc_A
            agent.performance -= 1
        elif (action == "Suck"):
            agent.performance -= 3
            if self.status[agent.location] == "Dirty":
                agent.performance += 10
            self.status[agent.location] = "Clean"

        # No points change for all other actions, including:
        # - Staying
        # - Moving left from A
        # - Moving right from B



    def random_agent(self, agent: Agent) -> str:
        """
        This agent function (policy) will randomly choose an action from the action space.

        Args:
            agent (Agent): the agent

        Returns:
            str: the action taken by the agent

        # DOCTEST
        >>> agent = Agent(loc_A)
        >>> env = TrivialVacuumEnvironment(agent)
        >>> action = env.random_agent(agent)
        >>> assert action in env.action_space
        """
        
        action_num = random.randint(0, 3)
        return self.action_space[action_num]

    def reflex_agent(self, agent: Agent) -> str:
        """
        This agent function (policy) will always suck if the location is dirty, and move if the location is clean.

        Args:
            agent (Agent): the agent

        Returns:
            str: the action taken by the agent

        # DOCTEST
        >>> agent = Agent(loc_A)
        >>> env = TrivialVacuumEnvironment(agent)
        >>> env.status = {loc_A: 'Dirty', loc_B: 'Clean'}
        >>> action = env.reflex_agent(agent)
        >>> assert action == 'Suck'
        >>> env.execute_action(agent, action)
        >>> assert env.status == {loc_A: 'Clean', loc_B: 'Clean'}
        >>> assert agent.performance == 7
        >>> action = env.reflex_agent(agent)
        >>> assert action == 'Right'
        >>> env.execute_action(agent, action)
        >>> assert agent.location == loc_B
        >>> assert agent.performance == 6
        >>> action = env.reflex_agent(agent)
        >>> assert action == 'Left'
        >>> env.execute_action(agent, action)
        >>> assert agent.location == loc_A
        """
        
        # Suck if dirty, otherwise move Left or Right accordingly
        if (self.status[agent.location] == "Dirty"):
            return "Suck"
        elif (agent.location == loc_A):
            return "Right"
        elif (agent.location == loc_B):
            return "Left"


    def model_based_agent(self, agent: AgentMemory) -> str:
        """
        This agent function (policy) will always suck if the location is dirty, and move if the location is clean.
        It will also keep track of the locations it has visited and whether they were dirty or clean.
        If the agent has visited both locations and they are clean, it will stay.
        There won't be any additional Dirty locations after the environment is initialized.

        - If the agent's current location is not in its visited memory, add it with its status (Clean/Dirty).
        - If the current location is Dirty, return 'Suck'.
        - If the current location is Clean and the agent is at location A, return 'Right'.
        - If the current location is Clean and the agent is at location B, return 'Left'.
        - If both locations A and B are Clean, return 'Stay'.

        Args:
            agent (Agent): the agent

        Returns:
            str: the action taken by the agent

        # DOCTEST
        >>> agent = AgentMemory(loc_A)
        >>> env = TrivialVacuumEnvironment(agent)
        >>> env.status = {loc_A: 'Dirty', loc_B: 'Clean'}
        >>> action = env.model_based_agent(agent)
        >>> assert action == 'Suck', "agent should suck the dirty location"
        >>> env.execute_action(agent, action)
        >>> assert env.status == {loc_A: 'Clean', loc_B: 'Clean'}, "location A should be clean"
        >>> action = env.model_based_agent(agent)
        >>> assert action == 'Right', f"agent should move to the right, your action: {action}"
        >>> env.execute_action(agent, action)
        >>> assert agent.location == loc_B, "agent should be at location B"
        >>> action = env.model_based_agent(agent)
        >>> assert action == 'Stay', f"agent should stay at B since both locations are clean, however your action is {action}"
        """
        
        '''
        - If the agent's current location is not in its visited memory, add it with its status (Clean/Dirty).
        - If the current location is Dirty, return 'Suck'.
        - If the current location is Clean and the agent is at location A, return 'Right'.
        - If the current location is Clean and the agent is at location B, return 'Left'.
        - If both locations A and B are Clean, return 'Stay'.
        '''

        # Append location and status to memory
        agent.visited[agent.location] = self.status[agent.location]
        
        # Suck if dirty
        if (agent.visited[agent.location] == "Dirty"):
            return "Suck"
        
        # Check if both locations visited
        if (len(agent.visited) == 2):
            if (agent.visited[loc_A] == "Clean" and agent.visited[loc_B] == "Clean"):
                return "Stay"
        
        if (agent.location == loc_A and agent.visited[loc_A] == "Clean"):
            return "Right"
        elif (agent.location == loc_B and agent.visited[loc_B] == "Clean"):
            return "Left"
        