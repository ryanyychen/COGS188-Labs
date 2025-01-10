import unittest
from src.agent import Agent, AgentMemory, TrivialVacuumEnvironment, loc_A, loc_B


class TestSimpleAgent(unittest.TestCase):
    
    """Test simple agent implementation"""
    
    def setUp(self):
        self.agent = Agent(loc_A)
        self.env = TrivialVacuumEnvironment(self.agent)
        self.agent_memory = AgentMemory(loc_A)
        
        
    def test_execute_action_move_right(self):
        """execute_action: move right"""
        self.env.execute_action(self.agent, "Right")
        self.assertEqual(self.agent.location, loc_B)
    
    def test_execute_action_move_left(self):
        """execute_action: move left"""
        self.agent.location = loc_B
        self.env.execute_action(self.agent, "Left")
        self.assertEqual(self.agent.location, loc_A)
    
    def test_execute_action_suck(self):
        """execute_action: suck"""
        self.env.status = {loc_A: "Dirty", loc_B: "Clean"}
        self.env.execute_action(self.agent, "Suck")
        self.assertEqual(self.env.status, {loc_A: "Clean", loc_B: "Clean"})
    
    def test_execute_action_performance(self):
        """execute_action: performance movement"""
        self.env.execute_action(self.agent, "Right")
        self.assertEqual(self.agent.performance, -1)
        self.env.execute_action(self.agent, "Left")
        self.assertEqual(self.agent.performance, -2)
        self.env.execute_action(self.agent, "Suck")
        self.assertEqual(self.agent.performance, 5)

    def test_random_agent(self):
        """random_agent: movements in action space"""
        action = self.env.random_agent(self.agent)
        self.assertIn(action, self.env.action_space)

    def test_reflex_agent_clean(self):
        """reflex_agent: clean location"""
        self.env.status = {loc_A: "Clean", loc_B: "Clean"}
        action = self.env.reflex_agent(self.agent)
        self.assertIn(action, ["Right", "Left"])
    
    def test_reflex_agent_dirty(self):
        """reflex_agent: dirty location"""
        self.env.status = {loc_A: "Dirty", loc_B: "Clean"}
        action = self.env.reflex_agent(self.agent)
        self.assertEqual(action, "Suck")
        
    def test_reflex_agent_clean_then_dirty(self):
        """reflex_agent: from clean location to dirty location"""
        self.env.status = {loc_A: "Clean", loc_B: "Dirty"}
        action = self.env.reflex_agent(self.agent)
        self.assertEqual(action, "Right")
        self.env.execute_action(self.agent, action)
        action = self.env.reflex_agent(self.agent)
        self.assertEqual(action, "Suck")
    
    def test_model_based_agent_clean(self):
        """model_based_agent: two clean locations"""
        self.env.status = {loc_A: "Clean", loc_B: "Clean"}
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Right")
        self.env.execute_action(self.agent_memory, action)
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Stay")
        
    def test_model_based_agent_dirty(self):
        """model_based_agent: two dirty locations"""
        self.env.status = {loc_A: "Dirty", loc_B: "Dirty"}
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Suck")
        self.env.execute_action(self.agent_memory, action)
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Right")
        self.env.execute_action(self.agent_memory, action)
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Suck")
        self.env.execute_action(self.agent_memory, action)
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Stay")
        
    def test_model_based_agent_dirty_then_clean(self):
        """model_based_agent: from dirty location to clean location"""
        self.env.status = {loc_A: "Dirty", loc_B: "Clean"}
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Suck")
        self.env.execute_action(self.agent_memory, action)
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Right")
        self.env.execute_action(self.agent_memory, action)
        action = self.env.model_based_agent(self.agent_memory)
        self.assertEqual(action, "Stay")
        