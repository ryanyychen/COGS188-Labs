import unittest
from src.graph import Node, Graph, Tree, BinarySearchTree
from src.search import bfs, dfs

class TestSearchImplementation(unittest.TestCase):
    
    """Test graph implementation"""
    
    def setUp(self):
        self.node_list = [Node(i) for i in range(20)]
        self.graph = Graph(self.node_list[0])
        self.graph.add_edge(self.node_list[0], self.node_list[1])
        self.graph.add_edge(self.node_list[0], self.node_list[2])
        self.graph.add_edge(self.node_list[1], self.node_list[3])
        self.graph.add_edge(self.node_list[1], self.node_list[4])
        self.graph.add_edge(self.node_list[2], self.node_list[5])
        self.graph.add_edge(self.node_list[2], self.node_list[6])
        self.graph.add_edge(self.node_list[2], self.node_list[7])
    
    def test_search_bfs(self):
        visited, found = bfs(self.graph, self.node_list[7])
        self.assertEqual(visited, [self.node_list[0], self.node_list[1], self.node_list[2], self.node_list[3], self.node_list[4], self.node_list[5], self.node_list[6], self.node_list[7]])
        self.assertTrue(found)
        
    def test_search_dfs(self):
        visited, found = dfs(self.graph, self.node_list[7])
        sequence = [0,1,3,4,2,5,6,7]
        self.assertEqual(visited, [self.node_list[i] for i in sequence])
        self.assertTrue(found)