import unittest
from src.graph import Node, BinaryTreeNode, Graph, Tree, BinarySearchTree

class TestGraphImplementation(unittest.TestCase):

    """Test graph implementation"""

    def setUp(self):
        self.n_list = [Node(i) for i in range(21)] # create a list of nodes for testing
        self.tree_n_list = [BinaryTreeNode(i) for i in range(21)] # create a list of binary tree nodes for testing

    def test_node(self):
        """Node: create node"""
        node = Node(1)
        self.assertEqual(node.value, 1)

    def test_node_add_directed_edge(self):
        """Node: add directed edge"""
        self.n_list[0].add_directed_edge(self.n_list[1])
        self.assertTrue(self.n_list[1] in self.n_list[0].children)
        self.assertTrue(self.n_list[0] in self.n_list[1].parents)

    def test_node_add_undirected_edge(self):
        """Node: add undirected edge"""
        self.n_list[0].add_undirected_edge(self.n_list[1])
        self.assertIn(self.n_list[1], self.n_list[0].parents)
        self.assertIn(self.n_list[1], self.n_list[0].children)
        self.assertIn(self.n_list[0], self.n_list[1].parents)
        self.assertIn(self.n_list[0], self.n_list[1].children)

    def test_graph(self):
        """Graph: create simple graph with two nodes"""
        graph = Graph()
        self.assertEqual(graph.nodes, set())
        graph.add_edge(self.n_list[0], self.n_list[1], directed=True)
        # test add directed edge
        self.assertEqual(graph.nodes, {self.n_list[0], self.n_list[1]})
        self.assertIn(self.n_list[1], self.n_list[0].children)
        self.assertIn(self.n_list[0], self.n_list[1].parents)
        # test add undirected edge
        graph.add_edge(self.n_list[1], self.n_list[2], directed=False)
        self.assertEqual(graph.nodes, {self.n_list[0], self.n_list[1], self.n_list[2]})
        self.assertIn(self.n_list[2], self.n_list[1].children)
        self.assertIn(self.n_list[1], self.n_list[2].parents)

    def test_tree(self):
        """Tree: create simple tree with two nodes"""
        tree = Tree(self.n_list[0])
        self.assertEqual(tree.nodes, set([self.n_list[0]]))
        # a simple tree with 4 nodes, 3 edges 0 ->1, 3, 1 -> 2
        tree.add_edge(self.n_list[0], self.n_list[1], directed=True)
        tree.add_edge(self.n_list[1], self.n_list[2], directed=True)
        tree.add_edge(self.n_list[0], self.n_list[3], directed=True)
        # the current tree is a valid tree
        self.assertTrue(tree.validate_tree())
        self.n_list[2].add_directed_edge(self.n_list[0])
        # the current tree has a cycle
        self.assertFalse(tree.validate_tree())

    def test_binary_search_tree_1(self):
        """BinarySearchTree.insert_node: create simple binary search tree with four nodes"""
        tree = BinarySearchTree()
        self.assertEqual(tree.nodes, set())
        tree.insert_node(self.tree_n_list[10])
        tree.insert_node(self.tree_n_list[5])
        tree.insert_node(self.tree_n_list[15])
        self.assertEqual(tree.root, self.tree_n_list[10])
        self.assertEqual(tree.root.left, self.tree_n_list[5])
        self.assertEqual(tree.root.right, self.tree_n_list[15])

    def test_binary_search_tree_2(self):
        """BinarySearchTree.validate_bst: create simple binary search tree with four nodes"""
        tree = BinarySearchTree()
        self.assertEqual(tree.nodes, set())
        tree.insert_node(self.tree_n_list[10])
        tree.insert_node(self.tree_n_list[15])
        tree.insert_node(self.tree_n_list[5])
        # the current tree is a valid binary search tree
        self.assertTrue(tree.validate_bst())
        tree.nodes.add(self.tree_n_list[20])
        self.tree_n_list[15].add_right_child(self.tree_n_list[20])
        # the current tree is a valid binary search tree
        self.assertTrue(tree.validate_bst())
