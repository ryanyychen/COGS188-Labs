from typing import List, Set


class Node:
    """
    A class to represent a node in a graph. Each node has a value,
    a list of parents, and a list of children. In this lab, you can assume
    that each individual node has unique value.
    """

    def __init__(self, value: int):
        self.value = value
        self.parents: List[Node] = []
        self.children: List[Node] = []

    def add_directed_edge(self, end: "Node"):
        """
        Adds a directed edge between self (start) and end node:
        - add end to self's children list
        - add self to the end's parent list

        Args:
            end (Node): the end node
        """
        # TODO your code here
        ...

    def add_undirected_edge(self, end: "Node"):
        """
        add two directed edges between self and end node

        self -> end and end -> self

        Args:
            end (Node): the end node
        """
        # TODO your code here
        ...

    def __repr__(self):
        """String representation of the node"""
        return f"Node({self.value})"

    def __lt__(self, other):
        """Less than comparison of two nodes based on their values"""
        return self.value < other.value


class Graph:
    """
    A class to represent a graph. A graph is a collection of nodes and edges.
    This class should have a list of nodes in the graph and a start node.
    """

    def __init__(self, root: Node = None):
        """init function for the graph

        Args:
            node (List[Node], optional): the initial node for the graph. Defaults to None.
        """
        self.nodes: Set[Node] = set()
        if root is not None:
            self.nodes.add(root)
            self.root = root
        else:
            self.root = None

    def add_node(self, node: Node):
        """
        Add a node to the graph

        Args:
            node (Node): the node to add
        """
        self.nodes.add(node)

    def add_edge(self, start: Node, end: Node, directed=True):
        """
        Add an edge between two nodes. Before adding edge, you need to make sure
        that both nodes are already in the graph. If not, add them to the graph.

        Args:
            start (Node): the start node
            end (Node): the end node
            directed (bool): whether the edge is directed or not, default is True
        """
        # TODO your code here
        ...


class Tree(Graph):
    """
    Tree class that inherits from Graph class and has additional properties and methods for trees.
    """

    def __init__(self, root: Node = None):
        """init function for the tree

        Args:
            root (Node, optional): the root node of the tree. Defaults to None.
        """
        super().__init__(root)
        self.root = root

    def validate_tree(self) -> bool:
        """
        Validates if the current tree is a valid tree. Use the following rules:
        - A tree must have exactly one root node
        - A tree must have no cycles

        Returns: (bool) whether the tree is valid or not
        """
        # TODO your code here
        ...


class BinaryTreeNode(Node):
    """Node that has at most two children, left and right"""

    def __init__(self, value: int):
        super().__init__(value)
        self.left: BinaryTreeNode | None = None
        self.right: BinaryTreeNode | None = None

    def add_left_child(self, node: "BinaryTreeNode"):
        """Add a left child to the current node"""
        self.add_directed_edge(node)
        self.left = node

    def add_right_child(self, node: "BinaryTreeNode"):
        """Add a right child to the current node"""
        self.add_directed_edge(node)
        self.right = node


class BinarySearchTree(Tree):

    def __init__(self, root: BinaryTreeNode = None):
        super().__init__(root)
        self.root = root

    def insert_node(self, node: BinaryTreeNode):
        """
        Insert a node into the binary search tree. The node should be inserted based on the value of the node.
        If the value of the node is less than the current node, it should be inserted in the left subtree.
        If the value of the node is greater than the current node, it should be inserted in the right subtree.

        Args:
            node (Node): the node to insert
        """
        if not self.root:
            self.root = node
            self.nodes.add(node)
            return
        current = self.root
        # TODO your code here
        ...

    def validate_bst(self) -> bool:
        """
        Validates if the current tree is a valid binary search tree. Use the following rules:
        - A binary search tree must have exactly one root node
        - A binary search tree must have no cycles
        - A binary search tree must have at most two children for each node
        - A binary search tree must have the property that for each node, all nodes in its left subtree have smaller values and all nodes in its right subtree have larger values.
        Specifically:
            - The left child of a node must be less than the parent node
            - The right child of a node must be greater than the parent node
        """
        # TODO your code here
        ...
