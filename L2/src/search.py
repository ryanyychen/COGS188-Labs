from src.graph import Node, Graph, Tree, BinarySearchTree
from typing import List, Tuple


def bfs(g: Graph, target: Node | None = None) -> Tuple[List[Node], bool]:
    """
    Perform a breadth-first search on the graph starting from the root node.

    Args:
        Graph (Graph): the graph to search on
        target (Node): the target node we are searching for

    Returns:
        List[Node]: a list of nodes in the order they were visited
        bool: whether the target node was found
    """

    visited = []
    queue = [g.root]
    found = False
    while queue:
        # NOTE: in the release, we give instruction to student to implement the following block
        # line be line
        # BEGIN SOLUTION
        node = queue.pop(0)
        visited.append(node)
        if node == target:
            found = True
            break
        for neighbor in sorted(node.children, key=lambda x: x.value):
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
        # END SOLUTION
    return visited, found


def dfs(g: Graph, target: Node | None = None) -> Tuple[List[Node], bool]:
    """
    Perform a depth-first search on the graph starting from the root node.

    Args:
        Graph (Graph): the graph to search on
        target (Node): the target node we are searching for

    Returns:
        List[Node]: a list of nodes in the order they were visited
        bool: whether the target node was found
    """

    # BEGIN SOLUTION
    visited = []
    stack = [g.root]
    found = False
    while stack:
        node = stack.pop()
        visited.append(node)
        if node == target:
            found = True
        for neighbor in node.children[::-1]:  # reverse to maintain order
            if neighbor not in visited and neighbor not in stack:
                stack.append(neighbor)
    return visited, found
    # END SOLUTION
