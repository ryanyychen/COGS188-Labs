# L2 COGS 188 Winter 2025
## Due Date: Tuesday Jan 21st, 2025 11:59 PM PST 

In this lab, you will implement the following files: `graph.py`, `search.py`, and `benchmark.py`. The objectives of this lab are to gain hands-on experience with basic data structures and to compare the efficiency of search algorithms through empirical analysis.

### `graph.py`

You will implement the foundational components of a graph data structure, which include:
- **Node:** Represents a single element in the graph, storing a value and possibly connections to other nodes.
- **Graph:** You will define methods for adding nodes, edges, and traversing the graph.
- **Tree** and **Binary Tree**: A tree is a special type of graph with no cycles. You will implement a binary tree, where each node has at most two children (left and right).

### `search.py`

This file will focus on implementing search algorithms within the context of binary search trees (BST) and basic lists:
- **Depth-First Search (DFS)**: DFS is a graph traversal algorithm that explores as far along a branch as possible before backtracking. It can be implemented using a stack (explicitly or via recursion). DFS is commonly used for pathfinding, cycle detection, and connectivity checks in graphs. Its time complexity is $O(V + E)$, where $V$ is the number of vertices and $E$ is the number of edges.

- **Breadth-First Search (BFS)**: BFS is a graph traversal algorithm that explores all neighbors at the current depth level before moving on to nodes at the next depth level. BFS is typically implemented using a queue and is useful for finding the shortest path in unweighted graphs. Its time complexity is also $O(V + E)$.

- **Linear Search:** A simple search algorithm that iterates through a list or array element by element, checking each value against the target. Linear search is straightforward but can be inefficient for large datasets, with a time complexity of $O(n)$.

- **Binary Search:** A more efficient search algorithm that works on sorted data. It repeatedly divides the search interval in half, reducing the problem size logarithmically. Binary search has a time complexity of $O(\log n)$. To implement binary search, you will first construct a binary search tree (BST) as part of this lab.


### Submission Guidelines

Download, and submit the `src/` folder to Gradescope, you should be able to see the autograder result within 2 minutes.
