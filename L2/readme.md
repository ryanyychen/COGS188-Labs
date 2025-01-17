# L2 COGS 188 Winter 2025
## Due Date: Tuesday Jan 21st, 2025 11:59 PM PST 

In this lab, you will implement the following files: `graph.py` and `search.py`. The objectives of this lab are to gain hands-on experience with basic data structures and to compare the efficiency of search algorithms through empirical analysis.

### `graph.py`

You will implement the foundational components of a graph data structure, which include:
- **Node:** Represents a single element in the graph, storing a value and possibly connections to other nodes.
- **Graph:** You will define methods for adding nodes, edges, and traversing the graph.
- **Tree** and **BinarySearchTree**: A tree is a special type of graph with no cycles. You will implement a binary tree, where each node has at most two children (left and right).

### `search.py`

This file will focus on implementing search algorithms within the context of binary search trees (BST) and basic lists:
- **Depth-First Search (DFS)**: DFS is a graph traversal algorithm that explores as far along a branch as possible before backtracking. It can be implemented using a stack (explicitly or via recursion). DFS is commonly used for pathfinding, cycle detection, and connectivity checks in graphs. Its time complexity is $O(V + E)$, where $V$ is the number of vertices and $E$ is the number of edges.

- **Breadth-First Search (BFS)**: BFS is a graph traversal algorithm that explores all neighbors at the current depth level before moving on to nodes at the next depth level. BFS is typically implemented using a queue and is useful for finding the shortest path in unweighted graphs. Its time complexity is also $O(V + E)$.


### How to Run Unit Test

After you finish, you can run the unit test to make sure that your implementation is correct. You can run the unit test via the following command on the `/L2` directory

```py
python -m unittest discover -s tests
```

If you see

```bash
(base) <your-username> L2 % python -m unittest discover -s tests
...........
----------------------------------------------------------------------
Ran 11 tests in 0.000s
```

this indicates that you have passed all the tests.


### Submission Guidelines

Download, and submit the `graph.py` and `search.py` to Gradescope, you should be able to see the autograder result within 2 minutes.