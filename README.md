# tsp
Code for solving and visualizing the Travelling Salesman Problem.

The code is inspired by the article series:
[An intelligent decision support system for tourism in Python] by Carlos Jimenez Uribe.

[An intelligent decision support system for tourism in Python]: https://medium.com/@carlosjuribe/list/an-intelligent-decision-support-system-for-tourism-in-python-b6ba165b4236

## Mathematical model

The TSP is formulated as a Mixed Integer Linear Programming (MILP) problem using the Miller-Tucker-Zemlin (MTZ) formulation for subtour elimination.

### Sets
- $S$: Set of all nodes to be visited
- $S^* = S \setminus \{\text{startend}\}$: Nodes of interest (all nodes except the start/end depot)
- $A = \{(i,j) : i,j \in S, i \neq j\}$: Set of valid arcs connecting different nodes

### Parameters
- $C_{ij}$: Cost of traveling from node $i$ to node $j$, for all $(i,j) \in A$
- $M = |S^*|$: Big-M constant for MTZ subtour elimination

### Decision Variables
- $x_{ij} \in \{0,1\}$: Binary variable indicating whether to travel from node $i$ to node $j$, for all $(i,j) \in A$
- $r_i \in \mathbb{R}_+$: Rank (visit order) of node $i$, for all $i \in S^*$, with bounds $1 \leq r_i \leq |S^*|$

### Objective Function
$$\min \sum_{(i,j) \in A} C_{ij} \cdot x_{ij}$$

### Constraints

**Degree Constraints:**
- Each node must be entered exactly once:
$$\sum_{i \in S : i \neq j} x_{ij} = 1, \quad \forall j \in S$$

- Each node must be exited exactly once:
$$\sum_{j \in S : j \neq i} x_{ij} = 1, \quad \forall i \in S$$

**Miller-Tucker-Zemlin (MTZ) Subtour Elimination:**
$$r_j \geq r_i + 1 - M \cdot (1 - x_{ij}), \quad \forall i,j \in S^*, i \neq j$$

This constraint ensures that if node $j$ is visited from node $i$ (i.e., $x_{ij} = 1$), then the rank of $j$ must be strictly greater than the rank of $i$, preventing subtours among the nodes in $S^*$.
