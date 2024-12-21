# Graph Analysis using Multi-Agent Collaboration with LLMs 
Reimplemented the research paper [GraphTeam: Facilitating Large Language Model-based Graph Analysis via Multi-Agent Collaboration](https://arxiv.org/pdf/2410.18032v2) to develop multi-agent systems optimized for understanding data within graph representations. 

## Example Output
```json
[
    {
        "role": "system",
        "content": "You are a Graph Learning Specialist. Your role is to solve graph-related tasks using Python libraries like NetworkX. You will analyze the input, generate Python code, and ensure the solution is efficient and correct."
    },
    {
        "role": "user",
        "content": "Determine if there is a path between node 7 and node 23 in the following graph: [(0, 26), (0, 15), (1, 26), (1, 15), (1, 25), (7, 17), (6, 19)]."
    },
    {
        "role": "assistant",
        "content": "```python\nimport networkx as nx\n\n# Define edges\nedges = [(0, 26), (0, 15), (1, 26), (1, 15), (1, 25), (7, 17), (6, 19)]\n\n# Create an undirected graph\nG = nx.Graph()\nG.add_edges_from(edges)\n\n# Check if there is a path between node 7 and node 23\npath_exists = nx.has_path(G, 7, 23)\n\n# Output the result\nprint(f'There is a path: {path_exists}')\n```"
    }
]
```
