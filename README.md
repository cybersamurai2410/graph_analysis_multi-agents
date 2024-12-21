# Graph Analysis using Multi-Agent Collaboration with LLMs 
Reimplemention of the research paper [GraphTeam: Facilitating Large Language Model-based Graph Analysis via Multi-Agent Collaboration](https://arxiv.org/pdf/2410.18032v2) to develop multi-agent systems optimized for understanding data within graph representations using OpenAI GPT LLM. The research achieved state-of-the-art performance with an average 25.85% improvement over the best baseline in terms of accuracy.  

**Multi-Agent System:**
1. Input-Output Normalization Module
    - Question Agent - Extracts and refines four key arguments (e.g., graph type and output format) from the original question to facilitate problem understanding.
    - Answer Agent - Organizes the results to meet the output requirements.
3. External Knowledge Retrieval Module
    - Search Agent - Retrieves the most relevant entries from the knowledge base for each question. The knowledge base consists of relevant documentation and experience information.
5. Problem-Solving Module
    - Coding Agent - Uses established algorithms via programming to generate solutions given the retrieved information from the Search Agent.
    - Reasoning Agent - Directly computes the results without programming if the Coding Agent fails.
  
## Architecture
<img width="605" alt="image" src="https://github.com/user-attachments/assets/fc01928c-1566-4738-9b0f-237a5d422dd6" />

## Benchmarks 
<img width="603" alt="image" src="https://github.com/user-attachments/assets/a6fc982b-dda8-425f-a986-3193622be0f4" />

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
```python
import networkx as nx

# Define edges
edges = [(0, 26), (0, 15), (1, 26), (1, 15), (1, 25), (7, 17), (6, 19)]

# Create an undirected graph
G = nx.Graph()
G.add_edges_from(edges)

# Check if there is a path between node 7 and node 23
path_exists = nx.has_path(G, 7, 23)

# Output the result
print(f'There is a path: {path_exists}')
```
```bash
There is a path: False
```
