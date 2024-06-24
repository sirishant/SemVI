# Implement DFID Algorithm

#                          A
#                         / \
#                        B   C
#                       / \ / \
#                      D  E F  G

from collections import defaultdict

graph = defaultdict(list)

def addEdge(u, v):
    graph[u].append(v)

def dfid(start, goal, depth):
    print(start, end=" -> ")
    if start == goal:
        return True
    if depth <= 0:
        return False
    for i in graph[start]:
        if dfid(i, goal, depth-1):
            return True
    return False

def IDDFS(start, goal, maxDepth):
    for i in range(maxDepth):
        print("\nDFID at depth: ", i+1)
        print("Path: ", end="")
        isPathFound = dfid(start, goal, i)

    if isPathFound:
        print("\nGoal Node Found!")
        return
    else:
        print("\nGoal Node Not Found!")

addEdge('A', 'B')
addEdge('A', 'C')
addEdge('B', 'D')
addEdge('B', 'E')
addEdge('C', 'F')
addEdge('C', 'G')

print("Graph: ", graph)

IDDFS('A', 'G', 3)