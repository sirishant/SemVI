#                   S
#                  / \
#                 A   B
#                / \ / \
#               C  D E  F
#                   /  / \
#                  H  I   G
# Implement Heuristic Search Algorithm

tree = {
    'S': [[('A', 7), ('B', 2)], 13],
    'A': [[('C', 4), ('D', 6)], 12],
    'B': [[('E', 5), ('F', 3)], 4],
    'E': [[('H', 2)], 8],
    'H': [[], 4],
    'F': [[('I', 4), ('G', 1)], 2],
    'G': [[], 0],
    'I': [[], 9],
    'C': [[], 7],
    'D': [[], 3],
}

start = 'S'
goal = 'A'

print("Start:", start, "\tEnd:", goal, "\n")

def get_children(node):
    return tree[node][0]

def sort_queue(queue):
    # x is a tuple (node, path_cost, heuristic)
    # x[2] is the heuristic value of the node
    return sorted(queue, key=lambda x: x[2])  # Sort the queue based on heuristic value

def bfs(start):
    queue = [(start, 0, tree[start][1])]  # (node, path_cost, heuristic)
    path = []

    while queue:
        queue = sort_queue(queue)
        node, path_cost, _ = queue.pop(0)

        path.append(node)
        print("Current Path:", path)

        if node == goal:
            return True, path, path_cost

        children = get_children(node)
        for child, cost in children:
            queue.append((child, path_cost + cost, tree[child][1]))

    return False, [], 0


success, path, final_cost = bfs(start)
if success:
    print("Goal reached. Path:", path)
    print("Final Cost of the Path:", final_cost)
else:
    print("Goal not reachable.")