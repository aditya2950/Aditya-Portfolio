#1 DFS Algorithm

# Define the graph as a dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Define the visited list and the queue (acting as a stack for DFS)
visited = [] 
queue = []

# Define the DFS function
def dfs(visited, graph, node):
    # Mark the current node as visited and add it to the queue
    visited.append(node)
    queue.append(node)

    # Loop while there are nodes in the queue
    while queue:
        # Pop the last node from the queue (LIFO for DFS)
        s = queue.pop()
        print(s, end=" ")

        # Visit all the neighbours of the current node
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

# Call the DFS function starting from node 'A'
dfs(visited, graph, 'A')



#2 DFS 

# Define the graph as a dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Define the visited set to keep track of visited nodes
visited = set()

# Define the DFS function
def dfs(visited, graph, node):
    if node not in visited:
        # Print the current node
        print(node)
        # Mark the node as visited
        visited.add(node)
        # Visit all the neighbours of the current node
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Call the DFS function starting from node 'A'
dfs(visited, graph, 'A')








# 3 tic tac toe


# Tic-Tac-Toe Program using
# random number in Python
# importing all necessary libraries
import numpy as np
import random
from time import sleep

# Creates an empty board
def create_board():
    return np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])

# Check for empty places on board
def possibilities(board):
    l = []
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
                l.append((i, j))
    return l

# Select a random place for the player
def random_place(board, player):
    selection = possibilities(board)
    current_loc = random.choice(selection)
    board[current_loc] = player
    return board

# Checks whether the player has three
# of their marks in a horizontal row
def row_win(board, player):
    for x in range(len(board)):
        win = True
        for y in range(len(board)):
            if board[x, y] != player:
                win = False
                continue
        if win == True:
            return win
    return win

# Checks whether the player has three
# of their marks in a vertical row
def col_win(board, player):
    for x in range(len(board)):
        win = True
        for y in range(len(board)):
            if board[y][x] != player:
                win = False
                continue
        if win == True:
            return win
    return win

# Checks whether the player has three
# of their marks in a diagonal row
def diag_win(board, player):
    win = True
    for x in range(len(board)):
        if board[x, x] != player:
            win = False
    if win:
        return win
    win = True
    for x in range(len(board)):
        if board[x, len(board) - 1 - x] != player:
            win = False
    return win

# Evaluates whether there is
# a winner or a tie
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if (row_win(board, player) or
            col_win(board, player) or
            diag_win(board, player)):
            winner = player
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

# Main function to start the game
def play_game():
    board, winner, counter = create_board(), 0, 1
    print(board)
    sleep(2)
    while winner == 0:
        for player in [1, 2]:
            board = random_place(board, player)
            print("Board after " + str(counter) + " move")
            print(board)
            sleep(2)
            counter += 1
            winner = evaluate(board)
            if winner != 0:
                break
    return winner

# Driver Code
print("Winner is: " + str(play_game()))










# 4th 8 puzzle

class Solution:
    def solve(self, board):
        state_dict = {}
        flatten = []
        for i in range(len(board)):
            flatten += board[i]
        flatten = tuple(flatten)
        
        state_dict[flatten] = 0
        if flatten == (0, 1, 2, 3, 4, 5, 6, 7, 8):
            return 0
        return self.get_paths(state_dict)

    def get_paths(self, state_dict):
        cnt = 0
        while True:
            current_nodes = [x for x in state_dict if state_dict[x] == cnt]
            if len(current_nodes) == 0:
                return -1
            
            for node in current_nodes:
                next_moves = self.find_next(node)
                for move in next_moves:
                    if move not in state_dict:
                        state_dict[move] = cnt + 1
                    if move == (0, 1, 2, 3, 4, 5, 6, 7, 8):
                        return cnt + 1
            cnt += 1

    def find_next(self, node):
        moves = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        results = []
        
        pos_0 = node.index(0)
        for move in moves[pos_0]:
            new_node = list(node)
            new_node[move], new_node[pos_0] = new_node[pos_0], new_node[move]
            results.append(tuple(new_node))
        return results

# Driver code
ob = Solution()
matrix = [
    [3, 1, 2],
    [4, 7, 5],
    [6, 8, 0]
]
print(ob.solve(matrix))










#5th water jug problem



from collections import defaultdict

jug1, jug2, aim = 4, 3, 2
visited = defaultdict(lambda: False)

def waterJugSolver(amt1, amt2):
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):
        print(amt1, amt2)
        return True
    
    if visited[(amt1, amt2)] == False:
        print(amt1, amt2)
        visited[(amt1, amt2)] = True
        
        # Try all possible moves recursively
        return (waterJugSolver(0, amt2) or  # Empty jug1
                waterJugSolver(amt1, 0) or  # Empty jug2
                waterJugSolver(jug1, amt2) or  # Fill jug1
                waterJugSolver(amt1, jug2) or  # Fill jug2
                waterJugSolver(amt1 + min(amt2, (jug1 - amt1)), amt2 - min(amt2, (jug1 - amt1))) or  # Pour water from jug2 to jug1
                waterJugSolver(amt1 - min(amt1, (jug2 - amt2)), amt2 + min(amt1, (jug2 - amt2))))  # Pour water from jug1 to jug2
    else:
        return False

print("Steps:")
waterJugSolver(0, 0)










#Traveling salesman problem



v = 4
answer = []

def tsp(graph, visited, currPos, n, count, cost):
    if count == n and graph[currPos][0]:
        answer.append(cost + graph[currPos][0])
        return
    
    for i in range(n):
        if not visited[i] and graph[currPos][i]:
            visited[i] = True
            tsp(graph, visited, i, n, count + 1, cost + graph[currPos][i])
            visited[i] = False

if __name__ == '__main__':
    n = 4
    graph = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    visited = [False for _ in range(n)]
    visited[0] = True
    tsp(graph, visited, 0, n, 1, 0)
    
    print(min(answer))












# 7 tower of hanoi

def TowerOfHanoi(n, source, destination, auxiliary):
    if n == 1:
        print("Move disk 1 from source", source, "to destination", destination)
        return
    TowerOfHanoi(n-1, source, auxiliary, destination)
    print("Move disk", n, "from source", source, "to destination", destination)
    TowerOfHanoi(n-1, auxiliary, destination, source)

n = 4
TowerOfHanoi(n, 'A', 'C', 'B')










#8 Monkey Banana problem





i = 0

def Monkey_go_box(x, y):
    global i
    i = i + 1
    print('step:', i, ', monkey slave', x, ', Go to', y)

def Monkey_move_box(x, y):
    global i
    i = i + 1
    print('step:', i, ', monkey take the box from', x, ', deliver to', y)

def Monkey_on_box():
    global i
    i = i + 1
    print('step:', i, ', Monkey climbs up the box')

def Monkey_get_banana():
    global i
    i = i + 1
    print('step:', i, ', Monkey picked a banana')

import sys

# Read the input from standard input
codeIn = sys.stdin.read()
codeInList = codeIn.split()
monkey = codeInList[0]
banana = codeInList[1]
box = codeInList[2]

print('The steps are as follows:')

Monkey_go_box(monkey, box)
Monkey_move_box(box, banana)
Monkey_on_box()
Monkey_get_banana()















#9 th alpha beta pruning


MAX, MIN = 1000, -1000

def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]

    if maximizingPlayer:
        best = MIN

        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)

            if beta <= alpha:
                break
        return best

    else:
        best = MAX

        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)

            if beta <= alpha:
                break
        return best

# Example usage:
values = [3, 5, 6, 9, 1, 2, 0, -1]
print("The optimal value is:", minimax(0, 0, True, values, MIN, MAX))















#10th N queens problem



N = 4

def printSolution(board):
    for i in range(N):
        for j in range(N):
            print(board[i][j], end=" ")
        print()

def isSafe(board, row, col):
    # Check this row on left side
    for i in range(col):
        if board[row][i] == 1:
            return False
    
    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    
    # Check lower diagonal on left side
    for i, j in zip(range(row, N, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    
    return True

def solveNQUtil(board, col):
    # base case: If all queens are placed
    # then return true
    if col >= N:
        return True
    
    for i in range(N):
        if isSafe(board, i, col):
            # Place this queen in board[i][col]
            board[i][col] = 1
            # recur to place rest of the queens
            if solveNQUtil(board, col + 1):
                return True
            # If placing queen in board[i][col] doesn't lead to a solution
            # then remove queen from board[i][col]
            board[i][col] = 0
    
    return False

def solveNQ():
    board = [[0]*N for _ in range(N)]
    
    if not solveNQUtil(board, 0):
        print("Solution does not exist")
        return False
    
    printSolution(board)
    return True

# Driver code
solveNQ()
