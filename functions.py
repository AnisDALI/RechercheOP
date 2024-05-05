import numpy as np
from collections import deque

def read_data(file_path):
    """Lire les données du fichier .txt et retourner les matrices de coûts, les provisions et les demandes."""
    try:
        with open(file_path, 'r') as file:
            n, m = map(int, file.readline().strip().split())
            cost_matrix = []
            supply = []
            for _ in range(n):
                row = list(map(int, file.readline().strip().split()))
                cost_matrix.append(row[:-1])
                supply.append(row[-1])
            demand = list(map(int, file.readline().strip().split()))
        return cost_matrix, supply, demand
    except FileNotFoundError:
        print(f"Le fichier {file_path} n'a pas été trouvé.")
        return [], [], []
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier {file_path}: {e}")
        return [], [], []

def display_matrices(cost_matrix, supply, demand):
    """Afficher les matrices de manière formatée et soignée."""
    if not cost_matrix:
        print("Aucune donnée à afficher.")
        return

    print("\n=====| Matrice des coûts |=====")
    print("+" + "---+" * len(cost_matrix[0]))
    for row in cost_matrix:
        print("|" + "|".join(map(lambda x: f"{x:3d}", row)) + "|")
        print("+" + "---+" * len(cost_matrix[0]))
    print("\nProvisions:", "|".join(map(lambda x: f"{x:3d}", supply)))
    print("Demandes:", "|".join(map(lambda x: f"{x:3d}", demand)))
    print("\n")

def initialize_solution(method, cost_matrix, supply, demand):
    if method == 1:
        return northwest_corner_method(cost_matrix, supply, demand)
    else:
        return balas_hammer(cost_matrix, supply, demand)

def northwest_corner_method(cost_matrix, supply, demand):
    n = len(supply)
    m = len(demand)
    solution = [[0] * m for _ in range(n)]
    i, j = 0, 0
    while i < n and j < m:
        amount = min(supply[i], demand[j])
        solution[i][j] = amount
        supply[i] -= amount
        demand[j] -= amount
        if supply[i] == 0 and i < n - 1:
            i += 1
        elif demand[j] == 0 and j < m - 1:
            j += 1
        if supply[i] == 0 and demand[j] == 0:
            break
    return solution


import random

def balas_hammer(cost_matrix, supply, demand):
    """
    Implémente l'algorithme de Balas-Hammer pour trouver une solution initiale.

    :param cost_matrix: Matrice des coûts.
    :param supply: Liste des provisions.
    :param demand: Liste des demandes.
    :return: Solution de transport avec les quantités transportées.
    """
    n = len(supply)
    m = len(demand)
    if sum(supply) != sum(demand):
        print("Problème non équilibré, nous ne pouvons pas appliquer Balas-Hammer")
        return [[0] * m for _ in range(n)]

    print("Problème équilibré, nous pouvons appliquer Balas-Hammer")
    transport_solution = [[0] * m for _ in range(n)]

    while sum(supply) > 0 and sum(demand) > 0:
        penalties = []
        for i in range(n):
            if supply[i] > 0:
                costs = [cost_matrix[i][j] for j in range(m) if demand[j] > 0]
                if len(costs) > 1:
                    sorted_costs = sorted(costs)
                    penalties.append((sorted_costs[1] - sorted_costs[0], 'row', i))
        for j in range(m):
            if demand[j] > 0:
                costs = [cost_matrix[i][j] for i in range(n) if supply[i] > 0]
                if len(costs) > 1:
                    sorted_costs = sorted(costs)
                    penalties.append((sorted_costs[1] - sorted_costs[0], 'col', j))

        if not penalties:
            for i in range(n):
                for j in range(m):
                    if supply[i] > 0 and demand[j] > 0:
                        amount = min(supply[i], demand[j])
                        transport_solution[i][j] += amount
                        supply[i] -= amount
                        demand[j] -= amount
                        print(f"Forçage de remplissage: ({i}, {j}) avec la quantité {amount}.")
            continue

        penalties.sort(reverse=True, key=lambda x: x[0])
        max_penalty = penalties[0][0]
        max_penalties = [p for p in penalties if p[0] == max_penalty]

        print(f"Pénalité maximale de {max_penalty} trouvée en :")
        for penalty in max_penalties:
            if penalty[1] == 'row':
                print(f" - Ligne {penalty[2]}")
            else:
                print(f" - Colonne {penalty[2]}")

        if len(max_penalties) > 1:
            selected_penalty = random.choice(max_penalties)
            print("Plusieurs emplacements pour la pénalité maximale ont été détectés.")
            print(f"Choix aléatoire : {'Ligne' if selected_penalty[1] == 'row' else 'Colonne'} {selected_penalty[2]}")
        else:
            selected_penalty = max_penalties[0]

        p_type, idx = selected_penalty[1], selected_penalty[2]

        if p_type == 'row':
            min_cost_index = min((cost_matrix[idx][j], j) for j in range(m) if demand[j] > 0)[1]
            quantity_to_fill = min(supply[idx], demand[min_cost_index])
            print(f"Remplir l'arête ({idx}, {min_cost_index}) avec la quantité {quantity_to_fill}.")
            transport_solution[idx][min_cost_index] += quantity_to_fill
            supply[idx] -= quantity_to_fill
            demand[min_cost_index] -= quantity_to_fill
        elif p_type == 'col':
            min_cost_index = min((cost_matrix[i][idx], i) for i in range(n) if supply[i] > 0)[1]
            quantity_to_fill = min(supply[min_cost_index], demand[idx])
            print(f"Remplir l'arête ({min_cost_index}, {idx}) avec la quantité {quantity_to_fill}.")
            transport_solution[min_cost_index][idx] += quantity_to_fill
            supply[min_cost_index] -= quantity_to_fill
            demand[idx] -= quantity_to_fill

    return transport_solution




def potential_step_method(cost_matrix, initial_solution, supply, demand):
    optimal = False
    while not optimal:
        display_current_solution_and_cost(initial_solution, cost_matrix)
        if is_degenerate(initial_solution):
            initial_solution = modify_to_non_degenerate(initial_solution, supply, demand)
            print("Modified to non-degenerate solution:", initial_solution)
        u, v = calculate_potentials(cost_matrix, initial_solution, supply, demand)
        display_potential_and_marginal_costs(cost_matrix, initial_solution, u, v)
        cycle = detect_cycle(initial_solution)
        if cycle:
            print("Cycle detected, optimizing transport along the cycle.")
            initial_solution = maximize_transport_on_cycle(initial_solution, cycle)
        optimal, edge_to_add = is_solution_optimal(cost_matrix, initial_solution, u, v)
        if not optimal:
            print(f"Suboptimal solution, improving by adding edge: {edge_to_add}")
            initial_solution = add_edge_to_solution(initial_solution, edge_to_add, supply, demand)
        else:
            print("Optimal solution found.")
    total_cost = calculate_total_cost(cost_matrix, initial_solution)
    print("Final optimal transport solution:", initial_solution)
    print("Total transport cost:", total_cost)
    return initial_solution, total_cost

def display_current_solution_and_cost(transport_solution, cost_matrix):
    total_cost = calculate_total_cost(cost_matrix, transport_solution)
    print("Current Transport Solution:")
    for row in transport_solution:
        print(row)
    print("Total Transport Cost:", total_cost)

def calculate_total_cost(cost_matrix, transport_solution):
    total_cost = 0
    for i in range(len(transport_solution)):
        for j in range(len(transport_solution[i])):
            total_cost += transport_solution[i][j] * cost_matrix[i][j]
    return total_cost

def calculate_potentials(cost_matrix, transport_solution, supply, demand):
    num_suppliers, num_clients = len(supply), len(demand)
    u = [None] * num_suppliers
    v = [None] * num_clients
    u[0] = 0  # Initialiser le premier potentiel pour démarrer la propagation

    change = True
    while change:
        change = False
        for i in range(num_suppliers):
            for j in range(num_clients):
                if transport_solution[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = cost_matrix[i][j] - u[i]
                        change = True
                    elif v[j] is not None and u[i] is None:
                        u[i] = cost_matrix[i][j] - v[j]
                        change = True

    # S'assurer que tous les potentiels sont définis pour éviter les valeurs None
    for i in range(num_suppliers):
        if u[i] is None:
            u[i] = 0  # ou une autre valeur basée sur un critère spécifique
    for j in range(num_clients):
        if v[j] is None:
            v[j] = 0  # ou une autre valeur basée sur un critère spécifique

    return u, v


def display_potential_and_marginal_costs(cost_matrix, transport_solution, u, v):
    num_suppliers = len(u)
    num_clients = len(v)
    print("Potentiels des Fournisseurs (u):", u)
    print("Potentiels des Clients (v):", v)

    # Calcul des coûts marginaux
    marginal_costs = []
    for i in range(num_suppliers):
        marginal_row = []
        for j in range(num_clients):
            if u[i] is not None and v[j] is not None:
                mc = cost_matrix[i][j] - u[i] - v[j]
            else:
                mc = None
            marginal_row.append(mc)
        marginal_costs.append(marginal_row)

    print("\n=====| Coûts Marginaux |=====")
    print("+" + "-----+" * num_clients)
    for row in marginal_costs:
        print("|" + "|".join(f"{x:5}" if x is not None else "-" for x in row) + "|")
        print("+" + "-----+" * num_clients)
    print("\n")    
    return marginal_costs

def is_degenerate(transport_solution):
    num_suppliers = len(transport_solution)
    num_clients = len(transport_solution[0])
    min_required_allocations = num_suppliers + num_clients - 1
    positive_count = sum(1 for row in transport_solution for value in row if value > 0)
    return positive_count < min_required_allocations

def modify_to_non_degenerate(transport_solution, supply, demand):
    num_suppliers = len(transport_solution)
    num_clients = len(transport_solution[0])
    epsilon = 1e-5
    for i in range(num_suppliers):
        for j in range(num_clients):
            if transport_solution[i][j] == 0 and supply[i] > epsilon and demand[j] > epsilon:
                transport_solution[i][j] += epsilon
                supply[i] -= epsilon
                demand[j] -= epsilon
                if not is_degenerate(transport_solution):
                    return transport_solution
    return transport_solution

def detect_cycle(transport_solution):
    num_suppliers = len(transport_solution)
    num_clients = len(transport_solution[0])
    visited = [[False] * num_clients for _ in range(num_suppliers)]
    parent = [[None] * num_clients for _ in range(num_suppliers)]
    queue = deque()

    for i in range(num_suppliers):
        for j in range(num_clients):
            if transport_solution[i][j] > 0 and not visited[i][j]:
                visited[i][j] = True
                queue.append((i, j))
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < num_suppliers and 0 <= ny < num_clients:
                            if transport_solution[nx][ny] > 0 and not visited[nx][ny]:
                                visited[nx][ny] = True
                                parent[nx][ny] = (x, y)
                                queue.append((nx, ny))
                            elif visited[nx][ny] and (nx, ny) != (x, y):
                                return trace_path(nx, ny, parent)
    return None

def trace_path(s, t, parent):
    path = []
    while parent[s][t] is not None:
        path.append((s, t))
        s, t = parent[s][t]
    path.append((s, t))
    return path[::-1]

def maximize_transport_on_cycle(transport_solution, cycle):
    if not cycle:
        return transport_solution
    min_qty = min(transport_solution[i][j] for i, j in cycle)
    for i, j in cycle:
        if i < j:
            transport_solution[i][j] += min_qty
        else:
            transport_solution[i][j] -= min_qty
            if transport_solution[i][j] < 0:
                transport_solution[i][j] = 0
    return transport_solution

def is_solution_optimal(cost_matrix, transport_solution, u, v):
    num_suppliers = len(transport_solution)
    num_clients = len(transport_solution[0])
    for i in range(num_suppliers):
        for j in range(num_clients):
            if transport_solution[i][j] == 0:
                if u[i] is not None and v[j] is not None:
                    reduced_cost = cost_matrix[i][j] - u[i] - v[j]
                    if reduced_cost < 0:
                        return False, (i, j)
    return True, None

def adjust_transport(transport_solution, edge_to_add, supply, demand):
    i, j = edge_to_add
    amount_to_adjust = min(supply[i], demand[j])
    transport_solution[i][j] += amount_to_adjust
    supply[i] -= amount_to_adjust
    demand[j] -= amount_to_adjust
    print(f"Adjusted transport by {amount_to_adjust} units from supplier {i} to client {j}.")
    print(f"Remaining supply at supplier {i}: {supply[i]} units.")
    print(f"Remaining demand at client {j}: {demand[j]} units.")

def add_edge_to_solution(transport_solution, edge_to_add, supply, demand):
    i, j = edge_to_add
    if supply[i] > 0 and demand[j] > 0:
        amount_to_add = min(supply[i], demand[j])
        transport_solution[i][j] += amount_to_add
        supply[i] -= amount_to_add
        demand[j] -= amount_to_add
        print(f"Edge added between supplier {i} and client {j}, amount: {amount_to_add}")
        print(f"Remaining supply at supplier {i}: {supply[i]}")
        print(f"Remaining demand at client {j}: {demand[j]}")
    return transport_solution


def update_potentials_after_changes(cost_matrix, transport_solution, supply, demand):
    num_suppliers = len(supply)
    num_clients = len(demand)
    u = [None] * num_suppliers
    v = [None] * num_clients
    u[0] = 0
    changes = True
    while changes:
        changes = False
        for i in range(num_suppliers):
            for j in range(num_clients):
                if transport_solution[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = cost_matrix[i][j] - u[i]
                        changes = True
                    elif v[j] is not None and u[i] is None:
                        u[i] = cost_matrix[i][j] - v[j]
                        changes = True
    if None in u:
        for i in range(num_suppliers):
            if u[i] is None:
                u[i] = 0
    if None in v:
        for j in range(num_clients):
            if v[j] is None:
                v[j] = 0
    return u, v
