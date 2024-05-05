from functions import *

def main():
    while True:
        # Liste des fichiers correspondant aux différents problèmes
        base_path = 'Fichierstest/'
        files = [
            't1.txt', 't2.txt', 't3.txt', 't4.txt',
            't5.txt', 't6.txt', 't7.txt', 't8.txt',
            't9.txt', 't10.txt', 't11.txt', 't12.txt'
        ]

        # Affichage du menu et choix de l'utilisateur
        print("\nChoisissez un fichier de problème à analyser (1-12) ou entrez 0 pour quitter :")
        for index, file in enumerate(files, start=1):
            print(f"{index}. {file}")
        choice = int(input("Entrez le numéro du problème ou 0 pour quitter: "))
        
        if choice == 0:
            break
        if not 1 <= choice <= 12:
            print("Choix invalide, veuillez choisir un numéro entre 1 et 12 ou 0 pour quitter.")
            continue

        # Détermination du chemin du fichier basé sur le choix
        file_path = base_path + files[choice - 1]
        
        # Lecture et affichage des données
        cost_matrix, supply, demand = read_data(file_path)
        display_matrices(cost_matrix, supply, demand)
        
        # Choix de l'algorithme d'initialisation
        print("Choisissez l'algorithme pour la solution initiale:")
        print("1. Algorithme du coin Nord-Ouest")
        print("2. Algorithme de Balas-Hammer")
        alg_choice = int(input("Entrez votre choix (1 ou 2): "))
        
        if alg_choice == 1:
            initial_solution = northwest_corner_method(cost_matrix, supply.copy(), demand.copy())
            print("Solution initiale par l'algorithme du coin Nord-Ouest:")
        elif alg_choice == 2:
            initial_solution = balas_hammer(cost_matrix, supply.copy(), demand.copy())
            print("Solution initiale par l'algorithme de Balas-Hammer:")
        else:
            print("Choix invalide, retour au menu principal.")
            continue

        for row in initial_solution:
            print(row)
        
        potential_step_method(cost_matrix, initial_solution, supply, demand)


  
if __name__ == "__main__":
    main()
