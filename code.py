import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from io import StringIO
from matplotlib.animation import FuncAnimation

# ---------- Gestion des instances ----------

def load_all_instances(folder_path):
    """Charge toutes les instances depuis un dossier spécifié."""
    instances = {}
    try:
        print(f"Recherche des fichiers dans : {folder_path}")
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".txt"):
                    full_path = os.path.join(root, filename)
                    print(f"Traitement du fichier : {filename}")
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    instances[filename] = parse_instance(content)
    except Exception as e:
        print(f"Erreur lors du chargement des instances : {e}")
    return instances

def parse_instance(content):
    """Parse une instance EVRP à partir du contenu du fichier dataset."""
    lines = content.split('\n')
    node_lines = []
    vehicle_params = {}
    
    print("Lignes du fichier :")
    for line in lines:
        line = line.strip()
        print(f"  {line}")
        if not line or line.startswith('#'):
            continue
        if line.startswith(('L', 'C', 'Q', 'r', 'g', 'v')):
            try:
                parts = line.split('/')
                if len(parts) < 3:
                    print(f"  Ligne de paramètre invalide ignorée : {line}")
                    continue
                key, value = parts[0].strip(), parts[1].strip()
                vehicle_params[key] = float(value)
            except Exception as e:
                print(f"  Erreur lors du parsing de la ligne de paramètre '{line}' : {e}")
        else:
            node_lines.append(line)

    if not node_lines:
        print("Aucune ligne de nœud trouvée dans le fichier.")
        return None

    try:
        df = pd.read_csv(StringIO('\n'.join(node_lines)), sep=r'\s+', engine='python')
        if df.empty:
            print("Le fichier contient des données de nœuds vides.")
            return None
    except Exception as e:
        print(f"Erreur lors du parsing des nœuds : {e}")
        return None

    cities = []
    for idx, row in df.iterrows():
        cities.append({
            'id': idx,
            'name': row['StringID'],
            'lon': row['x'],
            'lat': row['y'],
            'type': row['Type']
        })

    vehicle_config = {
        'battery_capacity': vehicle_params.get('Q', 194.6),
        'consumption_rate': vehicle_params.get('r', 1.0),
        'max_distance': vehicle_params.get('Q', 194.6) / vehicle_params.get('r', 1.0)
    }

    return {
        'nodes': cities,
        'vehicle_config': vehicle_config
    }

# ---------- Classe représentant une instance EVRP ----------

class EVRPInstance:
    def __init__(self, nodes, vehicle_config):
        """Initialise une instance EVRP avec des nœuds et une configuration de véhicule."""
        self.nodes = nodes
        self.vehicle_config = vehicle_config
        self.depot = None
        self.customers = []
        self.stations = []
        
        # Séparer les nœuds par type
        for node in nodes:
            if node['type'] == 'd':
                self.depot = node
            elif node['type'] == 'c':
                self.customers.append(node)
            elif node['type'] == 'f':
                self.stations.append(node)

    def get_distance(self, node1, node2):
        """Calcule la distance euclidienne entre deux nœuds."""
        dx = node1['lon'] - node2['lon']
        dy = node1['lat'] - node2['lat']
        return math.sqrt(dx**2 + dy**2)

# ---------- Algorithme Nearest Neighbor ----------

class NearestNeighborAlgorithm:
    def __init__(self, instance):
        """Initialise l'algorithme Nearest Neighbor."""
        self.instance = instance
        self.metrics = {
            'distance_calculations': 0,
            'recharge_checks': 0,
            'nodes_visited': 0
        }

    def find_nearest_node(self, current_node, unvisited_nodes):
        """Trouve le nœud non visité le plus proche du nœud actuel."""
        nearest = None
        min_distance = float('inf')
        
        for node in unvisited_nodes:
            distance = self.instance.get_distance(current_node, node)
            self.metrics['distance_calculations'] += 1
            
            if distance < min_distance:
                min_distance = distance
                nearest = node
        
        return nearest, min_distance

    def find_nearest_station(self, current_node):
        """Trouve la station de recharge la plus proche."""
        nearest_station = None
        min_distance = float('inf')
        
        for station in self.instance.stations:
            distance = self.instance.get_distance(current_node, station)
            self.metrics['distance_calculations'] += 1
            
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
        
        return nearest_station, min_distance

    def evaluate_solution(self, solution):
        """Évalue une solution (distance, recharges)."""
        total_distance = 0
        energy_consumed = 0
        recharges = 0
        battery_capacity = self.instance.vehicle_config['battery_capacity']
        consumption_rate = self.instance.vehicle_config['consumption_rate']
        battery = battery_capacity

        for i in range(len(solution) - 1):
            d = self.instance.get_distance(solution[i], solution[i + 1])
            energy_needed = d * consumption_rate

            if battery < energy_needed:
                recharges += 1
                battery = battery_capacity

            battery -= energy_needed
            total_distance += d
            energy_consumed += energy_needed

        return {
            'cost': round(total_distance, 2),
            'energy_consumed': round(energy_consumed, 2),
            'recharges': recharges,
            'vehicles_used': 1
        }

    def run(self, start_from_depot=True):
        """Exécute l'algorithme Nearest Neighbor."""
        solution = []
        
        # Choisir le point de départ
        if start_from_depot and self.instance.depot:
            current_node = self.instance.depot
            solution.append(current_node)
        else:
            # Commencer par un client aléatoire
            current_node = self.instance.customers[0]
            solution.append(current_node)
        
        unvisited = self.instance.customers.copy()
        if current_node in unvisited:
            unvisited.remove(current_node)
        
        battery = self.instance.vehicle_config['battery_capacity']
        consumption_rate = self.instance.vehicle_config['consumption_rate']
        
        print(f"\nDépart depuis : {current_node['name']} (Type: {current_node['type']})")
        
        # Visiter tous les clients
        while unvisited:
            self.metrics['nodes_visited'] += 1
            nearest, distance = self.find_nearest_node(current_node, unvisited)
            
            if nearest is None:
                break
            
            energy_needed = distance * consumption_rate
            
            # Vérifier si une recharge est nécessaire
            if battery < energy_needed:
                self.metrics['recharge_checks'] += 1
                # Trouver la station la plus proche
                station, station_distance = self.find_nearest_station(current_node)
                
                if station:
                    print(f"  Recharge nécessaire à la station : {station['name']}")
                    solution.append(station)
                    current_node = station
                    battery = self.instance.vehicle_config['battery_capacity']
                    
                    # Recalculer la distance vers le prochain client
                    distance = self.instance.get_distance(current_node, nearest)
                    energy_needed = distance * consumption_rate
            
            # Se déplacer vers le client le plus proche
            solution.append(nearest)
            battery -= energy_needed
            unvisited.remove(nearest)
            current_node = nearest
            
            print(f"  Visite : {nearest['name']} (Distance: {distance:.2f}, Batterie restante: {battery:.2f})")
        
        # Retour au dépôt
        if self.instance.depot and solution[0] != self.instance.depot:
            solution.append(self.instance.depot)
            distance_to_depot = self.instance.get_distance(current_node, self.instance.depot)
            print(f"  Retour au dépôt : {self.instance.depot['name']} (Distance: {distance_to_depot:.2f})")
        
        # Évaluer la solution finale
        metrics = self.evaluate_solution(solution)
        
        print("\n=== Statistiques de l'algorithme ===")
        print(f"Calculs de distance effectués : {self.metrics['distance_calculations']}")
        print(f"Vérifications de recharge : {self.metrics['recharge_checks']}")
        print(f"Nœuds visités : {self.metrics['nodes_visited']}")
        
        return solution, metrics

# ---------- Visualisation ----------

def plot_solution(solution):
    """Affiche la solution avec animation sur un graphe simple."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Définir les limites des axes
    lons = [node['lon'] for node in solution]
    lats = [node['lat'] for node in solution]
    ax.set_xlim(min(lons) - 10, max(lons) + 10)
    ax.set_ylim(min(lats) - 10, max(lats) + 10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Définir les marqueurs et couleurs par type de nœud
    markers = {'d': 's', 's': '^', 'f': 'D', 'c': 'o'}
    colors = {'d': 'red', 's': 'green', 'f': 'blue', 'c': 'black'}
    labels = {'d': 'Dépôt', 's': 'Satellite', 'f': 'Station', 'c': 'Client'}

    # Tracer les nœuds
    plotted_types = set()
    for node in solution:
        node_type = node['type']
        label = labels[node_type] if node_type not in plotted_types else ""
        ax.plot(node['lon'], node['lat'], marker=markers[node_type], 
                color=colors[node_type], markersize=10, label=label)
        ax.text(node['lon'], node['lat'], node['name'], fontsize=8, ha='right')
        plotted_types.add(node_type)

    edge_trace, = ax.plot([], [], 'r-', lw=2, alpha=0.6)
    text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        edge_trace.set_data([], [])
        text.set_text('')
        return edge_trace, text

    def update(num):
        if num < len(solution) - 1:
            lons = [solution[i]['lon'] for i in range(num + 2)]
            lats = [solution[i]['lat'] for i in range(num + 2)]
            edge_trace.set_data(lons, lats)
            text.set_text(f'{solution[num]["name"]} → {solution[num + 1]["name"]}')
        return edge_trace, text

    ax.set_title("Meilleur trajet trouvé (Nearest Neighbor)")
    ax.legend(loc='upper right')
    ani = FuncAnimation(fig, update, frames=len(solution) - 1, init_func=init, 
                       interval=800, repeat=False)
    plt.show()

def plot_comparison_results(df):
    """Affiche les graphiques comparatifs des résultats."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='start_point', y='cost', data=df)
    plt.title("Comparaison des coûts selon le point de départ")
    plt.xlabel("Point de départ")
    plt.ylabel("Coût (distance totale)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='start_point', y='recharges', data=df)
    plt.title("Comparaison des recharges selon le point de départ")
    plt.xlabel("Point de départ")
    plt.ylabel("Nombre de recharges")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------- Fonction principale ----------

def main():
    """Fonction principale pour exécuter l'algorithme Nearest Neighbor."""
    print("=== Algorithme Nearest Neighbor pour EVRP ===\n")

    folder = r"C:\Users\KHADIJA ELFETOUI\Downloads\2E-EVRP-Instances-v2\manilakbay-2E-EVRP-Instances-dc2c2d0\Type_x\Customer_100"
    dataset = load_all_instances(folder)

    if not dataset:
        print("Aucune instance valide trouvée. Vérifiez le dossier et le format des fichiers.")
        return

    instance_data = list(dataset.values())[0]
    if not instance_data:
        print("Les données de l'instance sont invalides.")
        return

    instance = EVRPInstance(instance_data['nodes'], instance_data['vehicle_config'])

    print(f"\nConfiguration du véhicule :")
    print(f"  Capacité de batterie : {instance.vehicle_config['battery_capacity']}")
    print(f"  Taux de consommation : {instance.vehicle_config['consumption_rate']}")
    print(f"  Distance maximale : {instance.vehicle_config['max_distance']}")
    print(f"\nNombre de clients : {len(instance.customers)}")
    print(f"Nombre de stations : {len(instance.stations)}")
    print(f"Dépôt : {instance.depot['name'] if instance.depot else 'Non défini'}")

    results = []
    num_runs = 5
    start_options = ['depot', 'first_customer']

    print("\n--- Exécution de l'algorithme Nearest Neighbor ---")
    for start_point in start_options:
        for run in range(num_runs):
            print(f"\n{'='*60}")
            print(f"Run {run+1} - Point de départ : {start_point}")
            print(f"{'='*60}")
            
            nn = NearestNeighborAlgorithm(instance)
            best_solution, metrics = nn.run(start_from_depot=(start_point == 'depot'))
            
            results.append({
                'start_point': start_point,
                'run': run + 1,
                'cost': metrics['cost'],
                'recharges': metrics['recharges'],
                'energy_consumed': metrics['energy_consumed']
            })
            
            print(f"\nRésultats du run {run+1} :")
            print(f"  Coût total : {metrics['cost']}")
            print(f"  Recharges : {metrics['recharges']}")
            print(f"  Énergie consommée : {metrics['energy_consumed']}")

    # Sauvegarder et afficher les résultats
    df = pd.DataFrame(results)
    df.to_csv('nn_comparison_results.csv', index=False)
    print("\nRésultats sauvegardés dans 'nn_comparison_results.csv'")

    # Statistiques globales
    print("\n=== Statistiques globales ===")
    print(df.groupby('start_point')[['cost', 'recharges', 'energy_consumed']].agg(['mean', 'min', 'max', 'std']))

    plot_comparison_results(df)

    # Afficher la meilleure solution
    best_idx = df['cost'].idxmin()
    best_run = df.iloc[best_idx]
    print(f"\n=== Meilleure solution ===")
    print(f"Point de départ : {best_run['start_point']}")
    print(f"Coût total : {best_run['cost']}")
    print(f"Recharges : {best_run['recharges']}")
    print(f"Énergie consommée : {best_run['energy_consumed']}")

    # Réexécuter pour obtenir la meilleure solution à visualiser
    nn_final = NearestNeighborAlgorithm(instance)
    final_solution, final_metrics = nn_final.run(start_from_depot=(best_run['start_point'] == 'depot'))
    
    print(f"\nOrdre des nœuds visités : {[n['name'] for n in final_solution]}")
    
    plot_solution(final_solution)

# ---------- Point d'entrée ----------

if __name__ == "__main__":
    main()