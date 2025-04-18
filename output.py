import matplotlib.pyplot as plt

# Constants from your CUDA program
NUM_CITIES = 10
POP_SIZE = 2000
MAX_GEN = 2000
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.25

def read_route_from_file(filename):
    """Read coordinates from the best_route.txt file"""
    x_coords = []
    y_coords = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split())
            x_coords.append(x)
            y_coords.append(y)
    return x_coords, y_coords

def visualize_tsp_solution(x_coords, y_coords, cities_names, total_distance):
    """Visualize the TSP solution with the optimal route"""
    fig, ax = plt.subplots()
    
    # Plot all possible connections between cities (gray lines)
    for i in range(len(x_coords)-1):  # -1 because last point is duplicate of first
        for j in range(i+1, len(x_coords)-1):
            ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                    'k-', alpha=0.09, linewidth=1)
    
    # Plot the optimal route (green line with markers)
    ax.plot(x_coords, y_coords, '--go', label='Best Route', linewidth=2.5)
    plt.legend()
    
    # Add city labels with numbers
    for i, (x, y) in enumerate(zip(x_coords[:-1], y_coords[:-1])):  # exclude last duplicate point
        ax.annotate(f"{i+1}- {cities_names[i]}", (x, y), fontsize=12)
    
    # Set titles
    plt.title(label="TSP Best Route Using Genetic Algorithm",
              fontsize=20,
              color="k")
    
    str_params = f"""
{MAX_GEN} Generations
{POP_SIZE} Population Size
{CROSSOVER_RATE*100}% Crossover Rate
{MUTATION_RATE*100}% Mutation Rate
"""
    plt.suptitle(f"Total Distance: {total_distance:.3f}" + str_params, 
                 fontsize=14, y=1.05)
    
    fig.set_size_inches(16, 12)
    plt.grid(color='k', linestyle='dotted', alpha=0.3)
    plt.savefig('tsp_solution.png', bbox_inches='tight')
    plt.show()

# City data
cities_names = ["Gliwice", "Cairo", "Rome", "Krakow", "Paris", 
                "Alexandria", "Berlin", "Tokyo", "Hong Kong", "Rio"]

# Read the best route from file
x_coords, y_coords = read_route_from_file('best_route.txt')

# Calculate total distance
total_distance = 0.0
for i in range(len(x_coords)-1):
    dx = x_coords[i+1] - x_coords[i]
    dy = y_coords[i+1] - y_coords[i]
    total_distance += (dx**2 + dy**2)**0.5

# Visualize the solution
visualize_tsp_solution(x_coords, y_coords, cities_names, total_distance)