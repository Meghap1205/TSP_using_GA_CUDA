#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

#define NUM_CITIES 10
#define POP_SIZE 2000         // Higher population = more diversity
#define MAX_GEN 2000          // More generations = better convergence
#define CROSSOVER_RATE 0.9f   // Higher crossover rate to explore new solutions
#define MUTATION_RATE 0.25f   // Slightly higher mutation to avoid local optima


// City coordinates
__constant__ float city_coords[NUM_CITIES][2] = {
    {0,1}, {3,2}, {6,1}, {7,4.5}, {15,-1},
    {10,2.5}, {16,11}, {5,6}, {8,9}, {1.5,12}
};

typedef struct {
    int path[NUM_CITIES];
    float fitness;
} Individual;

// Initialize random states
__global__ void initCurand(curandState* state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POP_SIZE) curand_init(seed, idx, 0, &state[idx]);
}

// Parallel population initialization
__global__ void initPopulation(Individual* population, curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= POP_SIZE) return;
    
    curandState localState = states[idx];
    Individual* ind = &population[idx];
    
    // Initialize with sequential path
    for(int i = 0; i < NUM_CITIES; i++) {
        ind->path[i] = i;
    }
    
    // Fisher-Yates shuffle
    for(int i = NUM_CITIES-1; i > 0; i--) {
        int j = curand(&localState) % (i+1);
        int temp = ind->path[i];
        ind->path[i] = ind->path[j];
        ind->path[j] = temp;
    }
}

// Distance calculation between two cities
__device__ float cityDistance(int city1, int city2) {
    float dx = city_coords[city1][0] - city_coords[city2][0];
    float dy = city_coords[city1][1] - city_coords[city2][1];
    return sqrtf(dx*dx + dy*dy);
}

// Parallel fitness calculation
__global__ void calculateFitness(Individual* population) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= POP_SIZE) return;
    
    Individual* ind = &population[idx];
    float total_dist = 0.0f;
    
    for(int i = 0; i < NUM_CITIES-1; i++) {
        total_dist += cityDistance(ind->path[i], ind->path[i+1]);
    }
    // Return to starting city
    total_dist += cityDistance(ind->path[NUM_CITIES-1], ind->path[0]);
    
    // Fitness is inverse of distance (higher is better)
    ind->fitness = 1.0f / total_dist;
}

// Check if city exists in path
__device__ bool isInPath(int* path, int city, int length) {
    for(int i = 0; i < length; i++) {
        if(path[i] == city) return true;
    }
    return false;
}

// Roulette wheel selection
__global__ void selection(Individual* population, Individual* parents, 
                        float totalFitness, curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= POP_SIZE * CROSSOVER_RATE) return;
    
    curandState localState = states[idx];
    float randVal = curand_uniform(&localState) * totalFitness;
    
    float cumulative = 0.0f;
    for(int i = 0; i < POP_SIZE; i++) {
        cumulative += population[i].fitness;
        if(cumulative >= randVal) {
            parents[idx] = population[i];
            break;
        }
    }
}

// Ordered crossover
__global__ void crossover(Individual* parents, Individual* offspring, 
                        curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= (POP_SIZE * CROSSOVER_RATE)/2) return;
    
    curandState localState = states[idx];
    int parent1_idx = 2 * idx;
    int parent2_idx = 2 * idx + 1;
    
    // Random crossover point
    int cut = 1 + (curand(&localState) % (NUM_CITIES-2));
    
    // Create two offspring
    for(int o = 0; o < 2; o++) {
        Individual* parent1 = &parents[parent1_idx];
        Individual* parent2 = &parents[parent2_idx];
        Individual* child = &offspring[2*idx + o];
        
        // Take first part from parent1
        for(int i = 0; i < cut; i++) {
            child->path[i] = parent1->path[i];
        }
        
        // Fill remaining from parent2 in order
        int child_pos = cut;
        for(int i = 0; i < NUM_CITIES; i++) {
            if(!isInPath(child->path, parent2->path[i], cut)) {
                child->path[child_pos++] = parent2->path[i];
            }
        }
        
        // Swap parents for second offspring
        Individual* temp = parent1;
        parent1 = parent2;
        parent2 = temp;
    }
}

// Swap mutation
__global__ void mutation(Individual* offspring, curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= POP_SIZE * CROSSOVER_RATE) return;
    
    curandState localState = states[idx];
    float randVal = curand_uniform(&localState);
    
    if(randVal < MUTATION_RATE) {
        // Select two distinct random positions
        int pos1 = curand(&localState) % NUM_CITIES;
        int pos2;
        do {
            pos2 = curand(&localState) % NUM_CITIES;
        } while(pos1 == pos2);
        
        // Swap cities
        int temp = offspring[idx].path[pos1];
        offspring[idx].path[pos1] = offspring[idx].path[pos2];
        offspring[idx].path[pos2] = temp;
    }
}

// Replacement: combine parents and offspring
__global__ void replacement(Individual* population, Individual* offspring) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= POP_SIZE * CROSSOVER_RATE) return;
    
    // Replace worst individuals with offspring
    population[POP_SIZE - 1 - idx] = offspring[idx];
}

// Fitness functor for thrust::transform_reduce
struct FitnessFunctor {
    __device__ float operator()(const Individual& ind) {
        return ind.fitness;
    }
};

// Host function to copy city coordinates and write to file
void saveBestRouteToFile(const Individual& best_individual) {
    float city_coords_host[NUM_CITIES][2];
    cudaMemcpyFromSymbol(city_coords_host, city_coords, sizeof(city_coords_host));
    
    FILE *fp = fopen("best_route.txt", "w");
    for(int i = 0; i < NUM_CITIES; i++) {
        int city = best_individual.path[i];
        fprintf(fp, "%f %f\n", city_coords_host[city][0], city_coords_host[city][1]);
    }
    // Return to start city
    int first_city = best_individual.path[0];
    fprintf(fp, "%f %f\n", city_coords_host[first_city][0], city_coords_host[first_city][1]);
    fclose(fp);
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start, 0);

    // Allocate memory
    Individual *d_population, *d_parents, *d_offspring, *h_population;
    curandState *d_states;
    
    cudaMalloc(&d_population, POP_SIZE * sizeof(Individual));
    cudaMalloc(&d_parents, (POP_SIZE * CROSSOVER_RATE) * sizeof(Individual));
    cudaMalloc(&d_offspring, (POP_SIZE * CROSSOVER_RATE) * sizeof(Individual));
    cudaMalloc(&d_states, POP_SIZE * sizeof(curandState));
    h_population = (Individual*)malloc(POP_SIZE * sizeof(Individual));
    
    // Initialize random states
    initCurand<<<(POP_SIZE+255)/256, 256>>>(d_states, time(NULL));
    
    // Initialize population
    initPopulation<<<(POP_SIZE+255)/256, 256>>>(d_population, d_states);
    
    for(int gen = 0; gen < MAX_GEN; gen++) {
        // Calculate fitness
        calculateFitness<<<(POP_SIZE+255)/256, 256>>>(d_population);
        
        // Sum fitness for selection using thrust
        thrust::device_ptr<Individual> pop_ptr(d_population);
        float totalFitness = thrust::transform_reduce(
            pop_ptr, pop_ptr + POP_SIZE,
            FitnessFunctor(),
            0.0f,
            thrust::plus<float>());
        
        // Selection
        selection<<<((POP_SIZE*CROSSOVER_RATE)+255)/256, 256>>>(d_population, d_parents, totalFitness, d_states);
        
        // Crossover
        crossover<<<((POP_SIZE*CROSSOVER_RATE/2)+255)/256, 256>>>(d_parents, d_offspring, d_states);
        
        // Mutation
        mutation<<<((POP_SIZE*CROSSOVER_RATE)+255)/256, 256>>>(d_offspring, d_states);
        
        // Calculate offspring fitness
        calculateFitness<<<(POP_SIZE*CROSSOVER_RATE+255)/256, 256>>>(d_offspring);
        
        // Replacement
        replacement<<<((POP_SIZE*CROSSOVER_RATE)+255)/256, 256>>>(d_population, d_offspring);
        
        if (gen % 10 == 0) {
            cudaMemcpy(h_population, d_population, sizeof(Individual), cudaMemcpyDeviceToHost);
            float best_fitness = 0.0f;
            for (int i = 0; i < POP_SIZE; i++) {
                if (h_population[i].fitness > best_fitness) {
                    best_fitness = h_population[i].fitness;
                }
            }
            printf("Generation %d: Best Distance = %.3f\n", gen, 1.0f / best_fitness);
        }
    }
    
    // Copy final results back to host
    cudaMemcpy(h_population, d_population, POP_SIZE * sizeof(Individual), cudaMemcpyDeviceToHost);
    
    // Find best solution
    float min_distance = INFINITY;
    int best_idx = 0;
    for(int i = 0; i < POP_SIZE; i++) {
        float distance = 1.0f / h_population[i].fitness;
        if(distance < min_distance) {
            min_distance = distance;
            best_idx = i;
        }
    }

    // Record stop time and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);  // Ensure all GPU work is done

    // Compute elapsed time (in milliseconds)
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    
    // Print results
    printf("\nFinal Result:\n");
    printf("Optimal distance: %.3f\n", min_distance);
    printf("Best route: ");
    for(int i = 0; i < NUM_CITIES; i++) {
        printf("%d ", h_population[best_idx].path[i]);
    }
    printf("%d\n", h_population[best_idx].path[0]); // Return to start
    
    // Save coordinates for plotting
    saveBestRouteToFile(h_population[best_idx]);
    
    printf("\nExecution Time: %.3f ms\n", elapsed_time_ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_population);
    cudaFree(d_population);
    cudaFree(d_parents);
    cudaFree(d_offspring);
    cudaFree(d_states);
    
    return 0;
}