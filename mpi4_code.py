from mpi4py import MPI
import matplotlib.pyplot as plt
import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import MCS
from itertools import combinations

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load structures from CSV file
file_path = 'new_final.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Ensure 'smiles' column exists in the DataFrame
if 'smiles' not in df.columns:
    print(f"Error in rank {rank}: 'smiles' column not found in the CSV file.")
    exit()

# Filter out rows with missing or None values in the 'smiles' column
df = df.dropna(subset=['smiles'])

# Extract SMILES strings from the DataFrame
smiles_list = df['smiles'].tolist()

# Convert SMILES strings to RDKit molecules
molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]

# Check if there are valid molecules to perform MCS identification
if not molecules:
    print(f"Error in rank {rank}: No valid molecules found.")
    exit()

# Function to calculate MCS for a pair of molecules
def calculate_mcs(pair):
    mol1, mol2 = pair
    result = MCS.FindMCS([mol1, mol2])
    return result

# Function to distribute workload and gather results using MPI
def calculate_mcs_parallel(num_cores):
    start_time = time.time()

    # Calculate workload distribution
    num_mols = len(molecules)
    chunk_size = num_mols // num_cores
    start_index = rank * chunk_size
    end_index = min((rank + 1) * chunk_size, num_mols)

    # Generate pairs of molecules for each process to handle
    molecule_pairs = combinations(molecules[start_index:end_index], 2)

    # Calculate MCS for each pair
    mcs_results = [calculate_mcs(pair) for pair in molecule_pairs]

    # Gather results from all processes
    all_results = comm.gather(mcs_results, root=0)

    if rank == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_matched = sum(1 for results in all_results for result in results if result is not None)
        return elapsed_time, total_matched
    else:
        return None, None

# Main process
if __name__ == '__main__':
    results_data = []

    for num_cores in range(1, 21):  # Run with 1 to 20 cores
        elapsed_time, total_matched = calculate_mcs_parallel(num_cores)

        # Collect results on rank 0
        data = comm.gather((num_cores, elapsed_time, total_matched), root=0)

        # Master process appends the data
        if rank == 0:
            results_data.extend(data)

            for num_cores, elapsed_time, _ in data:
                print(f"For {num_cores} cores - Total time taken: {elapsed_time:.2f} seconds")

    # Master process plots the graph
    if rank == 0:
        cores = [num_cores for num_cores, _, _ in results_data]
        times = [elapsed_time for _, elapsed_time, _ in results_data]

        plt.plot(cores, times, marker='o')
        plt.title('MCS Identification Time vs. Number of Cores')
        plt.xlabel('Number of Cores')
        plt.ylabel('Time (Seconds)')
        plt.grid(True)

        # Save the graph as a PNG file
        plt.savefig('mpi4py.png')
        plt.show()
