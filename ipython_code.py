import matplotlib.pyplot as plt
import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import MCS
import numpy as np
from ipyparallel import Client
from itertools import combinations

# Load structures from CSV file
file_path = 'new_final.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Ensure 'smiles' column exists in the DataFrame
if 'smiles' not in df.columns:
    print("Error: 'smiles' column not found in the CSV file.")
    exit()

# Filter out rows with missing or None values in the 'smiles' column
df = df.dropna(subset=['smiles'])

# Extract SMILES strings from the DataFrame
smiles_list = df['smiles'].tolist()

# Convert SMILES strings to RDKit molecules
molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]

# Check if there are valid molecules to perform MCS identification
if not molecules:
    print("Error: No valid molecules found.")
    exit()

# Initialize IPython client
client = Client()

# Define the function to identify MCS between two molecules
def identify_mcs(pair):
    mol1, mol2 = pair
    return MCS.FindMCS([mol1, mol2])

# Measure the time for MCS identification for 1, 2, 3, 4, and 5 engines (cores)
cores_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
num_runs = 3  # Number of runs for each configuration
times = []

print("Number of Engines | Average Time Taken (seconds)")
print("--------------------------------------------------")

def main():
    for cores in cores_range:
        elapsed_times = []
        for _ in range(num_runs):
            start_time = time.time()

            # Generate all combinations of pairs of molecules
            pairs = combinations(molecules, 2)
            
            # Distribute tasks across the engines (cores)
            dview = client[:]
            with dview.sync_imports():
                from rdkit import Chem
                from rdkit.Chem import MCS

            # Perform MCS identification for all pairs of molecules in parallel
            results = dview.map_sync(identify_mcs, pairs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)

        average_time = np.mean(elapsed_times)
        times.append(average_time)
        print(f"{cores:16} | {average_time:.2f}")

    # Plot the graph
    plt.plot(cores_range, times, marker='o')
    plt.title('Number of cpu vs Average Time Taken')
    plt.xlabel('Number of cpus')
    plt.ylabel('Average Time Taken (seconds)')
    plt.grid(True)
    plt.xticks(cores_range)

    # Save the graph as a PNG file
    plt.savefig('ipyparallel_all_to_all.png')
    plt.show()

if __name__ == '__main__':
    main()
