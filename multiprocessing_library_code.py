import matplotlib.pyplot as plt
import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import MCS
from multiprocessing import Pool, cpu_count
from itertools import product

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

# Define the ZINC53 molecule
zinc53_smiles = "CS(=O)(=O)CCCNN"
zinc53_mol = Chem.MolFromSmiles(zinc53_smiles)

# Function to calculate MCS for a pair of molecules
def calculate_mcs(pair):
    mol1, mol2 = pair
    return MCS.FindMCS([mol1, mol2])

# Function to calculate MCS identification time for a given number of processes
def calculate_mcs_time(num_processes):
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        pairs = product(molecules, repeat=2)  # Generate all pairs of molecules
        mcs_results = pool.map(calculate_mcs, pairs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_matched = sum(1 for result in mcs_results if result is not None)
    return elapsed_time, total_matched

# List to store results for each core configuration
results_data = []

# Define a main function to be invoked on execution
def main():
    # Calculate MCS identification time for each core configuration (up to 20 cores)
    for num_cores in range(1, cpu_count() + 1):
        elapsed_time, total_matched = calculate_mcs_time(num_cores)
        results_data.append((num_cores, elapsed_time, total_matched))

    # Print results
    for num_cores, elapsed_time, total_matched in results_data:
        print(f"For {num_cores} cores - Total time taken: {elapsed_time:.2f} seconds")

    # Plotting the graph
    cores = [num_cores for num_cores, _, _ in results_data]
    times = [elapsed_time for _, elapsed_time, _ in results_data]

    plt.plot(cores, times, marker='o')
    plt.title('MCS Identification Time vs. Number of Cores')
    plt.xlabel('Number of Cores')
    plt.ylabel('Time (Seconds)')
    plt.grid(True)

    # Save the graph as a PNG file
    plt.savefig('Multiprocessing.png')

    plt.show()

# Ensures that the code inside the "main" function is only executed
# when the script is run directly, not when it's imported as a module
if __name__ == '__main__':
    main()
