import pandas as pd
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import MCS

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

# Define the function to identify MCS between two molecules
def identify_mcs(molecules):
    num_molecules = len(molecules)
    for i in range(num_molecules):
        for j in range(i + 1, num_molecules):
            mol1 = molecules[i]
            mol2 = molecules[j]
            mcs = MCS.FindMCS([mol1, mol2])
            # Do something with the MCS if needed

# Measure the time for MCS identification for 1 core
num_runs = 1  # Number of runs for each configuration
times = []

print("Average Time Taken (seconds)")
print("------------------------------")

def main():
    elapsed_times = []
    for _ in range(num_runs):
        start_time = time.time()
        identify_mcs(molecules)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)

    average_time = np.mean(elapsed_times)
    times.append(average_time)
    print(f"{average_time:.2f} seconds")

if __name__ == '__main__':
    main()
