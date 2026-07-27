from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures, rdFingerprintGenerator
from rdkit import RDConfig
import os
import pandas as pd
import numpy as np
import pubchempy as pcp
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=Warning)

def get_mfp(smiles):
    """
    Takes a SMILES string and returns a numpy array of the Morgan fingerprint (1024 bits).
    """
    try:
        morganfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        mol = Chem.MolFromSmiles(smiles)
        return np.array(morganfp.GetFingerprint(mol))
    except:
        return None

def get_pharmacophore_features(smiles):
    """
    Takes a SMILES string and returns a dictionary containing the number of chemical features found (e.g., {‘Donor’: 2, ‘Acceptor’: 4}).
    """
    # initialise the "feature factory"
    # RDKIT provides a standard file (BaseFeatures.fdef) that defines what exactly a H-donor, acceptor, etc. is.
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name) # is a MolChemicalFeatureFactory object
    if pd.isna(smiles):
        return {}
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
        
    # extract all features for the molecule
    features = factory.GetFeaturesForMol(mol)
    
    # count how many times each feature family appears
    feature_counts = {}
    for feat in features:
        family = feat.GetFamily()
        feature_counts[family] = feature_counts.get(family, 0) + 1
            
    return feature_counts

def smiles_to_morgan_columns(df, smiles_col='SMILES', n_bits=1024, radius=2):
    # Calculates 1024-bit Morgan Fingerprints and expands them into individual columns.
    mols = [Chem.MolFromSmiles(s) if pd.notna(s) else None for s in df[smiles_col]]

    fingerprints = []
    for mol in mols:
        if mol:
            # Generate the bit vector
            morganfp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            # Convert to list of 0s and 1s
            fingerprints.append(list(morganfp.GetFingerprint(mol)))
        else:
            # Maintain row alignment with a list of zeros if SMILES was invalid
            fingerprints.append([0] * n_bits)

    # Create new DataFrame from the list of lists
    fp_df = pd.DataFrame(fingerprints, index=df.index)

    # Rename columns to 'Bit_0', 'Bit_1', etc.
    fp_df.columns = [f'Bit_{i}' for i in range(n_bits)]
    fp_df = fp_df.astype('uint8')

    # Concatenate with the original DataFrame
    result_df = pd.concat([df, fp_df], axis=1)
    print("Fingerprints expanded into columns.")
    return result_df


# Define the function to fetch SMILES via PubChem CID
def fetch_smiles_by_cid(drug_id):
    try:
        # Clean the ID (ensure it is an integer)
        clean_cid = int(drug_id)
        
        # Add a small delay to respect PubChem API rate limits (5 requests/sec)
        time.sleep(0.2) 
        
        # Fetch the compound and return the canonical SMILES
        compound = pcp.get_compounds(clean_cid, namespace='cid')[0]
        return compound.connectivity_smiles
        
    except Exception as e:
        # Print the exact ID that failed for easier debugging later
        print(f"Failed to fetch SMILES for CID {drug_id}. Reason: {e}")
        return None