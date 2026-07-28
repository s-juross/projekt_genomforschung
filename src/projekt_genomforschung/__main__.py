from pathlib import Path

from data_download import download_dataset
from data_harmonization import harmonize_dataset, remove_duplicated
from chemical_feature_addition import get_smiles, fetch_smiles_by_cid, get_mfp, get_pharmacophore_features, smiles_to_morgan_columns
from baseline_models import randomForest_genomic, randomForest_chemical, elasticNet_genomic, elasticNet_chemical

def main():
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("Step 1: Downloading dataset...")
    download_dataset()

    print("Step 2: Harmonizing dataset...")
    master_df = harmonize_dataset(output_dir=data_dir)

    print("Step 3: Adding chemical features...")
    print("SMILES...")
    master_df = get_smiles(master_df)

    # Some drugs are duplicated under a different name. If Model ID and SMILES are the same, 
    # we assume it's the same drug-cell line combination and can be averaged.
    master_df = remove_duplicated(master_df)

    print("Morgan fingerprints...")
    if 'MorganFP' not in master_df.columns:
        master_df['MorganFP'] = master_df['SMILES'].apply(get_mfp)
    
    print("Pharmacophore features...")
    if 'PharmacophoreFeatures' not in master_df.columns:
        master_df['PharmacophoreFeatures'] = master_df['SMILES'].apply(get_pharmacophore_features)
    
    # save
    master_df.to_pickle("harmonized_data.pkl")
    # load
    #master_df = pd.read_pickle("harmonized_data.pkl")
    print(f"Done. Processed file saved to: {master_df}")
    
if __name__ == "__main__":
    main()