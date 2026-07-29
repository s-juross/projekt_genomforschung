from pathlib import Path
import argparse
import sys

from data_download import download_dataset, get_harmonized_dataset
from data_harmonization import harmonize_dataset, remove_duplicated
from chemical_feature_addition import get_smiles, fetch_smiles_by_cid, get_mfp, get_pharmacophore_features, smiles_to_morgan_columns
from baseline_models import randomForest_genomic, randomForest_chemical, elasticNet_genomic, elasticNet_chemical

def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Data processing pipeline")
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    parser.add_argument('-download', action='store_true', help="Starting download")
    parser.add_argument('-harmonize', action='store_true', help="Data harmonization")
    parser.add_argument('-chemical_features', action='store_true', help='Adding chemical features')
    parser.add_argument('-randomforest', action='store_true', help='Evaluate Random Forest baseline model')
    parser.add_argument('-elasticnet', action='store_true', help='Evaluate Elastic Net baseline model')

    args = parser.parse_args()

    if args.download:
        print("Step 1: Downloading dataset...")
        download_dataset()

    if args.harmonize:
        print("Step 2: Harmonizing dataset...")
        master_df = harmonize_dataset(output_dir=data_dir)

    if args.chemical_features:
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
    
    if args.randomforest:
        df = get_harmonized_dataset()
        randomForest_genomic(df=df, target='LN_IC50')
        randomForest_chemical(df=df, target='LN_IC50')
    
    if args.elasticnet:
        df = get_harmonized_dataset()
        elasticNet_genomic(df=df, target='LN_IC50')
        elasticNet_chemical(df=df, target='LN_IC50')
    
    if len(sys.argv) == 1:
        print('No action has been selected.')
        print("Use 'python __main__.py -h' for further help.")
    
if __name__ == "__main__":
    main()