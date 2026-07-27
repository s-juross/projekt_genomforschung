from pathlib import Path

from data_download import download_dataset
from data_harmonization import harmonize_dataset
from chemical_feature_addition import fetch_smiles_by_cid, get_mfp, get_pharmacophore_features, smiles_to_morgan_columns


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
    unique_drugs_df = master_df[['DRUG_ID']].drop_duplicates().dropna()
    unique_drugs_df['SMILES'] = unique_drugs_df['DRUG_ID'].apply(fetch_smiles_by_cid)
    master_df = master_df.merge(unique_drugs_df, on='DRUG_ID', how='left')
    missing_smiles = master_df['SMILES'].isna().sum()
    print(f"Process complete. Rows with missing SMILES: {missing_smiles}")

    # Some drugs are duplicated under a different name. If Model ID and SMILES are the same, 
    # we assume it's the same drug-cell line combination and can be averaged.
    print(f"Number of rows with NaN SMILES: {master_df['SMILES'].isna().sum()}, gonna remove them")
    master_df.dropna(subset=['SMILES'], inplace=True)
    print(f"Number of rows before: {len(master_df)}")
    # create a dictionary that contains all columns
    agg_dict = {col: 'first' for col in master_df.columns}
    target_metrics = ['LN_IC50', 'AUC', 'Z_SCORE', 'RMSE']
    for metric in target_metrics:
        if metric in agg_dict:
            agg_dict[metric] = 'mean'
    group_cols = ['ModelID', 'SMILES']
    for col in group_cols:
        if col in agg_dict:
            del agg_dict[col]
    # grouping; it Model ID and SMILES are the same, we assume it's the same drug-cell line combination and can be averaged
    master_df = master_df.groupby(group_cols, as_index=False).agg(agg_dict)
    print(f"Number of rows after: {len(master_df)}")

    print("Morgan fingerprints...")
    master_df['MorganFP'] = master_df['SMILES'].apply(get_mfp)
    print("Pharmacophore features...")
    master_df['PharmacophoreFeatures'] = master_df['SMILES'].apply(get_pharmacophore_features)
    print(f"Done. Processed file saved to: {master_df}")


if __name__ == "__main__":
    main()