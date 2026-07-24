import pandas as pd
import re

def harmonize_dataset(output_dir):
    df = pd.read_excel("data/GDSC2_fitted_dose_response_27Oct23.xlsx")
    ge = pd.read_csv("data/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
    metadata = pd.read_csv("data/Model.csv", usecols=['ModelID', 'COSMICID', 'SangerModelID', 'ModelIDAlias'])
    metadata.rename(columns={'SangerModelID': 'SANGER_MODEL_ID'}, inplace=True)
    L1000 = pd.read_csv("data/L1000.txt")

    # shrinking the dataset to the L1000 landmark genes
    ge_col2id = {}
    for col in ge.columns: 
        match = re.search(r"\((\d+)\)\s*$", col) # Extract Entrez IDs from ge column names like "TSPAN6 (7105)"
        if match:
            ge_col2id[col] = int(match.group(1))

    # Normalize L1000 IDs to integers and filter
    L1000_ids = set(pd.to_numeric(L1000['ID'], errors='coerce').dropna().astype(int))
    valid_ge_cols = [col for col, entrez in ge_col2id.items() if entrez in L1000_ids]
    # New dataframe containing only matching columns
    ge_matched = ge[['SequencingID', 'ModelID', 'IsDefaultEntryForModel', 'ModelConditionID', 'IsDefaultEntryForMC'] + valid_ge_cols].copy()
    print(f"Selected {len(valid_ge_cols)} matching columns out of {len(ge.columns)} total ge columns and saving it in ge_matched.")
    ge_matched.head()

    # Creating meta categories for cancer types so that the plots get more understandable
    category_map = {
        # Hematologic (Blood & Lymph)
        'T-Lymphoblastic Leukemia': 'Blood/Lymph',
        'Acute Myeloid Leukemia': 'Blood/Lymph',
        'Chronic Myelogenous Leukemia': 'Blood/Lymph',
        'B-Lymphoblastic Leukemia': 'Blood/Lymph',
        'Plasma Cell Myeloma': 'Blood/Lymph',
        "T-Cell Non-Hodgkin's Lymphoma": 'Blood/Lymph',
        "B-Cell Non-Hodgkin's Lymphoma": 'Blood/Lymph',
        "Burkitt's Lymphoma": 'Blood/Lymph',
        "Hodgkin's Lymphoma": 'Blood/Lymph',
        'Other Blood Cancers': 'Blood/Lymph',

        # Gastrointestinal (GI Tract)
        'Colorectal Carcinoma': 'Gastrointestinal',
        'Gastric Carcinoma': 'Gastrointestinal',
        'Pancreatic Carcinoma': 'Gastrointestinal',
        'Hepatocellular Carcinoma': 'Gastrointestinal',
        'Biliary Tract Carcinoma': 'Gastrointestinal',
        'Esophageal Carcinoma': 'Gastrointestinal',
        'Esophageal Squamous Cell Carcinoma': 'Gastrointestinal',

        # Thoracic (Lung & Chest)
        'Non-Small Cell Lung Carcinoma': 'Thoracic',
        'Squamous Cell Lung Carcinoma': 'Thoracic',
        'Small Cell Lung Carcinoma': 'Thoracic',
        'Mesothelioma': 'Thoracic',

        # Sarcomas (Bone & Soft Tissue)
        "Ewing's Sarcoma": 'Sarcoma/Bone',
        'Osteosarcoma': 'Sarcoma/Bone',
        'Chondrosarcoma': 'Sarcoma/Bone',
        'Rhabdomyosarcoma': 'Sarcoma/Bone',
        'Other Sarcomas': 'Sarcoma/Bone',

        # CNS (Brain)
        'Glioblastoma': 'CNS',
        'Glioma': 'CNS',
        'Neuroblastoma': 'CNS', # Oft PNS, aber meist mit ZNS gruppiert

        # Gynecologic / Urologic
        'Ovarian Carcinoma': 'Uro/Gyn',
        'Cervical Carcinoma': 'Uro/Gyn',
        'Endometrial Carcinoma': 'Uro/Gyn',
        'Breast Carcinoma': 'Uro/Gyn',
        'Prostate Carcinoma': 'Uro/Gyn',
        'Bladder Carcinoma': 'Uro/Gyn',
        'Kidney Carcinoma': 'Uro/Gyn',

        # Head & Neck / Skin / Endocrine
        'Melanoma': 'Skin/HNSC',
        'Head and Neck Carcinoma': 'Skin/HNSC',
        'Oral Cavity Carcinoma': 'Skin/HNSC',
        'Thyroid Gland Carcinoma': 'Skin/HNSC',

        # Others
        'Non-Cancerous': 'Control',
        'Other Solid Cancers': 'Other'
    }
    # Anwendung auf den DataFrame
    df['META_CANCERTYPE'] = df['CANCER_TYPE'].map(category_map).fillna('Other')
    ge_matched['META_CANCERTYPE'] = ge_matched['CANCER_TYPE'].map(category_map).fillna('Other')

    # big merge with complete drug data
    # delete columns that we don't use
    ge_matched = ge_matched.drop(columns=['IsDefaultEntryForModel', 'ModelConditionID', 'IsDefaultEntryForMC'])
    # filter for non-null SANGER_MODEL_ID and select only necessary columns for the merge
    mapping_bridge = metadata[['ModelID', 'SANGER_MODEL_ID']].dropna(subset=['SANGER_MODEL_ID'])

    # merge with left join to keep all rows in ge_matched and add SANGER_MODEL_ID where available
    if 'SANGER_MODEL_ID' in ge_matched.columns:
        ge_matched = ge_matched.drop(columns=['SANGER_MODEL_ID'])
    ge_matched = pd.merge(ge_matched, mapping_bridge, on='ModelID', how='left')

    # include cancer type information by merging with the original dataframe
    cancer_info = df[['SANGER_MODEL_ID', 'CANCER_TYPE']].drop_duplicates()
    ge_matched = pd.merge(ge_matched, cancer_info, on='SANGER_MODEL_ID', how='left')
    if 'CANCER_TYPE' in ge_matched.columns:
        ge_matched['CANCER_TYPE'] = ge_matched['CANCER_TYPE'].fillna('Unknown')
        print("CANCER_TYPE was added successfully.")
    else:
        ge_matched['CANCER_TYPE'] = 'Unknown'
        print("No matches found; CANCER_TYPE set to 'Unknown'.")

    # Selection of relevant columns from drug dataframe (df): need id to merge + target values (labels)
    drug_data_cols = [
        'SANGER_MODEL_ID', 'DRUG_ID', 'DRUG_NAME', 
        'LN_IC50', 'Z_SCORE', 'RMSE', 'AUC'
    ]
    df_subset = df[drug_data_cols]

    # create master df which contains the merged dataframes
    # merge, only keep rows where we have both gene expression data AND drug response data
    master_df = pd.merge(ge_matched, df_subset, on='SANGER_MODEL_ID', how='inner')

    # check results
    print(f"Dimension of the new master df: {master_df.shape}")
    print(f"Unique drugs: {master_df['DRUG_NAME'].nunique()}")
    print(f"Unique cell lines: {master_df['SANGER_MODEL_ID'].nunique()}")

    # save
    master_df.to_pickle("harmonized_data.pkl", path=output_dir)
    # load
    #master_df = pd.read_pickle("data/harmonized_data.pkl", )
    return master_df