import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import re
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

def download_dataset():
    # create data directory if necessary
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)

    # get drug sensitivity dataset from GDSC2 if necessary
    url = "https://cmp.cog.sanger.ac.uk/download/GDSC2_fitted_dose_response_27Oct23.xlsx"
    file_path = data_dir / "GDSC2_fitted_dose_response_27Oct23.xlsx"
    if file_path.exists():
        print(f"{file_path.name} already exists. Skipping download.")
    else:
        print(f"Downloading {file_path.name}...")
        urlretrieve(url, file_path)
        print("Download complete.")

    # get expression profiles from DepMap if necessary
    url = "https://depmap.org/portal/data_page/?tab=allData&releasename=DepMap%20Public%2026Q1&filename=OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    file_path = data_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    if file_path.exists():
        print(f"{file_path.name} already exists. Skipping download.")
    else:
        raise FileNotFoundError(
            f"{file_path.name} not found. Please download it on "
            f"{url} and save it in /data."
        )

    # get meta data from expression data
    url = "https://depmap.org/portal/data_page/?tab=allData&releasename=DepMap%20Public%2025Q3&filename=Model.csv"
    file_path = data_dir / "Model.csv"
    if file_path.exists():
        print(f"{file_path.name} already exists. Skipping download.")
    else:
        raise FileNotFoundError(
                    f"{file_path.name} not found. Please download it on "
                    f"{url} and save it in /data."
                )

    # get L1000 landmark genes
    url = "https://raw.githubusercontent.com/s-juross/projekt_genomforschung/refs/heads/main/data/L1000.txt"
    file_path = data_dir / "L1000.txt"
    if file_path.exists():
        print(f"{file_path.name} already exists. Skipping download.")
    else:
        print(f"Downloading {file_path.name}...")
        urlretrieve(url, file_path)
        print("Download complete.")
        
def get_harmonized_dataset():
    return pd.read_pickle("harmonized_data.pkl")