import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate, KFold, GroupShuffleSplit
import warnings
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer

def randomForest_genomic(df, target):
    ''' 
    Function to train a Random Forest Regressor on genomic features and evaluate its performance.
    Parameters:
    - df: DataFrame containing the data.
    - target: The target variable for prediction.
    - multi: Boolean indicating whether to perform multi-drug analysis.
    - single: Boolean indicating whether to perform single-drug analysis.'''
    print(f"\n--- Genomic Baseline for Target: {target} ---")
    # training set for genomic features
    X_genomic = df.filter(regex=r'.* \(.*\)').values # L1000 landmark genes
    y_genomic = df[target].values # y: drug response
    # train-test split
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_genomic, y_genomic, groups=df['ModelID'].values))
    X_train, X_test = X_genomic[train_idx], X_genomic[test_idx]
    y_train, y_test = y_genomic[train_idx], y_genomic[test_idx]
    groups = df.iloc[train_idx]['ModelID'].values # to ensure held-out validation

    # initialize the Random Forest Regressor
    rf_genomic = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )

    # perform Cross-Validation
    print("Starting Cross-Validation on Training Data...")
    cv_results = cross_validate(
        rf_genomic, X_train, y_train, 
        groups=groups, 
        cv=GroupKFold(n_splits=5),
        scoring=['neg_mean_squared_error', 'r2'],
        return_train_score=True
    )

    # output Results
    mse_scores = -cv_results['test_neg_mean_squared_error']
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = cv_results['test_r2']

    print(f"--- Genomic Baseline Performance ---")
    print(f"R² Score: {np.mean(r2_scores):.4f}")
    print(f"RMSE:     {np.mean(rmse_scores):.4f}")
    print(f"------------------------------------")

    # test fit
    rf_genomic.fit(X_train, y_train)
    y_pred = rf_genomic.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test RMSE:     {test_rmse:.4f}")
        
    importances = pd.Series(rf_genomic.feature_importances_, index=df.filter(regex=r'.* \(.*\)').columns.values)
    print("\nTop 5 Genetic Features:")
    print(importances.sort_values(ascending=False).head(5))
        
def randomForest_chemical(df, target):
    '''
    Function to train a Random Forest Regressor on chemical features and evaluate its performance.
    Parameters:
    - df: DataFrame containing the data.
    - target: The target variable for prediction.'''
    # remove duplicate rows
    # not sure if this is the right approach, but it can help to reduce noise in the data and speed up training
    df.drop_duplicates(subset=['SMILES', target], inplace=True)
    # get MFP in separate columns
    if 'Bit_0' not in df.columns:
        # get MorganFP as separate columns
        fp_df = pd.DataFrame(df['MorganFP'].tolist(), index=df.index)
        fp_df.columns = [f'Bit_{i}' for i in range(fp_df.shape[1])]
        df = pd.concat([df, fp_df], axis=1)
    # get pharmacophore features as separate columns
    if 'Donor' not in df.columns:
        expanded_features = pd.DataFrame(df['PharmacophoreFeatures'].tolist())
        df = pd.concat([df.drop('PharmacophoreFeatures', axis=1), expanded_features], axis=1)
        df = df.fillna(0)
    # training set for chemical features
    X_chem = pd.concat([df.loc[:, ['Donor', 'Acceptor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'PosIonizable', 'NegIonizable', 'ZnBinder']],
                        df.loc[:, df.columns.str.startswith('Bit_')]], axis=1)
    X_cols = X_chem.columns.tolist()
    X = X_chem.values.astype(float)
    y = df[target].values
    # Train-Test-Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['DRUG_ID']))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        max_features='sqrt'
        )

    # Use of GroupKFold makes sure that no data leakage is happening in CV
    print("Starting Cross-Validation on Training Data...")
    cv_results = cross_validate(
        rf_model, X_train, y_train, 
        cv=GroupKFold(n_splits=5),
        groups=df.iloc[train_idx]['DRUG_ID'].values,
        scoring=['r2', 'neg_mean_squared_error'],
        return_train_score=False
    )

    mean_cv_r2 = np.mean(cv_results['test_r2'])
    mean_cv_mse = -np.mean(cv_results['test_neg_mean_squared_error'])
    mean_cv_rmse = np.sqrt(mean_cv_mse)

    print(f"\n--- Cross-Validation Results (Train Data) ---")
    print(f"R² Score: {mean_cv_r2:.4f}")
    print(f"RMSE: {mean_cv_rmse:.4f}")

    # Build a forest of trees from the training set (X, y)
    rf_model.fit(X_train, y_train)

    # evaluation on unknown data set
    y_pred = rf_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n--- Test Set Results ---")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test RMSE: {test_mse:.4f}")

    # feature importance
    importances = pd.Series(rf_model.feature_importances_, index=X_cols)
    print("\nTop 5 Chemical Drivers for Drug Potency:")
    print(importances.sort_values(ascending=False).head(5))
    
def elasticNet_genomic(df, target):
    '''
    Function to train an Elastic Net model on genomic features and evaluate its performance.
    Parameters:
    - df: DataFrame containing the data.
    - target: The target variable for prediction.
    '''
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # ensures that all rows from one cell line stay together in the train/test split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['ModelID']))

    X_train = df.loc[df.index[train_idx], df.columns[df.columns.str.contains(r'.* \(.*\)')]].values.astype('float32')
    y_train = df.loc[df.index[train_idx], target].values.astype('float32')
    groups_train = df.loc[df.index[train_idx], 'ModelID'].values

    X_test = df.loc[df.index[test_idx], df.columns[df.columns.str.contains(r'.* \(.*\)')]].values.astype('float32')
    y_test = df.loc[df.index[test_idx], target].values.astype('float32')

    print(f"Number of unique cell lines in training set: {len(np.unique(groups_train))}")
    print(f"Number of unique cell lines in test set: {len(np.unique(df.loc[df.index[test_idx], 'ModelID'].values))}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNetCV(
            l1_ratio=[0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.99, 1], # model will find the best mix
            cv=GroupKFold(n_splits=5).split(X_train, y_train, groups=groups_train),
            random_state=42,
            max_iter=8000,
            alphas=20,
            tol=1e-3,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    fitted_model = pipeline.named_steps['model']
    best_alpha_idx = np.where(fitted_model.alphas_ == fitted_model.alpha_)[0][0]
    mean_mse_best_alpha = np.mean(fitted_model.mse_path_[best_alpha_idx])
    variance_y_train = np.var(y_train)
    train_cv_r2 = 1 - (mean_mse_best_alpha / variance_y_train)

    y_pred = pipeline.predict(X_test)
    test_r2 = pipeline.score(X_test, y_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "="*40)
    print("   ELASTIC NET CV PIPELINE RESULTS")
    print("="*40)
    print(f"Chosen Alpha:           {fitted_model.alpha_:.6f}")
    print(f"Chosen L1-Ratio:        {fitted_model.l1_ratio_:.2f}")
    print("-"*40)
    print(f"Internal CV training R²:    {train_cv_r2:.4f}")
    print(f"Test R² Score:       {test_r2:.4f}")
    print(f"Test RMSE:     {test_rmse:.4f}")
    print("="*40)

    # analyze feature importance
    final_model = pipeline.named_steps['model']
    coefs = final_model.coef_

    # Get the gene names from your original columns
    gene_names = df.loc[df.index[train_idx], df.columns[df.columns.str.contains(r'.* \(.*\)')]].columns

    # Create a summary table
    features_df = pd.DataFrame({'Gene': gene_names, 'Coefficient': coefs})
    features_df['Abs_Coef'] = features_df['Coefficient'].abs()

    # Filter for genes the model didn't set to zero
    selected_genes = features_df[features_df['Coefficient'] != 0]

    print(f"\nElastic Net selected {len(selected_genes)} genes out of 978.")
    print(f"Top 5 Positive Biomarkers (Increase {target}):")
    print(features_df.sort_values(by='Coefficient', ascending=False).head(5))
    print(f"\nTop 5 Negative Biomarkers (Decrease {target}):")
    print(features_df.sort_values(by='Coefficient', ascending=True).head(5))

def elasticNet_chemical(df, target):
    '''
    Function to train an Elastic Net model on chemical features and evaluate its performance.
    Parameters:
    - df: DataFrame containing the data.
    - target: The target variable for prediction.
    '''
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    chem_cols = ('Donor', 'Acceptor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'PosIonizable', 'NegIonizable', 'ZnBinder', 'Bit_')

    # get MFP in separate columns
    if 'Bit_0' not in df.columns:
        # get MorganFP as separate columns
        fp_df = pd.DataFrame(df['MorganFP'].tolist(), index=df.index)
        fp_df.columns = [f'Bit_{i}' for i in range(fp_df.shape[1])]
        df = pd.concat([df, fp_df], axis=1)
    # get pharmacophore features as separate columns
    if 'Donor' not in df.columns:
        expanded_features = pd.DataFrame(df['PharmacophoreFeatures'].tolist(), index=df.index)
        df = pd.concat([df.drop('PharmacophoreFeatures', axis=1), expanded_features], axis=1)
        df = df.fillna(0)
        
    # ensures that all rows from one cell line stay together in the train/test split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['DRUG_ID']))

    X_train = df.loc[df.index[train_idx], df.columns[df.columns.str.startswith(chem_cols)]]#.values.astype('float32')
    y_train = df.loc[df.index[train_idx], target].values.astype('float32')
    groups_train = df.loc[df.index[train_idx], 'DRUG_ID'].values

    X_test = df.loc[df.index[test_idx], df.columns[df.columns.str.startswith(chem_cols)]]#.values.astype('float32')
    y_test = df.loc[df.index[test_idx], target].values.astype('float32')

    print(f"Number of unique drugs in train set: {len(np.unique(groups_train))}")
    print(f"Number of unique drugs in test set: {len(np.unique(df.loc[df.index[test_idx], 'DRUG_ID'].values))}")

    # scale pharmacophore features
    pharmacophore_cols = ['Donor', 'Acceptor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'PosIonizable', 'NegIonizable', 'ZnBinder']
    mfp_cols = [col for col in df.columns if col.startswith('Bit_')]
    preprocessor = ColumnTransformer(
        transformers=[
            ('pharmacophore', StandardScaler(), pharmacophore_cols),
            ('mfp', 'passthrough', mfp_cols)
        ],
        remainder='drop'
    )
    # Use ColumnTransformer to scale the pharmacophore features and ElasticNetCV for regression with cross-validation
    pipeline = Pipeline([
        ('scaler', preprocessor),
        ('model', ElasticNetCV(
            l1_ratio=[0.05, 0.1, 0.5, 0.7, 0.9, 0.99, 1], # model will find the best mix
            cv=GroupKFold(n_splits=5).split(X_train, y_train, groups=groups_train),
            random_state=42,
            max_iter=10000,
            alphas=20,
            tol=1e-3,
            n_jobs=1 # Keeping it to 1 to avoid memory issues
        ))
    ])

    print("\nFitting final model on entire training set...")
    pipeline.fit(X_train, y_train)
    # neuer versuch
    fitted_model = pipeline.named_steps['model']
    best_alpha_idx = np.where(fitted_model.alphas_ == fitted_model.alpha_)[0][0]
    mean_mse_best_alpha = np.mean(fitted_model.mse_path_[best_alpha_idx])
    variance_y_train = np.var(y_train)
    train_cv_r2 = 1 - (mean_mse_best_alpha / variance_y_train)
    
    y_pred = pipeline.predict(X_test)
    if target == 'AUC':
        y_pred = np.clip(y_pred, 0, 1)  # Clip predictions to the range [0, 1] for AUC
    test_r2 = pipeline.score(X_test, y_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n" + "="*40)
    print("   ELASTIC NET CV PIPELINE ERGEBNISSE")
    print("="*40)
    print(f"Alpha:           {fitted_model.alpha_:.6f}")
    print(f"L1-Ratio:        {fitted_model.l1_ratio_:.2f}")
    print("-"*40)
    print(f"Internal CV Training R²:     {train_cv_r2:.4f}")
    print("="*40)

    print(f"\n--- Held-Out Test Results ---")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test RMSE:     {test_rmse:.4f}")

    final_model = pipeline.named_steps['model']
    coefs = final_model.coef_

    # Get the molecule structure names from your original columns
    bit_names = df.columns[df.columns.str.startswith(chem_cols)]
    # Create a summary table
    features_df = pd.DataFrame({'Molecule_Structure': bit_names, 'Coefficient': coefs})
    features_df['Abs_Coef'] = features_df['Coefficient'].abs()
    # Filter for genes the model didn't set to zero
    selected_struct = features_df[features_df['Coefficient'] != 0]

    print(f"\nElastic Net selected {len(selected_struct)} molecule structures out of 1032.")
    print(f"Top 5 Positive molecule structures (Increase {target}):")
    print(features_df.sort_values(by='Coefficient', ascending=False).head(5))
    print(f"\nTop 5 Negative molecule structures (Decrease {target}):")
    print(features_df.sort_values(by='Coefficient', ascending=True).head(5))