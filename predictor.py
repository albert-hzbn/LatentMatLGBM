
from pymatgen.io.vasp.outputs import Chgcar
from tensorflow.keras.models import load_model
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import pandas as pd
import joblib
import warnings
import os
import re

# --- NEW IMPORTS FOR FEATURE GENERATION ---
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

# --- 0. SETUP ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


# ----------------------------
# Config
# ----------------------------
INPUT_SHAPE = (128, 128, 128)


def resize_volume(volume, target_size):
    """Resize a 3D volume to target size using interpolation."""
    zoom_factors = [t / s for t, s in zip(target_size, volume.shape)]
    return scipy.ndimage.zoom(volume, zoom_factors, order=1)


def load_charge_density(chgcar_path):
    chg = Chgcar.from_file(chgcar_path)
    charge_density_3d = chg.data["total"]
    return charge_density_3d, "CHGCAR"

def preprocess_data(charge_density_3d, target_shape=INPUT_SHAPE):
    """Resize or pad the 3D charge density to the target shape."""
    if any(s > m for s, m in zip(charge_density_3d.shape, target_shape)):
        data = resize_volume(charge_density_3d, target_shape)
        method = "Resized"
    else:
        pad_width = [(0, max(0, target_shape[i] - charge_density_3d.shape[i])) for i in range(3)]
        data = np.pad(charge_density_3d, pad_width, mode="constant")
        method = "Padded"
    return data, method



def convert_to_formula_spacegroup(chgcar_path, symprec=1e-2):
    """Convert a CHGCAR file to 'formula_spacegroup' string."""
    
    # Read structure from CHGCAR file
    structure = Structure.from_file(chgcar_path)

    # Reduced chemical formula
    formula = structure.composition.reduced_formula

    # Determine space group
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    sg_number = sga.get_space_group_number()

    return f"{formula}_{sg_number}"



def generate_missing_magpie_features(formula_input, magpie_csv_path, assets_df):
    """
    Generates Magpie features for a missing formula, appends to CSV, 
    and returns the single row DataFrame.
    """
    print(f"   [Info] Magpie features missing for '{formula_input}'. Generating now...")
    
    # 1. Prepare Data
    # Clean the formula (remove _xxx suffix if present)
    formula_clean = re.sub(r'_\d+$', '', formula_input)
    
    # Create a temporary DataFrame
    temp_df = pd.DataFrame({
        'formula_sp': [formula_input], 
        'formula_clean': [formula_clean]
    })

    try:
        # 2. Convert to Composition
        str_to_comp = StrToComposition(target_col_id='composition')
        df_comp = str_to_comp.featurize_dataframe(temp_df, col_id='formula_clean', ignore_errors=False)
        
        # 3. Generate Features
        ep_feat = ElementProperty.from_preset("magpie")
        # Featurize
        df_features = ep_feat.featurize_dataframe(df_comp, col_id='composition', ignore_errors=False)
        
        # 4. Filter Columns
        # Ensure we only keep columns that match the loaded assets (excluding formula_sp)
        required_cols = [col for col in assets_df.columns if col != 'formula_sp']
        
        # Check if generation failed (sometimes matminer returns NaNs for invalid elements)
        if df_features[required_cols].isnull().values.any():
            print(f"   [Error] Generated features contain NaNs for {formula_clean}.")
            return None

        # Prepare final row
        new_row = df_features[['formula_sp'] + required_cols]

        # 5. Save to CSV (Append mode)
        header = not os.path.exists(magpie_csv_path)
        new_row.to_csv(magpie_csv_path, mode='a', header=header, index=False)
        print(f"   [Success] Features generated and appended to {magpie_csv_path}")

        return new_row

    except Exception as e:
        print(f"   [Error] Failed to generate features: {e}")
        return None
    


def load_prediction_assets_lgbm(type=None):
    """Loads all necessary files (model, scalers, PCA, lookup table)."""
    assets = {}
    try:
        # Load the trained model
        assets['model'] = joblib.load(f'./model/{type}_LGBM.pkl')
        
        # Load the preprocessors (the "keys")
        assets['scaler_3d'] = joblib.load(f'./model/{type}_pca_scaler.pkl')
        assets['pca'] = joblib.load(f'./model/{type}_pca_model.pkl')
        assets['magpie_scaler'] = joblib.load(f'./model/{type}_magpie_scaler.pkl')

        # Load the Magpie feature lookup table
        assets['magpie_csv_path'] = './model/magpie_features.csv' # Store path
        assets['magpie_lookup_df'] = pd.read_csv(assets['magpie_csv_path'])
        assets['magpie_cols'] = [col for col in assets['magpie_lookup_df'].columns if col != 'formula_sp']
        
        print("All LGBM model assets loaded successfully.")
        
    except FileNotFoundError as e:
        print(f"--- FATAL ERROR: Could not load all model assets. ---")
        print(f"Missing file: {e.filename}")
        print("Please ensure all files are in the same directory:")
        print("[To_Publish_Formation_Energy_Best.pkl, scaler_3d.pkl, pca_transformer.pkl, magpie_scaler.pkl, magpie_features.csv]")
        return None
    except Exception as e:
        print(f"An unknown error occurred loading assets: {e}")
        return None
    return assets

def predict_lgbm(formula: str, compressed_chgcar, assets: dict) -> float:
    """
    Predicts the Formation Energy (eV/atom) from a formula and a .npy file.
    """
    if not assets:
        print("Error: Model assets are not loaded.")
        return None
        
    print(f"\n--- Predicting for {formula} ---")
    
    # # === A: PROCESS 3D (PCA) INPUT ===
        
    x_3d_flat = compressed_chgcar.reshape(1, -1)
    x_3d_scaled = assets['scaler_3d'].transform(x_3d_flat)
    x_pca = assets['pca'].transform(x_3d_scaled)

    # === B: PROCESS MAGPIE INPUT ===
    try:
        # 1. Try to find features in loaded DataFrame
        x_1d_features = assets['magpie_lookup_df'][assets['magpie_lookup_df']['formula_sp'] == formula][assets['magpie_cols']]
        
        # 2. If not found, GENERATE THEM
        if x_1d_features.empty:
            new_row = generate_missing_magpie_features(formula, assets['magpie_csv_path'], assets['magpie_lookup_df'])
            
            if new_row is not None:
                # Use the newly generated row
                x_1d_features = new_row[assets['magpie_cols']]
                # Update memory so we don't have to reload CSV
                assets['magpie_lookup_df'] = pd.concat([assets['magpie_lookup_df'], new_row], ignore_index=True)
            else:
                print(f"Error: Could not generate Magpie features for '{formula}'")
                return None
                
    except Exception as e:
        print(f"Error looking up/generating Magpie features: {e}")
        return None
    
    x_1d_scaled = assets['magpie_scaler'].transform(x_1d_features)

    # === C: COMBINE AND PREDICT ===
    try:
        X_combined = np.concatenate([x_pca, x_1d_scaled], axis=1)
        y_pred_real = assets['model'].predict(X_combined)
        
        # This model predicts the real value directly, no y_scaler needed
        final_prediction = y_pred_real[0]
        return final_prediction
        
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None


def predict_property(chgcar_path, data, prop_type, encoder_path):

    # Prepare input
    x = np.expand_dims(data, axis=-1)  # (128,128,128,1)
    x = np.expand_dims(x, axis=0)      # (1,128,128,128,1)

    # Load models
    encoder = load_model(encoder_path)

    # convert loaded model to numPy array
    features = encoder.predict(x)

    # obtain formula_spacegroup
    formula_spacegroup = convert_to_formula_spacegroup(chgcar_path)

    # Load all assets once at the start
    model_assets = load_prediction_assets_lgbm(prop_type)

    if model_assets:

        prediction = predict_lgbm(
            formula=formula_spacegroup,
            compressed_chgcar=features,
            assets=model_assets
        )

        if prediction is not None:
            print(prediction)
            return float(prediction)

        else:
            return 0.0
        