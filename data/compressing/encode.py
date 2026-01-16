import numpy as np
import scipy.ndimage
import tensorflow as tf
from pymatgen.io.vasp.outputs import Chgcar

# ----------------------------
# Config
# ----------------------------
INPUT_SHAPE = (128, 128, 128)

model_path = "/home/albert/OneDrive/Work/Charge_density_ML/GUI_charge_density/model"
encoder_path = f"{model_path}/6059_encoder_model_128.h5"

load_path = "/home/albert/OneDrive/Work/Charge_density_ML/GUI_charge_density/data/compressing/input"
save_path = "/home/albert/OneDrive/Work/Charge_density_ML/GUI_charge_density/data/compressing/output"
file_name = "primitive"

input_file = f"{load_path}/{file_name}.npy"   # OR CHGCAR file
input_type = "npy"  # use "chgcar" if loading CHGCAR

# ----------------------------
# Resize function
# ----------------------------
def resize_volume(volume, target_size):
    """Resize a 3D volume to target size using interpolation."""
    zoom_factors = [t / s for t, s in zip(target_size, volume.shape)]
    return scipy.ndimage.zoom(volume, zoom_factors, order=1)

# ----------------------------
# Load charge density
# ----------------------------
def load_charge_density(input_path, input_type="npy"):
    """Load charge density from CHGCAR or NPY file."""
    if input_type.lower() == "chgcar":
        chg = Chgcar.from_file(input_path)
        charge_density_3d = chg.data["total"]
        return charge_density_3d, "CHGCAR"
    elif input_type.lower() == "npy":
        charge_density_3d = np.load(input_path)
        return charge_density_3d, "NPY"
    else:
        raise ValueError("Invalid input_type. Use 'chgcar' or 'npy'.")

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_data(charge_density_3d, target_shape=INPUT_SHAPE):
    """Resize or pad the 3D charge density to the target shape."""
    if any(s > t for s, t in zip(charge_density_3d.shape, target_shape)):
        data = resize_volume(charge_density_3d, target_shape)
        method = "Resized"
    else:
        pad_width = [(0, max(0, target_shape[i] - charge_density_3d.shape[i])) for i in range(3)]
        data = np.pad(charge_density_3d, pad_width, mode="constant")
        method = "Padded"
    return data, method

# ----------------------------
# Main Execution
# ----------------------------
print("🔹 Loading charge density...")
charge_density_3d, source_type = load_charge_density(input_file, input_type)

print(f"✅ Loaded from: {source_type}")
print("Original Shape:", charge_density_3d.shape)

print("🔹 Preprocessing...")
data, method = preprocess_data(charge_density_3d)
print(f"✅ Preprocessing Method: {method}")
print("Processed Shape:", data.shape)

# Expand dimensions for model
x = np.expand_dims(data, axis=-1)   # (128,128,128,1)
x = np.expand_dims(x, axis=0)       # (1,128,128,128,1)

# ----------------------------
# Load encoder model
# ----------------------------
print("🔹 Loading encoder model...")
encoder = tf.keras.models.load_model(encoder_path)
print(encoder.summary())

# ----------------------------
# Generate latent vectors
# ----------------------------
print("🔹 Generating latent vectors...")
latent_vectors = encoder.predict(x)

# ----------------------------
# Save latent vectors
# ----------------------------
latent_save_file = f"{save_path}/{file_name}_latent.npy"
np.save(latent_save_file, latent_vectors)

print("✅ Latent vectors saved at:")
print(latent_save_file)
print("✅ Latent Shape:", latent_vectors.shape)
