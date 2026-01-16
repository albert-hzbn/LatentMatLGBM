import os
import numpy as np
from ase.io import read
from tensorflow.keras.models import load_model
import pyvista as pv

def write_chgcar(filename, atoms, charge_density):
    """Write CHGCAR-like file from ASE atoms and 3D charge density array."""
    charge_density = np.real(charge_density)
    if charge_density.ndim != 3:
        raise ValueError("Charge density must be 3D before writing CHGCAR.")
    
    with open(filename, 'w') as f:
        f.write("CHGCAR reconstructed from autoencoder\n")
        f.write("  1.0\n")
        for vec in atoms.get_cell():
            f.write(f"  {vec[0]:18.10f} {vec[1]:18.10f} {vec[2]:18.10f}\n")
        symbols = atoms.get_chemical_symbols()
        unique_symbols = list(dict.fromkeys(symbols))
        counts = [symbols.count(sym) for sym in unique_symbols]
        f.write("  " + " ".join(unique_symbols) + "\n")
        f.write("  " + " ".join(str(c) for c in counts) + "\n")
        f.write("Direct\n")
        for pos in atoms.get_scaled_positions():
            f.write(f"  {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")
        nx, ny, nz = charge_density.shape
        f.write(f"\n  {nx}  {ny}  {nz}\n")
        data_flat = charge_density.flatten(order='F')
        for i in range(0, len(data_flat), 10):
            chunk = data_flat[i:i+10]
            f.write(" ".join(f"{x:14.6E}" for x in chunk) + "\n")
        f.write("\naugmentation factor   1.000000\n")
        f.write("total charge written above\n")

def visualize_charge_density(charge_density, atoms=None):
    """Visualize 3D charge density using PyVista."""
    charge_density = np.real(charge_density)
    nx, ny, nz = charge_density.shape
    grid = pv.ImageData()
    grid.dimensions = np.array([nx, ny, nz]) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    if atoms is not None:
        lattice = atoms.get_cell()
        spacing = lattice.diagonal() / np.array([nx, ny, nz])
        grid.spacing = spacing
    grid.cell_data["charge_density"] = charge_density.flatten(order="F")
    plotter = pv.Plotter()
    plotter.add_volume(grid, cmap="viridis", opacity="sigmoid", shade=True)
    if atoms is not None:
        positions = atoms.get_positions()
        plotter.add_points(positions, color="red", point_size=10, render_points_as_spheres=True)
    plotter.show()

def main():
    decoder_path = "/home/albert/OneDrive/Work/Charge_density_ML/GUI_charge_density/model/6059_decoder_model_128.h5"
    latent_dir = "/home/albert/OneDrive/Work/Charge_density_ML/GUI_charge_density/data/latent_data"
    poscar_dir = "/home/albert/OneDrive/Work/Charge_density_ML/GUI_charge_density/data/structures"
    output_dir = "/home/albert/OneDrive/Work/Charge_density_ML/GUI_charge_density/data/reconstructed_chgcar"
    filename = "AcCdRh2_225"

    latent_path = os.path.join(latent_dir, f"{filename}_latent.npy")
    poscar_path = os.path.join(poscar_dir, f"POSCAR_{filename}")
    output_path = os.path.join(output_dir, f"CHGCAR_{filename}")

    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file not found: {latent_path}")
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"POSCAR not found: {poscar_path}")
    os.makedirs(output_dir, exist_ok=True)

    decoder = load_model(decoder_path, compile=False)
    print("✅ Decoder loaded.")

    latent = np.load(latent_path)
    latent = np.expand_dims(latent, axis=0)  # add batch
    if latent.ndim == 4:  # 3D latent missing channel
        latent = np.expand_dims(latent, axis=-1)

    print(f"Latent shape fed to decoder: {latent.shape}")
    print("Latent min/max/mean/std:", latent.min(), latent.max(), latent.mean(), latent.std())

    # Optional: rescale latent to 0-1 to improve decoder output
    latent_scaled = latent / latent.max()

    charge_density = decoder.predict(latent_scaled, verbose=0)[0]
    if charge_density.ndim == 4 and charge_density.shape[-1] == 1:
        charge_density = charge_density[..., 0]
    print(f"Decoded charge density shape: {charge_density.shape}")

    atoms = read(poscar_path)
    write_chgcar(output_path, atoms, charge_density)
    print("✅ CHGCAR written.")

    visualize_charge_density(charge_density, atoms)

if __name__ == "__main__":
    main()
