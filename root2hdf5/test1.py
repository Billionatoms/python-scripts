# %%
import uproot
import h5py
import numpy as np
import awkward as ak  # Awkward Array library for handling jagged data

# %%
def convert_lctuple_to_h5(root_file, tree_name, output_file):
    # Open the ROOT file and retrieve the tree with uproot
    with uproot.open(root_file) as file:
        tree = file[tree_name]

        # Retrieve arrays as awkward arrays for jagged data handling
        jmox = tree["jmox"].array()
        jmoy = tree["jmoy"].array()
        jmoz = tree["jmoz"].array()
        jmas = tree["jmas"].array()
        jene = tree["jene"].array()

        # Truth level information
        mcgst = tree["mcgst"].array()
        mcpdg = tree["mcpdg"].array()
        mcmox = tree["mcmox"].array()
        mcmoy = tree["mcmoy"].array()
        mcmoz = tree["mcmoz"].array()
        mcmas = tree["mcmas"].array()
        mcene = tree["mcene"].array()

        # Calculate pT, phi, and eta values for jets
        jpt = ak.sqrt(jmox**2 + jmoy**2)
        jphi = ak.arctan2(jmoy, jmox)
        jmo = ak.sqrt(jmox**2 + jmoy**2 + jmoz**2)
        epsilon = 1e-10  # A small value to prevent division by zero
        jeta = 0.5 * ak.log((jmo + jmoz) / (jmo - jmoz + epsilon))

        # Calculate pT, phi, and eta values for truth-level quarks
        mcpt = ak.sqrt(mcmox**2 + mcmoy**2)
        mcphi = ak.arctan2(mcmoy, mcmox)
        mcmo = ak.sqrt(mcmox**2 + mcmoy**2 + mcmoz**2)
        mceta = 0.5 * ak.log((mcmo + mcmoz) / (mcmo - mcmoz + epsilon))

        # Convert all arrays to NumPy arrays, using to_numpy for jagged data
        jpt_np = ak.to_numpy(jpt, allow_missing=True)
        jphi_np = ak.to_numpy(jphi, allow_missing=True)
        jeta_np = ak.to_numpy(jeta, allow_missing=True)
        jmas_np = ak.to_numpy(jmas, allow_missing=True)

        mcpt_np = ak.to_numpy(mcpt, allow_missing=True)
        mcphi_np = ak.to_numpy(mcphi, allow_missing=True)
        mceta_np = ak.to_numpy(mceta, allow_missing=True)

        # Write the arrays to the HDF5 file
        with h5py.File(output_file, "w") as h5file:
            # Create datasets with jagged array structure
            h5file.create_dataset("jpt", data=jpt_np)
            h5file.create_dataset("jphi", data=jphi_np)
            h5file.create_dataset("jeta", data=jeta_np)
            h5file.create_dataset("jmas", data=jmas_np)
            h5file.create_dataset("mcpt", data=mcpt_np)
            h5file.create_dataset("mcphi", data=mcphi_np)
            h5file.create_dataset("mceta", data=mceta_np)

# %%
# Usage example
root_file = "/eos/user/s/ssaini/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root"  # Path to your input LCTuple ROOT file
tree_name = "LCTuple"    # Name of the tree in the ROOT file
output_file = "output.h5" # Desired output HDF5 file name

convert_lctuple_to_h5(root_file, tree_name, output_file)

# %%
