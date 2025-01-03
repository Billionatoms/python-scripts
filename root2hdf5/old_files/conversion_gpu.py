# %%
# Importing the necessary libraries
import uproot
import h5py
import numpy as np
import awkward as ak

# %%
# Step 1: Open the ROOT file and extract data
file_path = "/mnt/c/Users/Saurabh/cernbox/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root"

with uproot.open(file_path) as f:
    tree = f["LCTuple"]
    
    # List of branches to extract
    branches = [
        "evevt",
        # Truth level information
        "nmcp", "mcpdg", "mcgst", "mcmox", "mcmoy", "mcmoz", "mcene",
        # Jet Information
        "njet", "jmox", "jmoy", "jmoz", "jene"
    ]
    
    # Extracting data using Awkward Array
    data = tree.arrays(branches, library="ak")
    
    # Event number
    evevt = data["evevt"]
    
    # Truth level information
    nmcp = data["nmcp"]
    mcpdg = data["mcpdg"]
    mcgst = data["mcgst"]
    mcmox = data["mcmox"]
    mcmoy = data["mcmoy"]
    mcmoz = data["mcmoz"]
    mcene = data["mcene"]
    
    # Jet Information
    njet = data["njet"]
    jmox = data["jmox"]
    jmoy = data["jmoy"]
    jmoz = data["jmoz"]
    jene = data["jene"]

# %%
# Dataset 'jets' definition
dtype_jets = np.dtype([
    ("pt", np.float32),
    ("eta", np.float32),
    ("flavour", np.int32)
])

# %%
# Step 2: Calculate jet and truth particle kinematics using Awkward Arrays

# Calculate jet kinematics
jet_px = jmox
jet_py = jmoy
jet_pz = jmoz
jet_e = jene

jet_pt = np.sqrt(jet_px**2 + jet_py**2)
jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
jet_eta = 0.5 * np.log((jet_p + jet_pz + 1e-8) / (jet_p - jet_pz + 1e-8))
jet_phi = np.arctan2(jet_py, jet_px)

# Calculate truth particle kinematics
mc_px = mcmox
mc_py = mcmoy
mc_pz = mcmoz
mc_e = mcene

mc_pt = np.sqrt(mc_px**2 + mc_py**2)
mc_p = np.sqrt(mc_px**2 + mc_py**2 + mc_pz**2)
mc_eta = 0.5 * np.log((mc_p + mc_pz + 1e-8) / (mc_p - mc_pz + 1e-8))
mc_phi = np.arctan2(mc_py, mc_px)

# %%
# Step 3: Define delta R function

def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # Normalize between -π and π
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

# %%
# Step 4: Match truth-level quarks to jets based on proximity in eta-phi space

# Label mapping and class names
label_map = {0: 0, 4: 1, 5: 2}
class_names = ["ujets", "cjets", "bjets"]

# Initialize lists for storing matched jets
matched_jet_pt = []
matched_jet_eta = []
matched_jet_flavour = []

# Iterate over events
num_events = len(evevt)

for i in range(num_events):
    # Get jet kinematics for the event
    jets_pt = jet_pt[i]
    jets_eta = jet_eta[i]
    jets_phi = jet_phi[i]

    if len(jets_pt) == 0:
        continue  # Skip if no jets in the event

    # Get truth particles for the event
    mc_gst_event = mcgst[i]
    mc_pdg_event = mcpdg[i]
    mc_eta_event = mc_eta[i]
    mc_phi_event = mc_phi[i]

    # Apply the mask to the event-specific data
    quark_mask = (mc_gst_event != 0) & (np.abs(mc_pdg_event) <= 5)

    quark_eta = mc_eta_event[quark_mask]
    quark_phi = mc_phi_event[quark_mask]
    quark_pdg = mc_pdg_event[quark_mask]

    if len(quark_eta) == 0:
        continue  # Skip if no quarks

    # Expand arrays to compute delta R
    jets_eta_exp, quark_eta_exp = ak.broadcast_arrays(jets_eta[:, np.newaxis], quark_eta)
    jets_phi_exp, quark_phi_exp = ak.broadcast_arrays(jets_phi[:, np.newaxis], quark_phi)

    # Calculate delta R between jets and quarks
    dRs = delta_r(jets_eta_exp, jets_phi_exp, quark_eta_exp, quark_phi_exp)

    # Find the jet with minimum delta R for each quark
    min_dR_indices = ak.argmin(dRs, axis=0)
    min_dR_jets_pt = jets_pt[min_dR_indices]
    min_dR_jets_eta = jets_eta[min_dR_indices]
    min_dR_quark_pdg = quark_pdg

    # Assign flavour labels
    jet_flavour = ak.full_like(min_dR_quark_pdg, 0)
    jet_flavour = ak.where(np.abs(min_dR_quark_pdg) == 5, 2, jet_flavour)
    jet_flavour = ak.where(np.abs(min_dR_quark_pdg) == 4, 1, jet_flavour)

    # Append matched jets to lists
    matched_jet_pt.extend(ak.to_list(min_dR_jets_pt))
    matched_jet_eta.extend(ak.to_list(min_dR_jets_eta))
    matched_jet_flavour.extend(ak.to_list(jet_flavour))

# %%
# Step 5: Create structured arrays and define dataset shapes

# Create structured array for jets
jets_data = np.zeros(len(matched_jet_pt), dtype=dtype_jets)
jets_data['pt'] = matched_jet_pt
jets_data['eta'] = matched_jet_eta
jets_data['flavour'] = matched_jet_flavour

# %%
# Step 6: Create the HDF5 file and datasets

output_file = "/home/ssaini/dev/muonc/btagging/output_data/output_with_awkward.h5"

with h5py.File(output_file, "w") as f:
    # Create 'jets' dataset with LZF compression
    dataset_jets = f.create_dataset(
        "jets",
        data=jets_data,
        dtype=dtype_jets,
        compression="lzf"
    )
    dataset_jets.attrs["flavour_label"] = np.array(class_names, dtype="S")

print(f"Conversion complete. Data saved as a .h5 file at {output_file}")
# %%
