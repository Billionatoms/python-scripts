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
        "njet", "jmox", "jmoy", "jmoz", "jene",
        # Track Information
        "ntrk", "tsdze", "tsphi", "tsome", "tszze", "tstnl"
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
    
    # Track Information
    ntrk = data["ntrk"]
    tsdze = data["tsdze"]   # d0 parameter
    tsphi = data["tsphi"]   # phi parameter
    tsome = data["tsome"]   # curvature Ω
    tszze = data["tszze"]   # z0 parameter
    tstnl = data["tstnl"]   # tanλ

# %%
# Dataset 'jets' definition
dtype_jets = np.dtype([
    ("pt", np.float32),
    ("eta", np.float32),
    ("flavour", np.int32)
])

# Dataset 'tracks' definition
dtype_tracks = np.dtype([
    ("jet_idx", np.int32),
    ("pt", np.float32),
    ("eta", np.float32),
    ("phi", np.float32),
    ("d0", np.float32),
    ("z0", np.float32),
    ("charge", np.int32)
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
# Step 3: Calculate track kinematics

# Function to calculate track charge from curvature
def track_charge(curvature):
    return np.sign(curvature)

# Function to calculate track momentum from curvature and tan(lambda)
def track_p(curvature, tan_lambda, magnetic_field=3.57):
    return np.abs(0.3 * magnetic_field / curvature) / np.cos(np.arctan(tan_lambda))

# Function to calculate transverse momentum (pt) from total momentum (p) and pitch angle (lambda)
def track_pT(momentum, tan_lambda):
    angle = np.arctan(tan_lambda)
    return momentum * np.sin(angle)

# Function to calculate eta from tan(lambda)
def track_eta(tan_lambda):
    angle = np.arctan(tan_lambda)
    return -np.log(np.tan(angle / 2))

# Calculate track kinematics
trk_charge = track_charge(tsome)
trk_momentum = track_p(tsome, tstnl)
trk_pt = track_pT(trk_momentum, tstnl)
trk_eta = track_eta(tstnl)
trk_phi = tsphi
trk_d0 = tsdze
trk_z0 = tszze

# %%
# Step 4: Define delta R function

def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # Normalize between -π and π
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

# %%
# Step 5: Match truth-level quarks to jets based on proximity in eta-phi space

# Label mapping and class names
label_map = {0: 0, 4: 1, 5: 2}
class_names = ["ujets", "cjets", "bjets"]

# Initialize lists for storing matched jets
matched_jet_pt = []
matched_jet_eta = []
matched_jet_flavour = []

# Initialize list for matched tracks
matched_tracks = []

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

    # Match tracks to jets
    # Get track kinematics for the event
    tracks_pt = trk_pt[i]
    tracks_eta = trk_eta[i]
    tracks_phi = trk_phi[i]
    tracks_d0 = trk_d0[i]
    tracks_z0 = trk_z0[i]
    tracks_charge = trk_charge[i]

    if len(tracks_pt) == 0:
        continue  # Skip if no tracks in the event

    # Expand arrays to compute delta R between tracks and jets
    tracks_eta_exp, jets_eta_exp = ak.broadcast_arrays(tracks_eta[:, np.newaxis], jets_eta)
    tracks_phi_exp, jets_phi_exp = ak.broadcast_arrays(tracks_phi[:, np.newaxis], jets_phi)

    # Calculate delta R between tracks and jets
    dRs_tracks_jets = delta_r(tracks_eta_exp, tracks_phi_exp, jets_eta_exp, jets_phi_exp)

    # Find the jet with minimum delta R for each track
    min_dR_track_indices = ak.argmin(dRs_tracks_jets, axis=1)
    
    # **Convert Awkward Arrays to NumPy Arrays for Indexing**
    min_dR_track_indices_np = ak.to_numpy(min_dR_track_indices)
    dRs_tracks_jets_np = ak.to_numpy(dRs_tracks_jets)
    
    # Perform advanced indexing using NumPy arrays
    min_dR_track_values = dRs_tracks_jets_np[np.arange(len(tracks_pt)), min_dR_track_indices_np]

    # Optionally, set a maximum delta R to consider a track as matched to a jet
    max_delta_r = 0.4
    matched_mask = min_dR_track_values < max_delta_r

    # Extract matched track information
    matched_tracks_pt = tracks_pt[matched_mask]
    matched_tracks_eta = tracks_eta[matched_mask]
    matched_tracks_phi = tracks_phi[matched_mask]
    matched_tracks_d0 = tracks_d0[matched_mask]
    matched_tracks_z0 = tracks_z0[matched_mask]
    matched_tracks_charge = tracks_charge[matched_mask]
    matched_jets_indices = min_dR_track_indices_np[matched_mask]

    # Append matched track data
    for idx in range(len(matched_tracks_pt)):
        track_info = (
            int(matched_jets_indices[idx]),        # jet_idx
            float(matched_tracks_pt[idx]),         # pt
            float(matched_tracks_eta[idx]),        # eta
            float(matched_tracks_phi[idx]),        # phi
            float(matched_tracks_d0[idx]),         # d0
            float(matched_tracks_z0[idx]),         # z0
            int(matched_tracks_charge[idx])        # charge
        )
        matched_tracks.append(track_info)

# %%
# Step 6: Create structured arrays and define dataset shapes

# Create structured array for jets
jets_data = np.zeros(len(matched_jet_pt), dtype=dtype_jets)
jets_data['pt'] = matched_jet_pt
jets_data['eta'] = matched_jet_eta
jets_data['flavour'] = matched_jet_flavour

# Create structured array for tracks
tracks_data = np.array(matched_tracks, dtype=dtype_tracks)

# %%
# Step 7: Create the HDF5 file and datasets

output_file = "/home/ssaini/dev/muonc/btagging/output_data/output_with_tracks.h5"

with h5py.File(output_file, "w") as f:
    # Create 'jets' dataset with LZF compression
    dataset_jets = f.create_dataset(
        "jets",
        data=jets_data,
        dtype=dtype_jets,
        compression="lzf"
    )
    dataset_jets.attrs["flavour_label"] = np.array(class_names, dtype="S")

    # Create 'tracks' dataset with LZF compression
    dataset_tracks = f.create_dataset(
        "tracks",
        data=tracks_data,
        dtype=dtype_tracks,
        compression="lzf"
    )

print(f"Conversion complete. Data saved as a .h5 file at {output_file}")
# %%