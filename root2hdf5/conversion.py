# %%
# Importing the necessary libararies
import uproot
import h5py
import numpy as np
import awkward as ak
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# Step 1: Open the ROOT file and extract data
file_path = "/mnt/c/Users/Saurabh/cernbox/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root"

with uproot.open(file_path) as f:
    tree = f["LCTuple"]
    
    # List of branches to extract
    branches = [
        # Event Information
        "evevt",
        # Truth level information
        "nmcp", "mcpdg", "mcgst", "mcmox", "mcmoy", "mcmoz", "mcene", "mcvtx", "mcvty", "mcvtz",
        # Jet Information
        "njet", "jmox", "jmoy", "jmoz", "jene", "jcha", "jmas",
        # Track Information
        "ntrk", "tsdze", "tsphi", "tsome", "tszze", "tstnl", "tscov", "tsrpx", "tsrpy", "tsrpz", "trsip", "trsfh", "trslh", "trsca", "ntrst" 
        
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
    mcvtx = data["mcvtx"]
    mcvty = data["mcvty"]
    mcvtz = data["mcvtz"]
    
    # Jet Information
    njet = data["njet"]
    jmox = data["jmox"]
    jmoy = data["jmoy"]
    jmoz = data["jmoz"]
    jene = data["jene"]
    jmas = data["jmas"]
    jcha = data["jcha"]
    
    # Track Information
    ntrk = data["ntrk"]     # Total number of tracks in the event
    tsdze = data["tsdze"]   # Track state's d0 parameter
    tsphi = data["tsphi"]   # Track state's phi parameter
    tsome = data["tsome"]   # Track state's curvature Ω
    tszze = data["tszze"]   # Track state's z0 parameter
    tstnl = data["tstnl"]   # Tracks state's tanλ
    tscov = data["tscov"]   # Tracks state's covariance matrix
    tsrpx = data["tsrpx"]   # global reference position of the track state in the x direction
    tsrpy = data["tsrpy"]   # global reference position of the track state in the y direction
    tsrpz = data["tsrpz"]   # global reference position of the track state in the z direction
    trsip = data["trsip"]   # impact parameter significance of the track
    trsfh = data["trsfh"]   # first hit of the track
    trslh = data["trslh"]   # last hit of the track
    trsca = data["trsca"]   # track state at calorimeter
    ntrst = data["ntrst"]   # number of track states


# %%
# Step 2: Define the compound data type for each dataset

# Dataset 'consts'
dtype_consts = np.dtype([
    ("truth_hadron_idx", np.int32),                 # particle hadron ID?
    ("truth_vertex_idx", np.int32),                 # particle vertex ID?
    ("truth_origin_label", np.int32),               # particle origin label?
    ("valid", np.bool_),                            # Valid???
    ("is_gamma", np.bool_),                         # is it a photon?
    ("is_neutral_had", np.bool_),                   # is it a neutral hadron?
    ("is_electron", np.bool_),                      # is it a electron?
    ("is_muon", np.bool_),                          # is it a muon?
    ("is_charged_had", np.bool_),                   # is it a charged hadron?
    ("charge", np.int32),                           # Charge of the particle
    ("phi_rel", np.float32),                        # phi of the particle?
    ("eta_rel", np.float32),                        # eta of the particle
    ("pt_frac", np.float32),                        # pT fraction of the particle?
    ("d0", np.float32),                             # d0 of the particle
    ("z0", np.float32),                             # z0 of the particle 
    ("dr", np.float32),                             # dr of the particle
    ("signed_2d_ip", np.float32),                   # signed impact parameter in transverse
    ("signed_3d_ip", np.float32)                    # signed impact parameter in transverse and longitudinal
])


# Dataset 'hadrons'
dtype_hadrons = np.dtype([                          # Tracks???
    ("valid", np.bool_),                            # Valid???
    ("pt", np.float32),                             # pT of the tracks?
    ("Lxy", np.float32),                            # Angular momentum 
    ("flavour", np.int32),                          # Flavour
    ("hadron_idx", np.int32),                       # Hadron ID???
    ("hadron_parent_idx", np.int32),                # Hadron parent ID
    ("mass", np.float32),                           # Mass
    ("dr", np.float32)                              # dr
])


# Dataset 'jets'
dtype_jets = np.dtype([                             # Jets
    ("pt", np.float32),                             # pT of the jets
    ("eta", np.float32),                            # eta of the jets
    ("flavour", np.int32)                          # flavour of the jets
#    ("flavour_label", np.int32)                     # flavour label of the jets
])



# %%
# Step 3: Calculations of quantities not in root file

# Function to calculate total momentum
def mo(px, py, pz):
    return np.sqrt(px**2 + py**2 + pz**2)

# Function to calculate transverse momentum (pt)
def pt(px, py):
    return np.sqrt(px**2 + py**2)

# Funtion to calculate theta
def theta(px, py, pz):
    pt_values = pt(px, py)
    return np.arctan2(pt_values, pz)

# Function to calculate pseudorapidity (eta)
def eta(px, py, pz):
    pt_values = pt(px, py)
    theta_values = theta(px, py, pz)
    return -1 * np.log(np.tan(theta_values/2))

# Function to calculate azimuthal angle (phi)
def phi(px, py):
    return np.arctan2(py, px)

def track_eta(tan_lambda):
    angle = np.arctan(tan_lambda)
    return -1 * np.log(np.tan(angle/2))

# Function to calculate ΔR
def delta_phi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    return np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

# Functions for delta_phi and delta_r
def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)

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

# Function to calculate signed impact parameter significance
def signed_ip(d0, d0sig, phi_jet, phi_track):
    theta = phi_jet - phi_track
    sj = 1.0 if (np.sin(theta))*d0 >= 0 else -1.0
    return np.abs(d0 / d0sig) * sj

# Function to determine particle type based on PDG ID
def pid(pdg_id):
    if pdg_id == 22:
        return "gamma"
    elif pdg_id in [111, 130, 310, 2112]:
        return "neutral_had"
    elif pdg_id in [11, -11]:
        return "electron"
    elif pdg_id in [13, -13]:
        return "muon"
    elif pdg_id in [211, -211, 321, -321, 2212, -2212]:
        return "charged_had"
    else:
        return "unknown"


# calculating pt and eta values into a arrays

# calculating values for jets
jpt = pt(jmox, jmoy)
jeta = eta(jmox, jmoy, jmoz)
jphi = phi(jmox, jmoy)

# calculating values for truth particles
mcpt = pt(mcmox, mcmoy)
mceta = eta(mcmox, mcmoy, mcmoz)
mcphi = phi(mcmox, mcmoy)

# calculating values for tracks
trk_charge = track_charge(tsome)
trk_momentum = track_p(tsome, tstnl)
trk_pt = track_pT(trk_momentum, tstnl)
trk_eta = track_eta(tstnl)
trk_phi = tsphi


# %%
# Step 4: Match truth-level quarks to jets based on proximity in eta-phi space

# Define the label mapping and class names
label_map = {0: 0, 4: 1, 5: 2}
class_names = ["ujets", "cjets", "bjets"]

# Create a structured array to hold the jets data
b_quark_indices = []
c_quark_indices = []
u_quark_indices = []

# Identify b-quarks, c-quarks, and light quarks (u, d, s)
for i in range(len(mcpt)):
    b_quark_indices.append(np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) == 5))[0])
    c_quark_indices.append(np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) == 4))[0])
    u_quark_indices.append(np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) <= 3))[0])

matched_jet_pt = []
matched_jet_eta = []
matched_jet_flavour = []

# Function to match jets to quarks and label them
def match_jets_to_quarks(jet_eta, jet_phi, quark_eta, quark_phi, flavour):
    if len(jet_eta) == 0 or len(quark_eta) == 0:
        return  # Skip if there are no jets or no quarks
    distances = np.array([[delta_r(jet_eta[j], jet_phi[j], quark_eta[q], quark_phi[q])
                           for q in range(len(quark_eta))]
                          for j in range(len(jet_eta))])
    closest_jets = np.argmin(distances, axis=0)
    for j in closest_jets:
        matched_jet_pt.append(jpt[i][j])
        matched_jet_eta.append(jeta[i][j])
        matched_jet_flavour.append(label_map[flavour])
        
        
# Match jets to b-quarks
for i in range(len(jpt)):
    if len(b_quark_indices[i]) > 0:
        match_jets_to_quarks(jeta[i], jphi[i], mceta[i][b_quark_indices[i]], mcphi[i][b_quark_indices[i]], 5)

# Match jets to c-quarks
for i in range(len(jpt)):
    if len(c_quark_indices[i]) > 0:
        match_jets_to_quarks(jeta[i], jphi[i], mceta[i][c_quark_indices[i]], mcphi[i][c_quark_indices[i]], 4)

# Match jets to light quarks (u, d, s)
for i in range(len(jpt)):
    if len(u_quark_indices[i]) > 0:
        match_jets_to_quarks(jeta[i], jphi[i], mceta[i][u_quark_indices[i]], mcphi[i][u_quark_indices[i]], 0)

# %%
# Step 5: Calculate 'truth_vertex_idx' based on stable particles

# Initialize a list to hold 'truth_vertex_idx' for each event
truth_vertex_idx_list = []

decimals = 3  # Precision for rounding to handle float comparisons

# Loop over each event to process vertices
for i in range(len(evevt)):
    # Extract per-event particle data and convert to NumPy arrays
    mcgst_event = ak.to_numpy(mcgst[i])
    mcvtx_event = ak.to_numpy(mcvtx[i])
    mcvty_event = ak.to_numpy(mcvty[i])
    mcvtz_event = ak.to_numpy(mcvtz[i])
    
    # Step 1: Filter stable particles (mcgst == 1)
    stable_mask = mcgst_event == 1
    
    # Extract vertex coordinates for stable particles
    stable_mcvtx = mcvtx_event[stable_mask]
    stable_mcvty = mcvty_event[stable_mask]
    stable_mcvtz = mcvtz_event[stable_mask]
    
    # Step 2: Round vertex coordinates to handle float comparison
    rounded_mcvtx = np.round(stable_mcvtx, decimals=decimals)
    rounded_mcvty = np.round(stable_mcvty, decimals=decimals)
    rounded_mcvtz = np.round(stable_mcvtz, decimals=decimals)
    
    # Step 3: Stack the rounded coordinates into a 2D array (N_stable_particles, 3)
    if len(rounded_mcvtx) > 0:
        try:
            vertex_coords = np.stack([rounded_mcvtx, rounded_mcvty, rounded_mcvtz], axis=1)
        except ValueError as e:
            logger.error(f"Event {i}: Error stacking coordinates - {e}")
            vertex_coords = np.empty((0, 3))
        
        # Find unique vertices and assign an index to each group
        unique_vertices, unique_indices = np.unique(vertex_coords, axis=0, return_inverse=True)
    else:
        # If no stable particles, create an empty array
        unique_indices = np.array([], dtype=int)
    
    # Initialize 'truth_vertex_idx' for all particles in the event with -1
    truth_vertex_idx_event = -1 * np.ones(len(mcgst_event), dtype=int)
    
    # Assign the unique vertex indices to stable particles
    stable_indices = np.where(stable_mask)[0]
    
    # Ensure that the lengths match before assignment
    if len(unique_indices) == len(stable_indices):
        truth_vertex_idx_event[stable_indices] = unique_indices
    elif len(unique_indices) < len(stable_indices):
        # In case unique_indices has fewer elements due to some unexpected reason
        truth_vertex_idx_event[stable_indices[:len(unique_indices)]] = unique_indices
        logger.warning(f"Event {i}: Mismatch in unique_indices and stable_indices lengths.")
    else:
        # If unique_indices somehow has more elements, which shouldn't happen
        truth_vertex_idx_event[stable_indices] = unique_indices[:len(stable_indices)]
        logger.warning(f"Event {i}: Excess unique_indices, some indices may be ignored.")
    
    # Append the per-event 'truth_vertex_idx' to the list
    truth_vertex_idx_list.append(truth_vertex_idx_event)

# Convert the list of arrays into an Awkward Array
truth_vertex_idx = ak.Array(truth_vertex_idx_list)


# %%
# Step 6: Match tracks to jets and calculate quantities for 'consts' dataset

consts_data = []

# Function to match tracks to jets and calculate quantities
for i in range(len(evevt)):
    event_consts = []
    for j in range(njet[i]):
        jet_consts = []
        for k in range(ntrk[i]):
            if delta_r(jeta[i][j], jphi[i][j], trk_eta[i][k], trk_phi[i][k]) < 0.4:  # Matching criterion
                d0sig = np.sqrt(tscov[i][k][0])  # Extract d0 uncertainty from the covariance matrix
                z0sig = np.sqrt(tscov[i][k][9])  # Extract z0 uncertainty from the covariance matrix
                signed_2d_ip = signed_ip(tsdze[i][k],
                                         d0sig,
                                         jphi[i][j], tsphi[i][k])
                signed_3d_ip = signed_ip(np.sqrt(tsdze[i][k]**2 + tszze[i][k]**2),
                                         np.sqrt(d0sig**2 + z0sig**2),
                                         jphi[i][j], tsphi[i][k])
                
                # Determine particle type
                particle_type = pid(mcpdg[i][k])
                is_gamma = particle_type == "gamma"
                is_neutral_had = particle_type == "neutral_had"
                is_electron = particle_type == "electron"
                is_muon = particle_type == "muon"
                is_charged_had = particle_type == "charged_had"
                
                const = (
                    -1,  # truth_hadron_idx (not available)
                    truth_vertex_idx,  # truth_vertex_idx (not available)
                    mcpdg[i][k],  # truth_origin_label (using PDG ID)
                    True,  # valid (assuming all tracks are valid)
                    is_gamma,
                    is_neutral_had,
                    is_electron,
                    is_muon,
                    is_charged_had,
                    trk_charge[i][k],
                    tsphi[i][k] - jphi[i][j],
                    trk_eta[i][k] - jeta[i][j],
                    trk_pt[i][k] / jpt[i][j],
                    tsdze[i][k],
                    tszze[i][k],
                    delta_r(jeta[i][j], jphi[i][j], trk_eta[i][k], trk_phi[i][k]),
                    signed_2d_ip,
                    signed_3d_ip
                )
                jet_consts.append(const)
        event_consts.append(jet_consts)
    consts_data.append(event_consts)




# %%
# Step 7: Create a structured array to hold data and define its shapes and chunk sizes

import itertools

# Create a structured array to hold the jets data
jets_data = np.empty(len(matched_jet_pt), dtype=dtype_jets)
jets_data['pt'] = matched_jet_pt
jets_data['eta'] = matched_jet_eta
jets_data['flavour'] = matched_jet_flavour

# Flatten the consts_data for HDF5 storage (since it's a nested list)
# Estimate total size by iterating once (optional, if possible)
total_consts = sum(len(subsublist) for sublist in consts_data for subsublist in sublist)

# Define dataset shapes and chunk sizes
shape_consts = (total_consts,)
chunks_consts = (min(100, shape_consts[0]),)

shape_hadrons = (13500000, 5)
chunks_hadrons = (100, 5)


# %%
# Step 8: Create the HDF5 file and datasets
with h5py.File("/home/ssaini/dev/muonc/btagging/output_data/output_03Jan2025_v2.h5", "w") as f:
    # Create 'consts' dataset with LZF compression
    dataset_consts = f.create_dataset(
        "consts",
        shape=shape_consts,
        dtype=dtype_consts,
        chunks=chunks_consts,
        compression="lzf"
    )

    # Create 'hadrons' dataset with LZF compression
    dataset_hadrons = f.create_dataset(
        "hadrons",
        shape=shape_hadrons,
        dtype=dtype_hadrons,
        chunks=chunks_hadrons,
        compression="lzf"
    )
    dataset_hadrons[...] = np.zeros(shape_hadrons, dtype=dtype_hadrons)  # Optionally initialize

    # Create 'jets' dataset with LZF compression
    dataset_jets = f.create_dataset(
        "jets",
        data=jets_data,
        dtype=dtype_jets,
        compression="lzf"
    )
    dataset_jets[...] = jets_data  # Store the calculated data
    
    # Set the 'flavour_label' attribute for the 'jets' dataset
    dataset_jets.attrs["flavour_label"] = np.array(class_names, dtype="S")

    # Flatten and write 'consts' data in batches to minimize memory usage
    batch_size = 10
    flat_consts_generator = (
        tuple(ak.to_list(item))
        for sublist in consts_data
        for subsublist in sublist
        for item in subsublist
    )
    
    batch = []
    index = 0
    for const in flat_consts_generator:
        batch.append(const)
        if len(batch) == batch_size:
            dataset_consts[index:index+batch_size] = np.array(batch, dtype=dtype_consts)
            index += batch_size
            batch = []
    
    # Write any remaining data
    if batch:
        dataset_consts[index:index+len(batch)] = np.array(batch, dtype=dtype_consts)


print("Conversion complete. Data saved as a .h5 file in /home/ssaini/dev/muonc/btagging/output_data")


# %%