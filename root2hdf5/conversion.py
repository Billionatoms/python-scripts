# %%
# Importing the necessary libararies
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
        # Event Information
        "evevt",
        # Truth level information
        "nmcp", "mcpdg", "mcgst", "mcmox", "mcmoy", "mcmoz", "mcene", "mcvtx", "mcvty", "mcvtz", "mcmas", "mcsst", "mcepx", "mcepy",
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
    mcmas = data["mcmas"]
    mcsst = data["mcsst"]
    mcepx = data["mcepx"]
    mcepy = data["mcepy"]
    
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
dtype_hadrons = np.dtype([                          # Truth particles
    ("valid", 'bool'),                            # Validity flag, for more robust selection, true for any truth particle that is defined
    ("pt", 'f4'),                             # pT of the truth particles
    ("Lxy", 'f4'),                            # truth particle decay displacement
    ("flavour", 'i4'),                          # 5 if the truth particle is a B hadron, 4 if the truth particle is a C hadron, -1 otherwise.
    ("hadron_idx", 'i4'),                       # generator level barcode of the truth particle, used to link to objects in other datasets
    ("hadron_parent_idx", 'i4'),                # barcode of the parent B hadron (if one exists) of the truth particle
    ("mass", 'f4'),                           # Mass of the truth particle
#    ("dr", 'f4')                              # truth particle dR(particle, jet)
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
    
    
def Lxy(epx, epy, vtx, vty):
    """
    Calculate the transverse decay length Lxy.
    
    Parameters:
    epx, epy (np.array): Decay position coordinates.
    vtx, vty (np.array): Primary vertex coordinates.
    
    Returns:
    float: Calculated Lxy value.
    """
    delta_x = epx - vtx
    delta_y = epy - vty
    Lxy = np.sqrt(delta_x**2 + delta_y**2)
    return Lxy

# Define vectorized functions for delta_phi and delta_r
def delta_phi_vectorized(phi1, phi2):
    """
    Vectorized delta-phi calculation.
    Ensures results lie in [-pi, pi].
    phi1 and phi2 can be broadcast arrays.
    """
    dphi = phi2 - phi1
    # Bring dphi into [-pi, pi]
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    return dphi

def delta_r_vectorized(eta1, phi1, eta2, phi2):
    """
    Calculate the Delta R between eta1, phi1 and arrays eta2, phi2.
    
    Handles both scalar and array inputs for eta1 and phi1.
    
    Parameters:
    - eta1, phi1: Scalars or arrays representing hadron eta and phi.
    - eta2, phi2: Arrays representing jet eta and phi.
    
    Returns:
    - delta_r: Array of Delta R values between the hadron and each jet.
    """
    eta1 = np.atleast_1d(eta1)
    phi1 = np.atleast_1d(phi1)
    eta2 = np.atleast_1d(eta2)
    phi2 = np.atleast_1d(phi2)
    
    # Expand dimensions to enable broadcasting
    dEta = eta1[:, np.newaxis] - eta2[np.newaxis, :]
    dPhi = delta_phi_vectorized(phi1[:, np.newaxis], phi2[np.newaxis, :])
    
    return np.sqrt(dEta**2 + dPhi**2)

# calculating pt and eta values into a arrays

# calculating values for jets
jpt = pt(jmox, jmoy)
jeta = eta(jmox, jmoy, jmoz)
jphi = phi(jmox, jmoy)

# calculating values for truth particles
mcpt = pt(mcmox, mcmoy)
mceta = eta(mcmox, mcmoy, mcmoz)
mcphi = phi(mcmox, mcmoy)

# calculating values for hadrons
# hadron_data = delta_r(mceta, mcphi, jeta, jphi)

# calculating values for tracks
trk_charge = track_charge(tsome)
trk_momentum = track_p(tsome, tstnl)
trk_pt = track_pT(trk_momentum, tstnl)
trk_eta = track_eta(tstnl)
trk_phi = tsphi

# %%
# Validation Function
# def validate_arrays(*arrays, array_names):
#     for array, name in zip(arrays, array_names):
#         if ak.any(ak.is_none(array)):
#             print(f"Warning: '{name}' contains None values.")
#         if ak.any(ak.isnan(array)):
#             print(f"Warning: '{name}' contains NaN values.")

# # Validate essential arrays
# validate_arrays(
#     mceta, mcphi, jeta, jphi,
#     array_names=["mceta", "mcphi", "jeta", "jphi"]
# )

# %%
# Step 4: Buidling the 'hadrons' dataset

# hadrons_list = []

# for i in range(len(evevt)):
#     # Create a NumPy array for hadrons in event i
#     hadrons_data = np.empty(len(mcpt[i]), dtype=dtype_hadrons)
    
#     # Iterate over each hadron in the event
#     for j in range(len(mcpt[i])):
#         if mcgst[i][j] != 0:
#             # Valid hadron
#             hadrons_data["valid"][j] = True
#             hadrons_data["pt"][j] = mcpt[i][j]
#             hadrons_data["Lxy"][j] = Lxy(mcepx[i][j], mcepy[i][j], mcvtx[i][j], mcvty[i][j])
            
#             # Determine flavour
#             if np.abs(mcpdg[i][j]) == 5:
#                 hadrons_data["flavour"][j] = 5  # b-quark
#             elif np.abs(mcpdg[i][j]) == 4:
#                 hadrons_data["flavour"][j] = 4  # c-quark
#             else:
#                 hadrons_data["flavour"][j] = -1  # Undefined flavour
            
#             hadrons_data["hadron_idx"][j] = mcsst[i][j]
#             hadrons_data["hadron_parent_idx"][j] = -1  # Default value
#             hadrons_data["mass"][j] = mcmas[i][j]
            
#             # Compute ΔR
#             if len(jeta[i]) == 0 or len(mceta[i]) == 0:
#                 hadrons_data["dr"][j] = -1.0
#             else:
#                 dr = delta_r_vectorized(mceta[i][j], mcphi[i][j], jeta[i], jphi[i])
#                 min_dr = np.min(dr)  # Compute as scalar
                
#                 # Handle non-finite ΔR values
#                 if not np.isfinite(min_dr):
#                     min_dr = -1.0
                
#                 hadrons_data["dr"][j] = min_dr
        
        
#         else:
#             # Invalid hadron
#             hadrons_data["valid"][j] = False
#             hadrons_data["pt"][j] = np.nan
#             hadrons_data["Lxy"][j] = np.nan
#             hadrons_data["flavour"][j] = -1
#             hadrons_data["hadron_idx"][j] = -1
#             hadrons_data["hadron_parent_idx"][j] = -1
#             hadrons_data["mass"][j] = np.nan
#             hadrons_data["dr"][j] = np.nan
    
#     # Append the processed hadrons_data to the list
#     hadrons_list.append(hadrons_data)


hadrons_list = []

for i in range(len(evevt)):
    num_hadrons = len(mcpt[i])
    hadrons_data = np.empty(num_hadrons, dtype=dtype_hadrons)
    
    # Validity Mask
    valid_mask = mcgst[i] != 0
    hadrons_data["valid"] = valid_mask
    
    # Assign values for valid hadrons
    hadrons_data["pt"][valid_mask] = mcpt[i][valid_mask]
    hadrons_data["Lxy"][valid_mask] = Lxy(mcepx[i][valid_mask], mcepy[i][valid_mask],
                                         mcvtx[i][valid_mask], mcvty[i][valid_mask])
    
    # Flavour Assignment
    hadrons_data["flavour"][valid_mask & (np.abs(mcpdg[i]) == 5)] = 5  # b-quark
    hadrons_data["flavour"][valid_mask & (np.abs(mcpdg[i]) == 4)] = 4  # c-quark
    hadrons_data["flavour"][valid_mask & (np.abs(mcpdg[i]) <= 3)] = -1  # Undefined
    
    # Assign indices and mass
    hadrons_data["hadron_idx"][valid_mask] = mcsst[i][valid_mask]
    hadrons_data["hadron_parent_idx"][valid_mask] = -1  # Default value
    hadrons_data["mass"][valid_mask] = mcmas[i][valid_mask]
    
    # Compute ΔR for valid hadrons
    # if len(jeta[i]) > 0 and len(mceta[i]) > 0:
    #     hadrons_eta = mceta[i][valid_mask]
    #     hadrons_phi = mcphi[i][valid_mask]
    #     dr = delta_r_vectorized(hadrons_eta, hadrons_phi, jeta[i], jphi[i])
    #     min_dr = np.min(dr, axis=1)
    #     # Handle non-finite ΔR values
    #     min_dr[~np.isfinite(min_dr)] = -1.0
    #     hadrons_data["dr"][valid_mask] = min_dr
    # else:
    #     hadrons_data["dr"][valid_mask] = -1.0
    
    # Assign placeholders for invalid hadrons
    hadrons_data["pt"][~valid_mask] = np.nan
    hadrons_data["Lxy"][~valid_mask] = np.nan
    hadrons_data["flavour"][~valid_mask] = -1
    hadrons_data["hadron_idx"][~valid_mask] = -1
    hadrons_data["hadron_parent_idx"][~valid_mask] = -1
    hadrons_data["mass"][~valid_mask] = np.nan
    # hadrons_data["dr"][~valid_mask] = np.nan
    
    # Append to the list
    hadrons_list.append(hadrons_data)



# %%
# Step 5: Match truth-level quarks to jets based on proximity in eta-phi space

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
# Step 6: Calculate 'truth_vertex_idx' based on stable particles

# Initialize a list to hold 'truth_vertex_idx' for each event
truth_vertex = []

decimals = 6  # Precision for rounding

for i in range(len(evevt)):
    # 1) Filter for stable particles
    mcgst_event = ak.to_numpy(mcgst[i])
    mcvtx_event = ak.to_numpy(mcvtx[i])
    mcvty_event = ak.to_numpy(mcvty[i])
    mcvtz_event = ak.to_numpy(mcvtz[i])

    stable_mask = (mcgst_event == 1)

    # 2) Round vertex coordinates for stable particles
    stable_mcvtx = np.round(mcvtx_event[stable_mask], decimals=decimals)
    stable_mcvty = np.round(mcvty_event[stable_mask], decimals=decimals)
    stable_mcvtz = np.round(mcvtz_event[stable_mask], decimals=decimals)

    # Stack into shape (N_stable, 3)
    stable_coords = np.column_stack((stable_mcvtx, stable_mcvty, stable_mcvtz))

    # Group identical vertices and assign a unique index
    # unique_coords is shape (K, 3), inverse_indices is shape (N_stable,)
    unique_coords, inverse_indices = np.unique(stable_coords, axis=0, return_inverse=True)

    # Prepare output array for the entire event (all particles, stable or not)
    truth_vertex_event = np.full(len(mcgst_event), -1, dtype=np.int32)

    # 3) Assign index to each stable particle
    # Each stable_mask entry gets the corresponding unique vertex index
    truth_vertex_event[stable_mask] = inverse_indices

    truth_vertex.append(truth_vertex_event)


# %%
# Step 7: Match tracks to jets and calculate quantities for 'consts' dataset

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
                
                # Safely assign truth_vertex_idx
                if i < len(truth_vertex) and k < len(truth_vertex[i]):
                    truth_vertex_idx = truth_vertex[i][k]
                else:
                    truth_vertex_idx = -1  # Default value if index is out of range
                
                
                const = (
                    -1,  # truth_hadron_idx (not available)
                    truth_vertex_idx,  # truth_vertex_idx
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
# Step 8: Create a structured array to hold data and define its shapes and chunk sizes

# Create a structured array to hold the jets data
jets_data = np.empty(len(matched_jet_pt), dtype=dtype_jets)
jets_data['pt'] = matched_jet_pt
jets_data['eta'] = matched_jet_eta
jets_data['flavour'] = matched_jet_flavour

# Create a structure array to hold the 'hadrons' data
hadron_data = np.concatenate(hadrons_list)

# Verify hadron_data integrity
print(f"Shape of hadron_data: {hadron_data.shape}")
print(f"Dtype of hadron_data: {hadron_data.dtype}")
print("First 5 hadrons:")
print(hadron_data[:100])

if hadron_data.size == 0:
    raise ValueError("hadrons_data is empty. Cannot create an empty 'hadrons' dataset.")


# Flatten the consts_data for HDF5 storage (since it's a nested list)
# Estimate total size by iterating once (optional, if possible)
flat_consts_data = [item for sublist in consts_data for subsublist in sublist for item in subsublist]

# Define dataset shapes and chunk sizes
shape_consts = (len(flat_consts_data),)
#chunks_consts = (min(1000, shape_consts[0]),)



# %%
# Step 9: Create the HDF5 file and datasets
with h5py.File("/home/ssaini/dev/muonc/btagging/output_data/output_10Jan2025_v6.h5", "w") as f:
    # Create 'consts' dataset with LZF compression
    dataset_consts = f.create_dataset(
        "consts",
        data=np.array(flat_consts_data, dtype=dtype_consts),
        dtype=dtype_consts,
        compression="lzf",
        chunks=True  # Enable chunking; h5py will choose the chunk size
    )
    

    # Create 'hadrons' dataset with LZF compression
    # Directly pass the data instead of specifying shape and then assigning
    dataset_hadrons = f.create_dataset(
        "hadrons",
        data=hadron_data,
        dtype=dtype_hadrons,
        compression="lzf",
        chunks=True  # Enable chunking; h5py will choose an appropriate chunk size
    )
#    dataset_hadrons[...] = hadron_data  # Optionally initialize

    # Create 'jets' dataset with LZF compression
    dataset_jets = f.create_dataset(
        "jets",
        data=jets_data,
        dtype=dtype_jets,
        compression="lzf",
        chunks=True
    )
#    dataset_jets[...] = jets_data  # Store the calculated data
    
    # Set the 'flavour_label' attribute for the 'jets' dataset
    dataset_jets.attrs["flavour_label"] = np.array(class_names, dtype="S")


print("Conversion complete. Data saved as a .h5 file in /home/ssaini/dev/muonc/btagging/output_data")


# %%