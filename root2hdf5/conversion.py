# %%
# Importing the necessary libararies
import uproot
import h5py
import numpy as np
import awkward as ak
from ROOT import TLorentzVector
import cupy as cp
from numba import njit, prange

# %%
# Step 1: Open the ROOT file and extract data
f = uproot.open("/mnt/c/Users/Saurabh/cernbox/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root")
tree = f["LCTuple"]

# %%
# Event level information
evevt = tree["evevt"].array()      # Event number
 
# evrun = tree["evrun"].array()      # Event run number
# evwgt = tree["evwgt"].array()      # Event weight
# evtim = tree["evtim"].array()      # Event time stamp
# evsig = tree["evsig"].array()      # Event cross-section
# evene = tree["evene"].array()      # Event energy
# evpoe = tree["evpoe"].array()      # Event Polarization 1
# evpop = tree["evpop"].array()      # Event Polarization 2
# evnch = tree["evnch"].array()      # ???
# evpro = tree["evpro"].array()      # Event process


# %%
# Truth level information
nmcp = tree["nmcp"].array()        # Number of truth particle
mcpdg = tree["mcpdg"].array()      # Truth particle PDG
mcgst = tree["mcgst"].array()      # Truth particle generator status
mcsst = tree["mcsst"].array()      # Truth particle simulator status
mcmox = tree["mcmox"].array()      # Truth partcile momentumn in x-direction
mcmoy = tree["mcmoy"].array()      # Truth partcile momentumn in y-direction
mcmoz = tree["mcmoz"].array()      # Truth partcile momentumn in z-direction
mcvtx = tree["mcvtx"].array()      # Truth particle vertex x
mcvty = tree["mcvty"].array()      # Truth particle vertex y
mcvtz = tree["mcvtz"].array()      # Truth particle vertex z
mcene = tree["mcene"].array()      # Truth particle energy

# mcepx = tree["mcepx"].array()      # Truth particle end point in x
# mcepy = tree["mcepy"].array()      # Truth particle end point in y
# mcepz = tree["mcepz"].array()      # Truth particle end point in z
# mcmass = tree["mcmas"].array()     # Truth particle mass
# mccha = tree["mccha"].array()      # Truth particle charge
# mctim = tree["mctim"].array()      # Truth particle time???
# mcspx = tree["mcspx"].array()      # Truth particle spin in x-direction
# mcspy = tree["mcspy"].array()      # Truth particle spin in y-direction
# mcspz = tree["mcspz"].array()      # Truth particle spin in z-direction
# mccf0 = tree["mccf0"].array()      # Truth particle color flow 0
# mccf1 = tree["mccf1"].array()      # Truth particle color flow 1
# mcpa0 = tree["mcpa0"].array()      # Truth particle parent 0
# mcpa1 = tree["mcpa1"].array()      # Truth particle parent 1
# mcda0 = tree["mcda0"].array()      # Truth particle daughter 0
# mcda1 = tree["mcda1"].array()      # Truth particle daughter 1
# mcda2 = tree["mcda2"].array()      # Truth particle daughter 2
# mcda3 = tree["mcda3"].array()      # Truth particle daughter 3
# mcda4 = tree["mcda4"].array()      # Truth particle daughter 4
# mcori = tree["mcori"].array()      # CollID???



# %%
# Jet Information
njet = tree["njet"].array()        # Number of jets
jmox = tree["jmox"].array()        # Jet momentumn in x-direction
jmoy = tree["jmoy"].array()        # Jet momentumn in y-direction
jmoz = tree["jmoz"].array()        # Jet momentumn in z-direction
jmas = tree["jmas"].array()        # Jet's mass
jene = tree["jene"].array()        # Jet's energy
jcha = tree["jcha"].array()        # Jet's charge

# jcov0 = tree["jcov0"].array()      # Jet covariance cov(d0, d0)
# jcov1 = tree["jcov1"].array()      # Jet covariance cov(ϕ, d0)
# jcov2 = tree["jcov2"].array()      # Jet covariance cov(ϕ, ϕ)
# jcov3 = tree["jcov3"].array()      # Jet covariance cov(Ω, d0)
# jcov4 = tree["jcov4"].array()      # Jet covariance cov(Ω, ϕ)
# jcov5 = tree["jcov5"].array()      # Jet covariance cov(Ω, Ω)
# jcov6 = tree["jcov6"].array()      # Jet covariance cov(z0, d0)
# jcov7 = tree["jcov7"].array()      # Jet covariance cov(z0, ϕ)
# jcov8 = tree["jcov8"].array()      # Jet covariance cov(z0, Ω)
# jcov9 = tree["jcov9"].array()      # Jet covariance cov(z0, z0)



# %%
# Track level information
ntrk = tree["ntrk"].array()       # Total Number of tracks in the event
tsdze = tree["tsdze"].array()     # Track d0 parameter
tsphi = tree["tsphi"].array()     # Track phi parameter
tsome = tree["tsome"].array()     # Curvature $1/mm$ of the track. Proportional to magnetic field. (Ω)
tszze = tree["tszze"].array()     # Track z0 parameter
tstnl = tree["tstnl"].array()     # tanλ, with λ = pitch angle
tscov = tree["tscov"].array()     # Covariance matrix, stored as array of size 15 per hit
tsrpx = tree["tsrpx"].array()     # Global reference position of the track state in x direction
tsrpy = tree["tsrpy"].array()     # Global reference position of the track state in y direction
tsrpz = tree["tsrpz"].array()     # Global reference position of the track state in z direction
trsip = tree["trsip"].array()     # Significance of the track's Impact Parameter(e.g., deviation from expected vertex)
trsfh = tree["trsfh"].array()     # First hit associated with the track
trslh = tree["trslh"].array()     # Last hit associated with the track
trsca = tree["trsca"].array()     # Track state at calorimeter
ntrst = tree["ntrst"].array()     # Number of tarck states

# trori = tree["trori"].array()     # Origin of the track, typically representing the detector or simulation process.
# trtyp = tree["trtyp"].array()     # Type of the track, categorizing its origin or purpose.
# trch2 = tree["trch2"].array()     # Chi-squared of the track fit, indicating the quality of the track reconstruction
# trndf = tree["trndf"].array()     # Number of degrees of freedom in the track fit
# tredx = tree["tredx"].array()     # Track's dE/dx
# trede = tree["trede"].array()     # Track's dE/dx error
# trrih = tree["trrih"].array()     # Radius of inner most hit
# trthn = tree["trthn"].array()     # ???
# trthi = tree["trthi"].array()     # ????
# trshn = tree["trshn"].array()     # Subdetector Hit number
# trthd = tree["trthd"].array()     # ???
# trnts = tree["trnts"].array()     # Track states size???
# trfts = tree["trfts"].array()     # ???
# tsloc = tree["tsloc"].array()     # Track location



# %%
# Step 2: Define the compound data type for each dataset

# # Dataset 'consts'
# dtype_consts = np.dtype([
#     ("truth_hadron_idx", np.int32),                 # particle hadron ID?
#     ("truth_vertex_idx", np.int32),                 # particle vertex ID?
#     ("truth_origin_label", np.int32),               # particle origin label?
#     ("valid", np.bool_),                            # Valid???
#     ("is_gamma", np.bool_),                         # is it a photon?
#     ("is_neutral_had", np.bool_),                   # is it a neutral hadron?
#     ("is_electron", np.bool_),                      # is it a electron?
#     ("is_muon", np.bool_),                          # is it a muon?
#     ("is_charged_had", np.bool_),                   # is it a charged hadron?
#     ("charge", np.int32),                           # Charge of the particle
#     ("phi_rel", np.float32),                        # phi of the particle?
#     ("eta_rel", np.float32),                        # eta of the particle
#     ("pt_frac", np.float32),                        # pT fraction of the particle?
#     ("d0", np.float32),                             # d0 of the particle
#     ("z0", np.float32),                             # z0 of the particle 
#     ("dr", np.float32),                             # dr of the particle
#     ("signed_2d_ip", np.float32),                   # signed impact parameter in transverse
#     ("signed_3d_ip", np.float32)                    # signed impact parameter in transverse and longitudinal
# ])


# %%
# # Dataset 'hadrons'
# dtype_hadrons = np.dtype([                          # Tracks???
#     ("valid", np.bool_),                            # Valid???
#     ("pt", np.float32),                             # pT of the tracks?
#     ("Lxy", np.float32),                            # Angular momentum 
#     ("flavour", np.int32),                          # Flavour
#     ("hadron_idx", np.int32),                       # Hadron ID???
#     ("hadron_parent_idx", np.int32),                # Hadron parent ID
#     ("mass", np.float32),                           # Mass
#     ("dr", np.float32)                              # dr
# ])


# %%
# Dataset 'jets'
dtype_jets = np.dtype([                             # Jets
    ("pt", np.float32),                             # pT of the jets
    ("eta", np.float32),                            # eta of the jets
    ("flavour", np.int32)                          # flavour of the jets
#    ("flavour_label", np.int32)                     # flavour label of the jets
])



# %%
# Step 3: Create TLorentzVector objects and calculate quantities using actual energies

# Function to create TLorentzVectors from momentum components and energies
def lorentz_vectors(px_array, py_array, pz_array, energy_array):
    vectors_events = []
    for px_event, py_event, pz_event, en_event in zip(px_array, py_array, pz_array, energy_array):
        vectors = []
        for px, py, pz, en in zip(px_event, py_event, pz_event, en_event):
            vec = TLorentzVector()
            vec.SetPxPyPzE(px, py, pz, en)
            vectors.append(vec)
        vectors_events.append(vectors)
    return vectors_events


# %%
# Function to create TLorentzVectors for tracks

# # Constants
# B_field = 3.57  # Magnetic field in Tesla
# mass_pion = 139.57  # Pion mass in MeV/c^2

# def track_lorentz_vectors(tsphi_array, tsome_array, tstnl_array):
#     vectors_events = []
#     for phi_event, omega_event, tan_lambda_event in zip(tsphi_array, tsome_array, tstnl_array):
#         vectors = []
#         for phi, omega, tan_lambda in zip(phi_event, omega_event, tan_lambda_event):
#             omega = np.abs(omega)  # Ensure omega is positive
#             if omega == 0:
#                 p_T = 0  # Avoid division by zero
#             else:
#                 p_T = (0.3 * B_field) / (omega * 1000)  # Convert omega from 1/mm to 1/m

#             px = p_T * np.cos(phi)
#             py = p_T * np.sin(phi)
#             pz = p_T * tan_lambda

#             energy = np.sqrt(px**2 + py**2 + pz**2)  # Assuming massless particles

#             vec = TLorentzVector()
#             vec.SetPxPyPzE(px, py, pz, energy)
#             vectors.append(vec)
#         vectors_events.append(vectors)
#     return vectors_events



# %%
# calculating pt and eta values into a arrays

# Create TLorentzVector objects for jets
jets = lorentz_vectors(jmox, jmoy, jmoz, jene)

# Create TLorentzVector objects for truth particles
truth_particles = lorentz_vectors(mcmox, mcmoy, mcmoz, mcene)

# Create TLorentzVector objects for tracks
# tracks = track_lorentz_vectors(tsphi, tsome, tstnl)


# %%
# Step 4: Calculating quantities using TLorentzVector methods

# For jets
# jpt = [[vec.Pt() for vec in event] for event in jets]
# jeta = [[vec.Eta() for vec in event] for event in jets]
# jphi = [[vec.Phi() for vec in event] for event in jets]

# # For truth particles
# mcpt = [[vec.Pt() for vec in event] for event in truth_particles]
# mceta = [[vec.Eta() for vec in event] for event in truth_particles]
# mcphi = [[vec.Phi() for vec in event] for event in truth_particles]

# Step 4: Calculating quantities using TLorentzVector methods with CuPy

# For jets
jpt = []
jeta = []
jphi = []
for event in jets:
    pt_list = cp.array([vec.Pt() for vec in event], dtype=cp.float32)
    eta_list = cp.array([vec.Eta() for vec in event], dtype=cp.float32)
    phi_list = cp.array([vec.Phi() for vec in event], dtype=cp.float32)
    jpt.append(pt_list)
    jeta.append(eta_list)
    jphi.append(phi_list)

# For truth particles
mcpt = []
mceta = []
mcphi = []
for event in truth_particles:
    pt_list = cp.array([vec.Pt() for vec in event], dtype=cp.float32)
    eta_list = cp.array([vec.Eta() for vec in event], dtype=cp.float32)
    phi_list = cp.array([vec.Phi() for vec in event], dtype=cp.float32)
    mcpt.append(pt_list)
    mceta.append(eta_list)
    mcphi.append(phi_list)

# # For tracks
# trk_pt = [[vec.Pt() for vec in event] for event in tracks]
# trk_eta = [[vec.Eta() for vec in event] for event in tracks]
# trk_phi = [[vec.Phi() for vec in event] for event in tracks]

# # Calculate track charges
# trk_charge = [np.sign(curv_event) for curv_event in tsome]

# %%
# Function to determine particle type based on PDG ID
# def determine_particle_type(pdg_id):
#     if pdg_id == 22:
#         return "gamma"
#     elif pdg_id in [111, 130, 310, 2112]:
#         return "neutral_had"
#     elif pdg_id in [11, -11]:
#         return "electron"
#     elif pdg_id in [13, -13]:
#         return "muon"
#     elif pdg_id in [211, -211, 321, -321, 2212, -2212]:
#         return "charged_had"
#     else:
#         return "unknown"

# %%
# Step 5: Match truth-level quarks to jets based on proximity in eta-phi space using Numba

# Label mapping and class names
label_map = {0: 0, 4: 1, 5: 2}
class_names = ["ujets", "cjets", "bjets"]

# Initialize lists for storing matched jets
matched_jet_pt = []
matched_jet_eta = []
matched_jet_flavour = []

# Define Numba-accelerated function to calculate delta R
@njit
def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    # Wrap-around for phi
    if dphi > np.pi:
        dphi -= 2 * np.pi
    elif dphi < -np.pi:
        dphi += 2 * np.pi
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

# Iterate over events
for i in range(len(evevt)):
    if len(jets[i]) == 0:
        continue  # Skip if no jets in the event

    # Get indices of quarks in this event
    b_quark_indices = np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) == 5))[0]
    c_quark_indices = np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) == 4))[0]
    u_quark_indices = np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) <= 3))[0]

    # Get jet kinematics
    jet_pt = jpt[i]
    jet_eta = jeta[i]
    jet_phi = jphi[i]

    # Function to match jets to quarks
    def match_jets_to_quarks(quark_indices, flavour):
        for q_idx in quark_indices:
            quark_eta = truth_particles[i][q_idx].Eta()
            quark_phi = truth_particles[i][q_idx].Phi()
            min_dr = np.inf
            matched_idx = -1
            for j in range(len(jet_pt)):
                dr = delta_r(jet_eta[j].get(), jet_phi[j].get(), quark_eta, quark_phi)
                if dr < min_dr:
                    min_dr = dr
                    matched_idx = j
            if matched_idx >= 0:
                matched_jet_pt.append(jet_pt[matched_idx].get())
                matched_jet_eta.append(jet_eta[matched_idx].get())
                matched_jet_flavour.append(label_map[flavour])

    # Match jets to b-quarks
    if len(b_quark_indices) > 0:
        match_jets_to_quarks(b_quark_indices, 5)

    # Match jets to c-quarks
    if len(c_quark_indices) > 0:
        match_jets_to_quarks(c_quark_indices, 4)

    # Match jets to light quarks (u, d, s)
    if len(u_quark_indices) > 0:
        match_jets_to_quarks(u_quark_indices, 0)



# # Step 5: Match truth-level quarks to jets based on proximity in eta-phi space

# # Label mapping and class names
# label_map = {0: 0, 4: 1, 5: 2}
# class_names = ["ujets", "cjets", "bjets"]

# # Initialize lists for storing matched jets
# matched_jet_pt = []
# matched_jet_eta = []
# matched_jet_flavour = []

# # Iterate over events
# for i in range(len(evevt)):
#     if len(jets[i]) == 0:
#         continue  # Skip if no jets in the event
#     # Get indices of quarks in this event
#     b_quark_indices = np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) == 5))[0]
#     c_quark_indices = np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) == 4))[0]
#     u_quark_indices = np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) <= 3))[0]

#     # Function to match jets to quarks
#     def match_jets_to_quarks(jet_vecs, quark_vecs, flavour):
#         for quark_vec in quark_vecs:
#             min_dr = float('inf')
#             matched_jet = None
#             for jet_vec in jet_vecs:
#                 dr = quark_vec.DeltaR(jet_vec)
#                 if dr < min_dr:
#                     min_dr = dr
#                     matched_jet = jet_vec
#             if matched_jet:
#                 matched_jet_pt.append(matched_jet.Pt())
#                 matched_jet_eta.append(matched_jet.Eta())
#                 matched_jet_flavour.append(label_map[flavour])

#     # Match jets to b-quarks
#     if len(b_quark_indices) > 0:
#         match_jets_to_quarks(jets[i], [truth_particles[i][idx] for idx in b_quark_indices], 5)

#     # Match jets to c-quarks
#     if len(c_quark_indices) > 0:
#         match_jets_to_quarks(jets[i], [truth_particles[i][idx] for idx in c_quark_indices], 4)

#     # Match jets to light quarks (u, d, s)
#     if len(u_quark_indices) > 0:
#         match_jets_to_quarks(jets[i], [truth_particles[i][idx] for idx in u_quark_indices], 0)
      

# %%
# Step 6: Match tracks to jets and calculate quantities for 'consts' dataset

# consts_data = []

# for i in range(len(evevt)):
#     event_consts = []
#     jet_vecs = jets[i]
#     trk_vecs = tracks[i]
#     for j, jet_vec in enumerate(jet_vecs):
#         jet_consts = []
#         for k, trk_vec in enumerate(trk_vecs):
#             if jet_vec.DeltaR(trk_vec) < 0.4:
#                 d0sig = np.sqrt(tscov[i][k][0]) if tscov[i][k][0] > 0 else 1e-6  # Avoid division by zero
#                 z0sig = np.sqrt(tscov[i][k][9]) if tscov[i][k][9] > 0 else 1e-6

#                 # Signed impact parameters
#                 theta = jet_vec.Phi() - trk_vec.Phi()
#                 sj = 1.0 if np.sin(theta) * tsdze[i][k] >= 0 else -1.0
#                 signed_2d_ip = np.abs(tsdze[i][k] / d0sig) * sj

#                 # 3D signed impact parameter
#                 d0z0 = np.sqrt(tsdze[i][k]**2 + tszze[i][k]**2)
#                 d0z0_sig = np.sqrt(d0sig**2 + z0sig**2)
#                 signed_3d_ip = np.abs(d0z0 / d0z0_sig) * sj

#                 # Determine particle type
#                 particle_type = determine_particle_type(mcpdg[i][k])
#                 is_gamma = particle_type == "gamma"
#                 is_neutral_had = particle_type == "neutral_had"
#                 is_electron = particle_type == "electron"
#                 is_muon = particle_type == "muon"
#                 is_charged_had = particle_type == "charged_had"

#                 const = (
#                     -1,  # truth_hadron_idx (if available, replace with correct index)
#                     -1,  # truth_vertex_idx (if available, replace with correct index)
#                     mcpdg[i][k],  # truth_origin_label
#                     True,  # valid
#                     is_gamma,
#                     is_neutral_had,
#                     is_electron,
#                     is_muon,
#                     is_charged_had,
#                     trk_charge[i][k],
#                     trk_vec.Phi() - jet_vec.Phi(),
#                     trk_vec.Eta() - jet_vec.Eta(),
#                     trk_vec.Pt() / jet_vec.Pt() if jet_vec.Pt() > 0 else 0,
#                     tsdze[i][k],
#                     tszze[i][k],
#                     jet_vec.DeltaR(trk_vec),
#                     signed_2d_ip,
#                     signed_3d_ip
#                 )
#                 jet_consts.append(const)
#         event_consts.extend(jet_consts)
#     consts_data.extend(event_consts)



# %%
# Step 7: Create structured arrays and define dataset shapes and chunk sizes

# Create structured array for jets
jets_data = np.zeros(len(matched_jet_pt), dtype=dtype_jets)
jets_data['pt'] = matched_jet_pt
jets_data['eta'] = matched_jet_eta
jets_data['flavour'] = matched_jet_flavour

# # Convert consts_data to structured array
# flat_consts_data = np.array(consts_data, dtype=dtype_consts)

# # Define dataset shapes and chunk sizes
# shape_consts = flat_consts_data.shape
# chunks_consts = (min(1000, shape_consts[0]),)


# %%
# Step 8: Create the HDF5 file and datasets
output_file = "/home/ssaini/dev/muonc/btagging/output_data/output_13Dec2024_v1.h5"

with h5py.File(output_file, "w") as f:
    # Create 'consts' dataset with LZF compression
    # dataset_consts = f.create_dataset(
    #     "consts",
    #     data=flat_consts_data,
    #     dtype=dtype_consts,
    #     chunks=chunks_consts,
    #     compression="lzf"
    # )

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
