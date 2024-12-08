# %%
# Importing the necessary libararies
import uproot
import h5py
import numpy as np
import awkward as ak

# %%
# Step 1: Open the ROOT file and extract data
f = uproot.open("/mnt/c/Users/Saurabh/cernbox/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root")
tree = f["LCTuple"]

# %%
# Event level information
evevt = tree["evevt"].array()      # Event number
evrun = tree["evrun"].array()      # Event run number
evwgt = tree["evwgt"].array()      # Event weight
evtim = tree["evtim"].array()      # Event time stamp
evsig = tree["evsig"].array()      # Event cross-section
evene = tree["evene"].array()      # Event energy
evpoe = tree["evpoe"].array()      # Event Polarization 1
evpop = tree["evpop"].array()      # Event Polarization 2
evnch = tree["evnch"].array()      # ???
evpro = tree["evpro"].array()      # Event process


# %%
# Truth level information
nmcp = tree["nmcp"].array()        # Number of truth particle
mcori = tree["mcori"].array()      # CollID???
mcpdg = tree["mcpdg"].array()      # Truth particle PDG
mcgst = tree["mcgst"].array()      # Truth particle generator status
mcsst = tree["mcsst"].array()      # Truth particle simulator status
mcvtx = tree["mcvtx"].array()      # Truth particle vertex x
mcvty = tree["mcvty"].array()      # Truth particle vertex y
mcvtz = tree["mcvtz"].array()      # Truth particle vertex z
mcepx = tree["mcepx"].array()      # Truth particle end point in x
mcepy = tree["mcepy"].array()      # Truth particle end point in y
mcepz = tree["mcepz"].array()      # Truth particle end point in z
mcmox = tree["mcmox"].array()      # Truth partcile momentumn in x-direction
mcmoy = tree["mcmoy"].array()      # Truth partcile momentumn in y-direction
mcmoz = tree["mcmoz"].array()      # Truth partcile momentumn in z-direction
mcmass = tree["mcmas"].array()     # Truth particle mass
mcene = tree["mcene"].array()      # Truth particle energy
mccha = tree["mccha"].array()      # Truth particle charge
mctim = tree["mctim"].array()      # Truth particle time???
mcspx = tree["mcspx"].array()      # Truth particle spin in x-direction
mcspy = tree["mcspy"].array()      # Truth particle spin in y-direction
mcspz = tree["mcspz"].array()      # Truth particle spin in z-direction
mccf0 = tree["mccf0"].array()      # Truth particle color flow 0
mccf1 = tree["mccf1"].array()      # Truth particle color flow 1
mcpa0 = tree["mcpa0"].array()      # Truth particle parent 0
mcpa1 = tree["mcpa1"].array()      # Truth particle parent 1
mcda0 = tree["mcda0"].array()      # Truth particle daughter 0
mcda1 = tree["mcda1"].array()      # Truth particle daughter 1
mcda2 = tree["mcda2"].array()      # Truth particle daughter 2
mcda3 = tree["mcda3"].array()      # Truth particle daughter 3
mcda4 = tree["mcda4"].array()      # Truth particle daughter 4


# %%
# Jet Information
njet = tree["njet"].array()        # Number of jets
jmox = tree["jmox"].array()        # Jet momentumn in x-direction
jmoy = tree["jmoy"].array()        # Jet momentumn in y-direction
jmoz = tree["jmoz"].array()        # Jet momentumn in z-direction
jmas = tree["jmas"].array()        # Jet's mass
jene = tree["jene"].array()        # Jet's energy
jcha = tree["jcha"].array()        # Jet's charge
jcov0 = tree["jcov0"].array()      # Jet covariance cov(d0, d0)
jcov1 = tree["jcov1"].array()      # Jet covariance cov(ϕ, d0)
jcov2 = tree["jcov2"].array()      # Jet covariance cov(ϕ, ϕ)
jcov3 = tree["jcov3"].array()      # Jet covariance cov(Ω, d0)
jcov4 = tree["jcov4"].array()      # Jet covariance cov(Ω, ϕ)
jcov5 = tree["jcov5"].array()      # Jet covariance cov(Ω, Ω)
jcov6 = tree["jcov6"].array()      # Jet covariance cov(z0, d0)
jcov7 = tree["jcov7"].array()      # Jet covariance cov(z0, ϕ)
jcov8 = tree["jcov8"].array()      # Jet covariance cov(z0, Ω)
jcov9 = tree["jcov9"].array()      # Jet covariance cov(z0, z0)


# %%
# Track level information
ntrk = tree["ntrk"].array()       # Total Number of tracks in the event
trori = tree["trori"].array()     # Origin of the track, typically representing the detector or simulation process.
trtyp = tree["trtyp"].array()     # Type of the track, categorizing its origin or purpose.
trch2 = tree["trch2"].array()     # Chi-squared of the track fit, indicating the quality of the track reconstruction
trndf = tree["trndf"].array()     # Number of degrees of freedom in the track fit
tredx = tree["tredx"].array()     # Track's dE/dx
trede = tree["trede"].array()     # Track's dE/dx error
trrih = tree["trrih"].array()     # Radius of inner most hit
trthn = tree["trthn"].array()     # ???
trthi = tree["trthi"].array()     # ????
trshn = tree["trshn"].array()     # Subdetector Hit number
trthd = tree["trthd"].array()     # ???
trnts = tree["trnts"].array()     # Track states size???
trfts = tree["trfts"].array()     # ???
trsip = tree["trsip"].array()     # Significance of the track's Impact Parameter(e.g., deviation from expected vertex)
trsfh = tree["trsfh"].array()     # First hit associated with the track
trslh = tree["trslh"].array()     # Last hit associated with the track
trsca = tree["trsca"].array()     # Track state at calorimeter
ntrst = tree["ntrst"].array()     # Number of tarck states
tsloc = tree["tsloc"].array()     # Track location
tsdze = tree["tsdze"].array()     # Track d0 parameter
tsphi = tree["tsphi"].array()     # Track phi parameter
tsome = tree["tsome"].array()     # Curvature $1/mm$ of the track. Proportional to magnetic field. (Ω)
tszze = tree["tszze"].array()     # Track z0 parameter
tstnl = tree["tstnl"].array()     # tanλ, with λ = pitch angle
tscov = tree["tscov"].array()     # Covariance matrix, stored as array of size 15 per hit
tsrpx = tree["tsrpx"].array()     # Global reference position of the track state in x direction
tsrpy = tree["tsrpy"].array()     # Global reference position of the track state in y direction
tsrpz = tree["tsrpz"].array()     # Global reference position of the track state in z direction



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


# %%
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


# %%
# Dataset 'jets'
dtype_jets = np.dtype([                             # Jets
    ("pt", np.float32),                             # pT of the jets
    ("eta", np.float32),                            # eta of the jets
    ("flavour", np.int32),                          # flavour of the jets
    ("flavour_label", np.int32)                     # flavour label of the jets
])





# %%
# Step 3: Calculations of variables not in root file

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

# Function to calculate ΔR
def delta_phi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    return np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

# Functions for delta_phi and delta_r
def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)

# Helper function to check if a variable is scalar
def is_scalar(value):
    return np.isscalar(value)

# %%
# calculating pt and eta values into a arrays
jpt = pt(jmox, jmoy)
jeta = eta(jmox, jmoy, jmoz)
jphi = phi(jmox, jmoy)

mcpt = pt(mcmox, mcmoy)
mceta = eta(mcmox, mcmoy, mcmoz)
mcphi = phi(mcmox, mcmoy)


# matched_jet_pt = []
# matched_jet_eta = []

# Define the label mapping and class names
label_map = {0: 0, 4: 1, 5: 2}
class_names = ["ujets", "cjets", "bjets"]

# %%
# Match truth-level quarks to jets based on proximity in eta-phi space
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
def match_jets_to_quarks(jet_eta, jet_phi, quark_eta, quark_phi, flavour_label):
    if len(jet_eta) == 0 or len(quark_eta) == 0:
        return  # Skip if there are no jets or no quarks
    distances = np.array([[delta_r(jet_eta[j], jet_phi[j], quark_eta[q], quark_phi[q])
                           for q in range(len(quark_eta))]
                          for j in range(len(jet_eta))])
    closest_jets = np.argmin(distances, axis=0)
    for j in closest_jets:
        matched_jet_pt.append(jpt[i][j])
        matched_jet_eta.append(jeta[i][j])
        matched_jet_flavour.append(label_map[flavour_label])
        
        
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
# Match truth-level quarks to jets based on proximity in eta-phi space
# for i in range(len(jpt)):
#    if len(jpt[i]) >= 2:
        # Find the indices of the first two b-quarks from the generator
#        b_quark_indices = np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) == 5))[0]
        
#        if len(b_quark_indices) >= 2:
#            b_quark_indices = b_quark_indices[:2]  # Get the first two indices

            # Compute delta R distances between jets and truth quarks
#            distances = np.array([[delta_r(jeta[i][j], jphi[i][j], mceta[i][b], mcphi[i][b])
#                                   for b in b_quark_indices]
#                                  for j in range(len(jpt[i]))])

            # Find the closest jets to each quark
#            closest_jets = np.argmin(distances, axis=0)
            
            
            # Store jet pT and eta values for the closest matches
#            for j in closest_jets:
#                matched_jet_pt.append(jpt[i][j])
#                matched_jet_eta.append(jeta[i][j])



# %%
# Create a structured array to hold the jets data
jets_data = np.empty(len(matched_jet_pt), dtype=dtype_jets)
jets_data['pt'] = matched_jet_pt
jets_data['eta'] = matched_jet_eta
jets_data['flavour'] = matched_jet_flavour

# Create a structured array to hold the track data
# consts_data = np.empty(len(ntrk), dtype=dtype_consts)
# consts_data['d0'] = tsdze
# consts_data['z0'] = tszze
# consts_data['phi_rel'] = tsphi
# consts_data['eta_rel'] = eta(trpx, trpy, trpz) # momentum of the track is not available in the root file
# consts_data['pt_frac'] = pt(trpx, trpy) / pt(jmox, jmoy) # pT fraction of the track is not available in the root file
# consts_data['charge'] = trcha # Charge of the truth particle is not available in the root file
# consts_data['dr'] =  
# consts_data['signed_2d_ip'] = trsip
# consts_data['signed_3d_ip'] = trsip

# %%
# Define dataset shapes and chunk sizes
shape_consts = (13500000, 50)
chunks_consts = (100, 50)
shape_hadrons = (13500000, 5)
chunks_hadrons = (100, 5)





# %%
# Create the HDF5 file and datasets
with h5py.File("/home/ssaini/dev/muonc/btagging/output_data/output_09Dec2024_v0.h5", "w") as f:
    # Create 'consts' dataset with LZF compression
    dataset_consts = f.create_dataset(
        "consts",
        shape=shape_consts,
        dtype=dtype_consts,
        chunks=chunks_consts,
        compression="lzf"
    )
    dataset_consts[...] = np.zeros(shape_consts, dtype=dtype_consts)  # Optionally initialize

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
    dataset_jets[...] = jets_data  # Optionally initialize
    
    # Set the 'flavour_label' attribute for the 'jets' dataset
    dataset_jets.attrs["flavour_label"] = np.array(class_names, dtype="S")

print("Conversion complete. Data saved as a .h5 file in /home/ssaini/dev/muonc/btagging/output_data")


# %%
