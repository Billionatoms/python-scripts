# %%
import uproot
import h5py
import numpy as np

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
ntrk = tree["ntrk"].array()       # Number of tracks
trori = tree["trori"].array()     # Track calorimeter ID???
trtyp = tree["trtyp"].array()     # Track type???
trch2 = tree["trch2"].array()     # Chi^2 of the track fit
trndf = tree["trndf"].array()     # Number of degrees of freedom of the track fit
tredx = tree["tredx"].array()     # Track's dE/dx
trede = tree["trede"].array()     # Track's dE/dx error
trrih = tree["trrih"].array()     # Radius of inner most hit
trthn = tree["trthn"].array()     # ???
trthi = tree["trthi"].array()     # ????
trshn = tree["trshn"].array()     # Subdetector Hit number
trthd = tree["trthd"].array()     # ???
trnts = tree["trnts"].array()     # Track states size???
trfts = tree["trfts"].array()     # ???
trsip = tree["trsip"].array()     # Track state at IP
trsfh = tree["trsfh"].array()     # Track state at first state???
trslh = tree["trslh"].array()     # Track state at last hit
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
# Step 2: calculating pT, phi, eta 
# Calculate pT, phi, and eta values for jets
jpt = np.sqrt(jmox**2 + jmoy**2)
jphi = np.arctan2(jmoy, jmox)
jmo = np.sqrt(jmox**2 + jmoy**2 + jmoz**2)
epsilon = 1e-10  # A small value to prevent division by zero
jeta = 0.5 * np.log((jmo + jmoz) / (jmo - jmoz + epsilon))

# Calculate pT, phi, and eta values for truth-level quarks
mcpt = np.sqrt(mcmox**2 + mcmoy**2)
mcphi = np.arctan2(mcmoy, mcmox)
mcmo = np.sqrt(mcmox**2 + mcmoy**2 + mcmoz**2)
mceta = 0.5 * np.log((mcmo + mcmoz) / (mcmo - mcmoz + epsilon))


# %%
# Functions for delta_phi and delta_r
def delta_phi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    return np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)

# %%
# Step 3: Prepare the data for HDF5 format
data_to_save = {
    'evevt': evevt,
    'mcpt': mcpt,
    'jpt': jpt
}


# %%
# Step 4: Write data to an HDF5 file
with h5py.File("../../output_data/output_13Nov2024_v0.h5", "w") as hdf5_file:
    for key, data in data_to_save.items():
        hdf5_file.create_dataset(key, data=data)


print("Conversion complete. Data saved as a .h5 file")

# %%
