# %%
import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# %%
# Open the ROOT file and extract the branches
f = uproot.open("/eos/user/s/ssaini/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root")
tree = f["LCTuple"]
jmox = tree["jmox"].array()
jmoy = tree["jmoy"].array()
jmoz = tree["jmoz"].array()
jmas = tree["jmas"].array()

# Truth level information
mcgst = tree["mcgst"].array()
mcpdg = tree["mcpdg"].array()
mcmox = tree["mcmox"].array()
mcmoy = tree["mcmoy"].array()
mcmoz = tree["mcmoz"].array()
mcmass = tree["mcmas"].array()

# %%
# Calculate pT, phi, and eta values for jets
jpt = np.sqrt(jmox**2 +jmoy**2)
jphi = np.arctan2(jmoy, jmox)
jmo = np.sqrt(jmox**2 +jmoy**2 +jmoz**2)
epsilon = 1e-10  # A small value to prevent division by zero
jeta = 0.5 * np.log((jmo + jmoz) / (jmo - jmoz + epsilon))

# Calculate pT, phi, and eta values for truth-level quarks
mcpt = np.sqrt(mcmox**2 +mcmoy**2)
mcphi = np.arctan2(mcmoy, mcmox)
mcmo = np.sqrt(mcmox**2 +mcmoy**2 +mcmoz**2)
mceta = 0.5*np.log((mcmo + mcmoz) / (mcmo - mcmoz))

# %%
# Function to calculate invariant mass
def invariant_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
  p1 = np.array([pt1 * np.cosh(eta1), pt1 * np.cosh(phi1), pt1 * np.sin(phi1), mass1])
  p2 = np.array([pt2 * np.cosh(eta2), pt2 * np.cosh(phi2), pt2 * np.sin(phi2), mass2])
  return np.sqrt(np.sum((p1 + p2)**2))

# %%
def delta_phi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    return np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)

# %%
# Match truth-level quarks to jets based on proximity in eta-phi space
invariant_masses = []
jet_quark_diff = []
for i in range(len(jpt)):
  if len(jpt[i]) >=2:
    
    # Find the indices of the first two b-quarks from the generator
    b_quark_indices = np.where((mcgst[i] != 0) & (mcpdg[i] == 5))[0]
    
    # Debugging print to check quark indices
    print(f"Event {i}, b_quark_indices: {b_quark_indices}")

    if len(b_quark_indices) >= 2:
        # Compute delta R distances between jets and truth quarks
        distances = np.array([[delta_r(jeta[i][j], jphi[i][j], mceta[i][b], mcphi[i][b])
                                for b in b_quark_indices]
                                for j in range(len(jpt[i]))])
            
        # Debugging print to check distances
        print(f"Event {i}, distances: {distances}")

        # Find the closest jets to each quark
        closest_jets = np.argmin(distances, axis=0)

        # Debugging print to check closest jets
        print(f"Event {i}, closest_jets: {closest_jets}")

        # Calculate invariant mass for the two closest jets
        mass = invariant_mass(jpt[i][closest_jets[0]], jeta[i][closest_jets[0]], jphi[i][closest_jets[0]], jmas[i][closest_jets[0]],
                              jpt[i][closest_jets[1]], jeta[i][closest_jets[1]], jphi[i][closest_jets[1]], jmas[i][closest_jets[1]])
        invariant_masses.append(mass)

        # Calculate pT difference
        jet_quark_diff.extend(mcpt[i][b_quark_indices] - jpt[i][closest_jets])

# %%
# Plot the Histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(invariant_masses, bins=50, range=(0, 600))
plt.ylabel("Number of Events")
plt.xlabel("Invariant Mass (GeV)")
plt.title("Invariant mass of two jets")
plt.subplot(1, 2, 2)
plt.hist(jet_quark_diff, bins=50, range=(-100, 100))
plt.ylabel("Number of Events")
plt.xlabel("pT Difference (GeV)")
plt.title("Difference in pT between truth-level quarks and jets")
plt.tight_layout()
plt.show()
