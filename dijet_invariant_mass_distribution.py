# %%
# importing necessary libararies
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
jene = tree["jene"].array()

# Truth level information
mcgst = tree["mcgst"].array()
mcpdg = tree["mcpdg"].array()
mcmox = tree["mcmox"].array()
mcmoy = tree["mcmoy"].array()
mcmoz = tree["mcmoz"].array()
mcmass = tree["mcmas"].array()
mcene = tree["mcene"].array()


# %%
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
# Function to calculate invariant mass
def invariant_mass(ene1, px1, py1, pz1, ene2, px2, py2, pz2):
    return np.sqrt((ene1 + ene2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)

# %%
# Functions for delta_phi and delta_r
def delta_phi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    return np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)


# %%
# Initialize lists for storing results
invariant_masses = []
jet_quark_diff = []
delta_r_values = []
jet_pt_values = []

# Match truth-level quarks to jets based on proximity in eta-phi space
for i in range(len(jpt)):
    if len(jpt[i]) >= 2:
        # Find the indices of the first two b-quarks from the generator
        b_quark_indices = np.where((mcgst[i] != 0) & (np.abs(mcpdg[i]) == 5))[0]
        
        if len(b_quark_indices) >= 2:
            b_quark_indices = b_quark_indices[:2]  # Get the first two indices

            # Compute delta R distances between jets and truth quarks
            distances = np.array([[delta_r(jeta[i][j], jphi[i][j], mceta[i][b], mcphi[i][b])
                                   for b in b_quark_indices]
                                  for j in range(len(jpt[i]))])

            # Find the closest jets to each quark
            closest_jets = np.argmin(distances, axis=0)

            # Calculate invariant mass using energies and momenta directly
            mass = invariant_mass(jene[i][closest_jets[0]], jmox[i][closest_jets[0]], jmoy[i][closest_jets[0]], jmoz[i][closest_jets[0]],
                                  jene[i][closest_jets[1]], jmox[i][closest_jets[1]], jmoy[i][closest_jets[1]], jmoz[i][closest_jets[1]])
            invariant_masses.append(mass)

            # Calculate pT difference
            jet_quark_diff.extend(mcpt[i][b_quark_indices] - jpt[i][closest_jets])

            # Store delta R values for the closest matches
            delta_r_values.extend([distances[closest_jets[j], j] for j in range(2)])

            # Store jet pT values
            jet_pt_values.extend(jpt[i])


# %%
# Function to calculate and display statistics on the plots
def plot_with_stats(ax, data, bins, range, xlabel, ylabel, title):
    ax.hist(data, bins=bins, range=range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Calculate statistics
    mean = np.mean(data)
    std_dev = np.std(data)
    entries = len(data)

    # Display statistics on the plot
    stats_text = f"Entries: {entries}\nMean: {mean:.2f}\nStd Dev: {std_dev:.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6))


# %%
# Plot the Histograms
# Plotting section
plt.figure(figsize=(12, 10))

# Plot 1: Invariant mass of two jets
ax1 = plt.subplot(2, 2, 1)
plot_with_stats(ax1, invariant_masses, bins=50, range=(0, 200), 
                xlabel="Invariant Mass (GeV)", ylabel="Number of Events", title="Invariant Mass of Two Jets")

# Plot 2: Difference in pT between truth-level quarks and jets
ax2 = plt.subplot(2, 2, 2)
plot_with_stats(ax2, jet_quark_diff, bins=50, range=(-100, 100), 
                xlabel="pT quark - pT jet (GeV)", ylabel="Number of Events", title="pT Difference Between Quarks and Jets")

# Plot 3: Delta R between closest jet and quark
ax3 = plt.subplot(2, 2, 3)
plot_with_stats(ax3, delta_r_values, bins=50, range=(0, 5), 
                xlabel="Delta R", ylabel="Number of Events", title="Delta R Between Closest Jet and Quark")

# Plot 4: Jet pT distribution
ax4 = plt.subplot(2, 2, 4)
plot_with_stats(ax4, jet_pt_values, bins=50, range=(0, 500), 
                xlabel="Jet pT (GeV)", ylabel="Number of Events", title="Jet pT Distribution")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# %%
