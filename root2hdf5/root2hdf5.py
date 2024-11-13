# %%
# Importing necessary libararies
import uproot
import h5py
import numpy as np

# %%
# Step 1: Open the ROOT file and extract data
f = uproot.open("/mnt/c/Users/Saurabh/cernbox/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root")
tree = f["LCTuple"]

# Extract relevant brnaches from the ROOT file
data = tree.arrays(library="np")    # Fetch data as numpy arrays

# %%
# Step 2: Process/match the objects (custom logic based on your needs)
# For example, match based on an event ID or some key
# Assuming 'evevt' is a key that we can use to match objects
event_ids = data['evevt']
jmox = data['jmox']
jmoy = data['jmoy']
jmoz = data['jmoz']
jmas = data["jmas"]
jene = data["jene"]


# %%
