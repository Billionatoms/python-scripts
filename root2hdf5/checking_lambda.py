# %%
# Importing the necessary libararies
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt


# Step 1: Open the ROOT file and extract data
f = uproot.open("/mnt/c/Users/Saurabh/cernbox/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root")
tree = f["LCTuple"]


# %%
# Track level information
tstnl = tree["tstnl"].array()     # tanλ, with λ = pitch angle

# %% 
def get_lambda(tan_lambda):
    return np.arctan(tan_lambda)

# %%
# Flatten the jagged array
tstnl_flat = ak.flatten(tstnl)

# Calculate lambda values
lambda_values = get_lambda(tstnl_flat)

# Convert to NumPy array
lambda_values_np = ak.to_numpy(lambda_values)




# %%
# plotting the lambda values
plt.hist(lambda_values_np, bins=100, range=(-2, 2))
plt.xlabel("λ")
plt.ylabel("Frequency")
plt.title("λ distribution")
plt.show()


# %%
