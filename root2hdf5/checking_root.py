# %%
import uproot


# %%
# Open the Root file
f = uproot.open("/eos/user/s/ssaini/muonc/btagging/samples/v0.0.1/mumu_H_bb_10TeV.00000.lctuple.root")


# %%
# List the keys in the file
print(f.keys())

# %%
# If the file contains a tree, you can open it and list its brnaches
tree = f["LCTuple"]
print(tree.keys())

# %%
# let's check a particular branch
branch = tree["evpro"]
print(branch)

# %%



