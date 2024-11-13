# %%
import h5py

# %%
def print_structure(name, obj):
    print(f"{name}:\n")
    if isinstance(obj, h5py.Group):
        print("Group\n")
    elif isinstance(obj, h5py.Dataset):
        print(f" Dataset, shape: {obj.shape}, dtype: {obj.dtype}\n")
    else:
        print(" Unkown type\n")
        
# Print attributes        
    for key, val in obj.attrs.items():
        print(f" Attribute: {key} = {val}\n")

# %%
# Open the hdf5 file
with h5py.File("/mnt/c/Users/Saurabh/cernbox/tutorial-Btagging-dataset/pp_output_train.h5") as file:
    print("HDF5 file structure:")
    file.visititems(print_structure)

# %%



