#!/usr/bin/env python
# coding: utf-8

# 1. Check meta data
# 2. Screen data (print some images from all direction)
# 3. Plot some time series

# %%
##### LOAD PACKAGES #####
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

##### PATHS TO DATA #####
sub = "01"
base_path = f"/home/td/fMRI_internship/raw/sub-{sub}/"
anat_path = f"{base_path}anat/sub-{sub}_T1w.nii"

task = "task-localizer"
func_path = f"{base_path}func/sub-{sub}_{task}_bold.nii.gz"

output_path = "/home/td/fMRI_internship/derivatives/data_screening/"
# %%
##### HEADER #####
anat = nib.load(anat_path)
print(anat.shape)
print(anat.header.get_zooms())
print(anat.header.get_xyzt_units())

func = nib.load(func_path)
print(func.shape)
print(func.header.get_zooms())
print(func.header.get_xyzt_units())

# %%
##### DATA #####
anat_data = anat.get_fdata()
print(type(anat_data))
print(anat_data.shape)

mid_voxel = anat_data[94:97, 125:128, 125:128]

slice_data = anat_data[50, :, :]
print(slice_data.shape)
plt.imshow(slice_data.T, cmap="gray", origin="lower")


# %%
##### PLOT #####
def plot_skim(data, slices, data_type="anat", fix_dim_value=1000):
    fig, axs = plt.subplots(
        slices.shape[0] // 4 + 1, 4, figsize=(10, 13), sharex=True, sharey=True
    )

    i = 0
    k = 0

    for s in slices:
        if data_type == "anat":
            slice_data = data[:, :, s]
        elif data_type == "func":
            slice_data = data[:, fix_dim_value, :, s]

        im = axs[i, k].imshow(slice_data.T, cmap="gray", origin="lower")
        axs[i, k].set_title(f"Slice {s}")
        axs[i, k].axis("off")

        if k + 1 < 4:
            k += 1
        else:
            k = 0
            i += 1

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.show()


# %%
##### PLOT ANAT #####
slices = np.arange(50, 200, 7)
plot_skim(anat_data, slices)

# %%
##### PLOT FUNC #####
func_data = func.get_fdata()
print(func_data.shape)

slices = np.arange(0, 500, 50)
plot_skim(func_data, slices, data_type="func", fix_dim_value=50)

# %%
##### VOXEL ACTIVITY #####
voxel = func_data[38, 30, 25, :]
plt.plot(voxel[:100], "o-")


from matplotlib import patches

fig, axes = plt.subplots(ncols=5, nrows=4, figsize=(20, 10))  # 20 timepoints
# Loop over the first 20 volumes/timepoints
for t, ax in enumerate(axes.flatten()):
    ax.imshow(func_data[:, 39, :, t].T, cmap="gray", origin="lower")  # index with t!
    rect = patches.Rectangle(
        (38, 25), 2, 2, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)
    ax.axis("off")
    ax.set_title("t = %i" % t, fontsize=20)
fig.tight_layout()

# %%
##### ANIMATION #####
from matplotlib.animation import FuncAnimation


def animate(data, data_type, plane, output_path, fix_dim_slice=0, task=None):
    """
    if data is 'func', take 2 image dimensions and time as 3. dim
    draw only that 2D image of choice over time to observe motion
    might be optimal to pick middle slide of each plane
    choose slice by setting fix_dim

    """
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        if data_type == "anat":
            if plane == "sagittal":
                ax.imshow(data[frame, :, :].T, cmap="gray", origin="lower")
            elif plane == "coronal":
                ax.imshow(data[:, frame, :].T, cmap="gray", origin="lower")
            elif plane == "axial":
                ax.imshow(data[:, :, frame].T, cmap="gray", origin="lower")
            ax.set_title(f"Frame {frame}")

        elif data_type == "func":
            if plane == "sagittal":
                ax.imshow(
                    data[fix_dim_slice, :, :, frame].T, cmap="gray", origin="lower"
                )
            elif plane == "coronal":
                ax.imshow(
                    data[:, fix_dim_slice, :, frame].T, cmap="gray", origin="lower"
                )
            elif plane == "axial":
                ax.imshow(
                    data[:, :, fix_dim_slice, frame].T, cmap="gray", origin="lower"
                )
            ax.set_title(f"Frame {frame}")

    if data_type == "anat":
        numb_frames = data.shape[{"sagittal": 0, "coronal": 1, "axial": 2}[plane]]
        file_name = f"{output_path}animated_{data_type}_{plane}.mp4"
    elif data_type == "func":
        numb_frames = data.shape[-1]
        file_name = f"{output_path}animated_{data_type}_{task}_{plane}.mp4"

    animation = FuncAnimation(fig, update, numb_frames)
    animation.save(file_name, writer="ffmpeg", fps=5)

    return f"Animation saved as {file_name}"


# %%
##### ANIMATION - EXAMPLE #####
# animate(
#     func_data,
#     data_type="func",
#     plane="sagittal",
#     output_path=output_path,
#     fix_dim_slice=32,
#     task="task-localizer",
# )

# %%
##### JSON #####
import json

json_path = f"{base_path}func/sub-{sub}_{task}_bold.json"
with open(json_path, "r") as file:
    content = file.read()
    json_data = json.loads(content)

json_data["time"]["samples"].keys()

# %%
##### TSV #####
import pandas as pd

tsv_path = f"{base_path}func/sub-{sub}_{task}_events.tsv"
tsv_data = pd.read_csv(tsv_path, sep="\t")
tsv_data

# %%
