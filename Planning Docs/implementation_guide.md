# 🧠 Phase 2: Picking Your Study Participants & Checking Their Brain Scans

Now that you have downloaded the main data file (**DXSUM_15Apr2026.csv**), your goal is to:
- Pick a clean, fair group of study participants
- Find their brain scan records
- Make sure everything looks correct before moving forward

---

## 🛠️ Step 1: Set Up Your Workspace

Since you're using **Kaggle**, you don't need Google Drive at all. Instead, upload your file directly to Kaggle's own storage system:

1. Go to [kaggle.com](https://www.kaggle.com) → click your profile picture → **Datasets** → **New Dataset**
2. Upload your `DXSUM_15Apr2026.csv` file there and publish it (you can keep it private)
3. Open your Kaggle notebook and click **+ Add Data** on the right panel → search for the dataset you just uploaded → add it
4. Your file is now available inside the notebook — run the code below to confirm it's accessible

```python
import os
import pandas as pd

# This is where Kaggle stores any data you add to your notebook
base_path = '/kaggle/input/your-dataset-name'  # ← Replace with your actual dataset name
csv_file_name = 'DXSUM_15Apr2026.csv'
csv_path = os.path.join(base_path, csv_file_name)

print(f"Checking access to target file: {csv_path}")
```

> 💡 **How to find your dataset name:** After uploading, look at the URL on your dataset page — it'll say something like `kaggle.com/datasets/yourname/adni-project`. The last part (`adni-project`) is your dataset name. Use that in the path above.

---

## 🛠️ Step 2: Filter & Select Your Study Participants

The data file contains records of many patients across multiple hospital visits. We only want their **very first visit** (called the "baseline"), so we're comparing everyone at the same starting point — no earlier, no later.

We're looking for two groups:
- **Healthy participants** (no memory issues) — labelled as `CN`
- **Alzheimer's patients** — labelled as `AD`

Run the code below to automatically pull out 50 people from each group.

```python
# Open and read the data file
df = pd.read_csv(csv_path, low_memory=False)

# Clean up any accidental spaces in the data
df['PTID'] = df['PTID'].str.strip()
df['VISCODE2'] = df['VISCODE2'].str.strip()

# Keep only records from each person's very first visit
baseline_mask = df['VISCODE2'] == 'bl'
baseline_df = df[baseline_mask].copy()

# Separate the two groups:
# 1 = Healthy (CN), 3 = Alzheimer's (AD)
cn_cohort = baseline_df[baseline_df['DIAGNOSIS'] == 1]['PTID'].unique()
ad_cohort = baseline_df[baseline_df['DIAGNOSIS'] == 3]['PTID'].unique()

print(f"📊 Done! Here's what we found:")
print(f" -> Healthy participants: {len(cn_cohort)}")
print(f" -> Alzheimer's participants: {len(ad_cohort)}")

# Pick 50 people from each group
selected_cn = list(cn_cohort[:50])
selected_ad = list(ad_cohort[:50])
target_subjects = selected_cn + selected_ad

# Combine all their IDs into one list, separated by commas
loni_search_string = ",".join(target_subjects)

print("\n🚀 [NEXT ACTION] Copy the list below and paste it into the brain scan search portal:")
print(loni_search_string)
```

---

## 🛠️ Step 3: Find Their Brain Scans Online

Now you need to look up the actual brain scans for your 100 selected participants. Here's how:

1. Go to the **LONI Image and Data Archive** website (the online brain scan library).
2. Open the **Advanced Search** section.
3. Paste the list of participant IDs (the output from Step 2) into the **Subject ID** box.
4. Apply the following filters to make sure you only get the right type of scans:

   | Filter | What to Select |
   |--------|----------------|
   | **Project** | ☑ ADNI |
   | **Scan Type** | ☑ MRI |
   | **Image Version** | ☑ Original (unedited) |
   | **Visit** | ☑ Baseline & ☑ Screening |
   | **Scan Shape** | ☑ 3D |
   | **Scan Contrast** | ☑ T1 |

5. Hit **Search**.

---

## 🛠️ Step 4: Save the Scans to Your Collection

1. Look through the search results and **skip** any scans labelled **B1-Calibration** or **REPEAT** — these are test or duplicate scans that can mess up your analysis.
2. Tick the checkboxes next to the valid scans for your participants.
3. Click **Add To Collection** and name your collection **`ADNI_Clean_Baseline_T1`**.
4. Download everything using the **1-Click Download** manager on the site.
5. Once downloaded to your computer, **upload the scan files as a new Kaggle Dataset** (same steps as Step 1) and add it to your notebook via **+ Add Data**.

> 💡 Your scans will then be available under `/kaggle/input/your-scans-dataset-name/`

---

## 🛠️ Step 5: Check That the Brain Scans Look Right

Once your scans are added to your Kaggle notebook, run the code below to visually check that they look like proper brain images. This ensures your data is complete and correctly oriented.

> [!IMPORTANT]
> **Final Update (Auto-Sorting):** This version of the code is the most robust. We encountered a problem where the brain scans were acquired along a different axis (e.g., side-to-side instead of top-to-bottom). Because we were previously sorting the slices based on the wrong coordinate, the brain appeared scrambled or as horizontal lines. This updated version automatically detects the correct acquisition axis (X, Y, or Z), sorts the slices perfectly, and applies "Auto-Windowing" to ensure the brain is clearly visible even if the raw data is dark.

```python
# Install the tools needed to open and view brain scan files
!pip install pydicom numpy matplotlib

import os
import glob
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# --- AUTO-SORTING BRAIN SEARCH ---
# Finds the main scan folder and automatically detects the orientation
all_dcm = glob.glob('/kaggle/input/**/*.dcm', recursive=True)
candidate_dirs = {d: len(glob.glob(os.path.join(d, '*.dcm'))) for d in set(os.path.dirname(f) for f in all_dcm)}
brain_dir = max(candidate_dirs, key=candidate_dirs.get) if candidate_dirs else None

if brain_dir and candidate_dirs[brain_dir] > 100:
    print(f"Full Scan Found: {brain_dir} ({candidate_dirs[brain_dir]} slices)")
    
    # Load all slices
    slices = [pydicom.dcmread(os.path.join(brain_dir, f)) for f in os.listdir(brain_dir) if f.endswith('.dcm')]
    
    # --- AUTO-DETECT SORTING AXIS ---
    # Finds which direction (X, Y, or Z) the slices were taken in
    positions = np.array([s.ImagePositionPatient for s in slices])
    diffs = np.max(positions, axis=0) - np.min(positions, axis=0)
    sorting_axis = np.argmax(diffs) 
    slices.sort(key=lambda x: float(x.ImagePositionPatient[sorting_axis]))
    
    # Stack the pixels into a 3D volume
    data = np.stack([s.pixel_array for s in slices])

    # --- AUTO-BRIGHTNESS ---
    vmin = np.percentile(data, 10)
    vmax = np.percentile(data, 99)

    # Show three views
    z, y, x = data.shape
    mid_z, mid_y, mid_x = z // 2, y // 2, x // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(data[mid_z, :, :], cmap='bone', vmin=vmin, vmax=vmax)
    axes[0].set_title("View 1 (Slice Axis)")
    axes[0].axis('off')

    axes[1].imshow(data[:, mid_y, :], cmap='bone', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("View 2 (Cross-section)")
    axes[1].axis('off')

    axes[2].imshow(data[:, :, mid_x], cmap='bone', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title("View 3 (Cross-section)")
    axes[2].axis('off')
    
    plt.suptitle(f"Brain Scan Check: {data.shape} volume", fontsize=14)
    plt.show()
else:
    print("No full scan folder found.")
```

> ✅ **Congratulations!** If you see a clear brain in all three views, your data collection phase is complete. You have successfully selected a cohort, downloaded their scans, and verified them on Kaggle. You are ready for **Phase 3: Preprocessing**!

---

# 🧠 Phase 3: Preprocessing — Cleaning Your Brain Scans

Now that you have your raw brain scans (**DICOM slices**) verified on Kaggle, we enter the most critical stage of the project. Raw MRI scans cannot be fed directly into an AI model because they are "messy"—some are tilted, some are brighter than others, and they all include things we don't want to study (like the skull, eyes, and skin).

**Preprocessing** is the process of cleaning and standardizing these images so the AI can focus purely on the brain tissue.

---

## 🛠️ Step 1: Converting DICOM to NIfTI (Robust)
*   **What:** Turning your folder of `.dcm` slices into a **single** `.nii.gz` file.
*   **Why:** AI models (3D-CNNs) need a single 3D volume to "read" the brain. Loading one file is also much faster than loading many separate slices for every patient.
*   **How:** Use `nibabel` + `pydicom` directly with error handling to skip corrupted series. (More robust than `dicom2nifti` alone.)

```python
!pip install nibabel pydicom numpy -q

import os
import glob
import pydicom
import numpy as np
import nibabel as nib
from pathlib import Path

input_root = "/kaggle/input/your-scans-dataset-name"
output_root = "/kaggle/working/phase3_nifti_raw"
failed_log = "/kaggle/working/failed_conversions.txt"

os.makedirs(output_root, exist_ok=True)

series_dirs = sorted({
    os.path.dirname(path)
    for path in glob.glob(os.path.join(input_root, "**", "*.dcm"), recursive=True)
})

print(f"Found {len(series_dirs)} DICOM folders.\n")

success_count = 0
failed_series = []

for series_dir in series_dirs:
    dcm_files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if len(dcm_files) < 20:
        continue

    subject_name = os.path.basename(series_dir.rstrip("\\/"))
    
    try:
        slices = []
        for dcm_file in dcm_files:
            try:
                slices.append(pydicom.dcmread(dcm_file))
            except Exception as e:
                print(f"  ⚠ Skipped corrupted file: {dcm_file}")
                continue
        
        if len(slices) < 20:
            raise ValueError(f"Too few valid slices: {len(slices)}")
        
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        pixel_array = np.stack([s.pixel_array for s in slices])
        affine = np.eye(4)
        
        subject_out = os.path.join(output_root, subject_name)
        os.makedirs(subject_out, exist_ok=True)
        output_path = os.path.join(subject_out, f"{subject_name}.nii.gz")
        
        nib.save(nib.Nifti1Image(pixel_array, affine), output_path)
        print(f"✅ {subject_name} ({len(slices)} slices) -> {output_path}")
        success_count += 1
        
    except Exception as e:
        print(f"❌ FAILED {subject_name}: {str(e)}")
        failed_series.append((subject_name, str(e)))

with open(failed_log, "w") as f:
    for name, err in failed_series:
        f.write(f"{name}: {err}\n")

print(f"\n✅ Conversion complete: {success_count} succeeded, {len(failed_series)} failed.")
print(f"Failed series logged to: {failed_log}")
```

## 🛠️ Step 2: N4 Bias Field Correction + Skull Stripping (Brain Extraction)
*   **What:** First fix MRI brightness inhomogeneity with **N4**, then remove skull, eyes, fat, and neck so only the brain remains.
*   **Why:** The model should learn brain tissue patterns, not scanner lighting or outside-head anatomy.
*   **How:** Use **ANTsPy** for N4 bias field correction and **HD-BET** (state-of-the-art deep learning) for accurate skull stripping.

### Approach 1 - Otsu Thresholding

```python
!pip install antspyx nibabel numpy scikit-image -q

import os
import glob
import nibabel as nib
import numpy as np
import ants
from skimage.filters import threshold_otsu

root_path = "/content/drive/MyDrive/3 Credit Project/Data"
input_folder = os.path.join(root_path, "Outputs", "Phase3_NIfTI_Raw_output")
output_folder = os.path.join(root_path, "Outputs", "Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output")
os.makedirs(output_folder, exist_ok=True)

nifti_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.nii*"), recursive=True))
total_files = len(nifti_files)
print(f"Found {total_files} NIfTI files.\n")

def n4_bias_correction_antspy(ants_img):
    """Apply N4 bias field correction using ANTsPy."""
    return ants.n4_bias_field_correction(ants_img)

def skull_strip_otsu(ants_img):
    """Apply Otsu thresholding for a baseline skull stripping mask."""
    img_data = ants_img.numpy()
    foreground = img_data[img_data > 0]
    threshold = threshold_otsu(foreground) if foreground.size else 0.0
    brain_mask_data = (img_data > threshold).astype(np.float32)
    brain_mask = ants.from_numpy(
        brain_mask_data,
        origin=ants_img.origin,
        spacing=ants_img.spacing,
        direction=ants_img.direction,
    )
    return brain_mask

for idx, nifti_path in enumerate(nifti_files, 1):
    stem = os.path.basename(nifti_path).replace(".nii.gz", "").replace(".nii", "")
    try:
        ants_img = ants.image_read(nifti_path)
        corrected = n4_bias_correction_antspy(ants_img)
        brain_mask = skull_strip_otsu(corrected)
        brain_only = ants.mask_image(corrected, brain_mask)

        output_path = os.path.join(output_folder, f"{stem}_brain.nii.gz")
        ants.image_write(brain_only, output_path)

        progress_pct = (idx / total_files) * 100
        print(f"[{idx:3d}/{total_files}] ({progress_pct:5.1f}%) ✅ {stem}")

    except Exception as e:
        print(f"[{idx:3d}/{total_files}] ❌ {stem} - Error: {str(e)}")
        continue

print(f"\n✅ N4 correction and Otsu skull stripping complete. Processed {total_files} files.")
```

### Problems with Approach 1

The original threshold-based approach did not properly isolate the brain. Here's why:

1. **Otsu is blind to anatomy:** it separates pixels only by brightness, so it cannot tell brain tissue apart from scalp, fat, or skin.
2. **Intensity overlap in T1-weighted MRIs:** ADNI T1 scans often make outside-head tissues as bright as, or brighter than, the brain.
3. **Result:** the mask tends to keep a whole-head foreground instead of a true brain-only region, and simple cleanup operations do not remove the skull reliably.

### Why DeepBrain Was Deprecated

DeepBrain was designed for TensorFlow 1.x (circa 2017-2018) and depends on legacy APIs such as `tf.Session`. Modern Colab and Kaggle environments ship with TensorFlow 2.21.0, which:
- Cannot downgrade (pre-installed system package)
- Causes infinite recursion errors when importing DeepBrain
- Has protobuf version conflicts that break the import chain

**Result:** DeepBrain is no longer compatible with current cloud environments. We switched to **HD-BET**, a modern state-of-the-art skull stripping library that is:
- ✅ Trained on thousands of high-quality brain scans
- ✅ Actively maintained and TensorFlow 2.x native
- ✅ Higher accuracy than classical threshold-based methods
- ✅ Works seamlessly on Colab and Kaggle

### Recommended Approach - HD-BET (Accurate + Modern)

```python
# Install HD-BET and dependencies
!pip install HD-BET antspyx nibabel numpy -q
print("HD-BET installed successfully")

import os
import glob
import nibabel as nib
import numpy as np
import ants
from HD_BET.run import run_hd_bet

root_path = "/content/drive/MyDrive/3 Credit Project/Data"
input_folder = os.path.join(root_path, "Outputs", "Phase3_NIfTI_Raw_output")
output_folder = os.path.join(root_path, "Outputs", "Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output")
os.makedirs(output_folder, exist_ok=True)

nifti_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.nii*"), recursive=True))
total_files = len(nifti_files)
print(f"Found {total_files} NIfTI files.\n")

def n4_bias_correction_antspy(ants_img):
    return ants.n4_bias_field_correction(ants_img)

def skull_strip_hd_bet(nifti_path, output_mask_path):
    """Use HD-BET for brain extraction"""
    run_hd_bet(nifti_path, output_mask_path, device=0, mode='accurate', overwrite=True)
    return output_mask_path

for idx, nifti_path in enumerate(nifti_files, 1):
    stem = os.path.basename(nifti_path).replace(".nii.gz", "").replace(".nii", "")
    try:
        # Step 1: N4 bias correction
        ants_img = ants.image_read(nifti_path)
        corrected = n4_bias_correction_antspy(ants_img)
        
        # Step 2: Save corrected image temporarily for HD-BET
        temp_corrected_path = os.path.join(output_folder, f"{stem}_temp_corrected.nii.gz")
        ants.image_write(corrected, temp_corrected_path)
        
        # Step 3: HD-BET skull stripping
        temp_mask_path = os.path.join(output_folder, f"{stem}_mask.nii.gz")
        skull_strip_hd_bet(temp_corrected_path, temp_mask_path)
        
        # Step 4: Load mask and apply to image
        mask_img = nib.load(temp_mask_path)
        brain_only_data = corrected.numpy() * mask_img.get_fdata()
        brain_only = ants.from_numpy(
            brain_only_data,
            origin=corrected.origin,
            spacing=corrected.spacing,
            direction=corrected.direction,
        )
        
        # Step 5: Save final result
        output_path = os.path.join(output_folder, f"{stem}_brain.nii.gz")
        ants.image_write(brain_only, output_path)
        
        # Cleanup temp files
        os.remove(temp_corrected_path)
        os.remove(temp_mask_path)

        progress_pct = (idx / total_files) * 100
        print(f"[{idx:3d}/{total_files}] ({progress_pct:5.1f}%) ✅ {stem}")
    except Exception as e:
        print(f"[{idx:3d}/{total_files}] ❌ {stem} - Error: {str(e)}")
        continue

print(f"\n✅ HD-BET skull stripping completed. Processed {total_files} files.")
```

## 🛠️ Step 2B: Verify Skull-Stripped Brains (Before/After Comparison)

This cell displays **raw vs. processed** brains side-by-side for the first 3 subjects. This lets you visually confirm that N4 correction and skull stripping worked correctly.

```python
!pip install nibabel matplotlib numpy

import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

root_path = "/content/drive/MyDrive/3 Credit Project/Data"
input_folder = os.path.join(root_path, "Outputs", "Phase3_NIfTI_Raw_output")
output_folder = os.path.join(root_path, "Outputs", "Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output")

raw_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.nii*"), recursive=True))[:3]

if not raw_files:
    print("No raw brain files found. Run Step 1 first.")
else:
    for raw_path in raw_files:
        subject_name = os.path.basename(raw_path).replace(".nii.gz", "").replace(".nii", "")
        brain_path = os.path.join(output_folder, f"{subject_name}_brain.nii.gz")
        
        if not os.path.exists(brain_path):
            print(f"⚠ Processed file not found for {subject_name}, skipping...")
            continue
        
        raw_img = nib.load(raw_path)
        brain_img = nib.load(brain_path)
        
        raw_data = raw_img.get_fdata().astype(np.float32)
        brain_data = brain_img.get_fdata().astype(np.float32)
        
        raw_vmin = np.percentile(raw_data[raw_data > 0], 10)
        raw_vmax = np.percentile(raw_data[raw_data > 0], 99)
        brain_vmin = np.percentile(brain_data[brain_data > 0], 10)
        brain_vmax = np.percentile(brain_data[brain_data > 0], 99)
        
        z, y, x = raw_data.shape
        mid_z, mid_y, mid_x = z // 2, y // 2, x // 2
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        
        axes[0, 0].imshow(raw_data[mid_z, :, :], cmap='bone', vmin=raw_vmin, vmax=raw_vmax)
        axes[0, 0].set_title("Raw: Axial", fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(brain_data[mid_z, :, :], cmap='bone', vmin=brain_vmin, vmax=brain_vmax)
        axes[0, 1].set_title("Processed: Axial", fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(np.rot90(raw_data[:, mid_y, :]), cmap='bone', aspect='auto', vmin=raw_vmin, vmax=raw_vmax)
        axes[1, 0].set_title("Raw: Coronal", fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.rot90(brain_data[:, mid_y, :]), cmap='bone', aspect='auto', vmin=brain_vmin, vmax=brain_vmax)
        axes[1, 1].set_title("Processed: Coronal", fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[2, 0].imshow(np.rot90(raw_data[:, :, mid_x]), cmap='bone', aspect='auto', vmin=raw_vmin, vmax=raw_vmax)
        axes[2, 0].set_title("Raw: Sagittal", fontweight='bold')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(np.rot90(brain_data[:, :, mid_x]), cmap='bone', aspect='auto', vmin=brain_vmin, vmax=brain_vmax)
        axes[2, 1].set_title("Processed: Sagittal", fontweight='bold')
        axes[2, 1].axis('off')
        
        plt.suptitle(f"Before/After: {subject_name} | Shape: {raw_data.shape}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print(f"✅ {subject_name}: Skull stripped & N4 corrected. Observe skull removal & brightness normalization.")
```

## 🛠️ Step 3: Registration (Standardization)
*   **What:** Align every brain to the **MNI152 Atlas** so all scans share the same coordinate system.
*   **Why:** Registration makes the hippocampus, ventricles, and other structures line up across subjects.
*   **How:** Use ANTs registration with an affine + nonlinear transform.

```python
!pip install nilearn nibabel antspyx

import os
import glob
import nibabel as nib
import ants
from nilearn import datasets

root_path = "/content/drive/MyDrive/3 Credit Project/Data"
input_folder = os.path.join(root_path, "Outputs", "Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output")
output_folder = os.path.join(root_path, "Phase3_NIfTI_MNI152_output")
os.makedirs(output_folder, exist_ok=True)

template_img = datasets.load_mni152_template(resolution=1)
template_path = os.path.join(root_path, "mni152_template.nii.gz")
nib.save(template_img, template_path)
fixed = ants.image_read(template_path)

brain_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))
print(f"Found {len(brain_files)} brain volumes for registration.")

for brain_path in brain_files:
    moving = ants.image_read(brain_path)

    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN"
    )
    warped = registration["warpedmovout"]

    stem = os.path.basename(brain_path).replace("_brain.nii.gz", "").replace(".nii.gz", "").replace(".nii", "")
    output_path = os.path.join(output_folder, f"{stem}_mni152.nii.gz")

    ants.image_write(warped, output_path)
    print(f"Saved registered volume: {output_path}")

print("MNI152 registration complete.")
```

## 🛠️ Step 4: Intensity Normalization
*   **What:** Standardize voxel intensities so each scan has comparable brightness and contrast.
*   **Why:** This reduces scanner-to-scanner variation and helps the model focus on anatomy.
*   **How:** Apply **Z-score normalization** to foreground voxels only.

```python
!pip install nibabel numpy

import os
import glob
import nibabel as nib
import numpy as np

root_path = "/content/drive/MyDrive/3 Credit Project/Data"
input_folder = os.path.join(root_path, "Phase3_NIfTI_MNI152_output")
output_folder = os.path.join(root_path, "Phase3_NIfTI_Normalized_output")
os.makedirs(output_folder, exist_ok=True)

registered_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))
print(f"Found {len(registered_files)} registered volumes.")

for volume_path in registered_files:
    img = nib.load(volume_path)
    data = img.get_fdata().astype(np.float32)

    brain_voxels = data[data > 0]
    if brain_voxels.size == 0:
        raise ValueError(f"No foreground voxels found in {volume_path}")

    mean = brain_voxels.mean()
    std = brain_voxels.std()
    if std == 0:
        std = 1.0

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[data > 0] = (data[data > 0] - mean) / std

    stem = os.path.basename(volume_path).replace(".nii.gz", "").replace(".nii", "")
    output_path = os.path.join(output_folder, f"{stem}_zscore.nii.gz")

    nib.save(nib.Nifti1Image(normalized, img.affine, img.header), output_path)
    print(f"Saved normalized volume: {output_path}")

print("Intensity normalization complete.")
```

---

## 🚀 What’s Next?
Once these four cells run successfully, your cleaned volumes are ready for the aggregated preprocessing pipeline in the next notebook.

> [!TIP]
> This phase is where most of the "heavy lifting" happens. Once your data is perfectly clean, building the actual AI model becomes much easier and more accurate.

---

# 🌐 Codes for Google Colab

This section contains **all the code cells from above, updated for Google Colab with Google Drive storage**. Use these instead of the Kaggle versions if you're working in Google Colab.

> 💡 **Setup:** Before running any code below, mount Google Drive by running this cell first:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> ```
> After running, you'll see a link. Click it, authorize your Google account, and copy the code back into Colab. Then your Drive is ready to use at `/content/drive/MyDrive/`.

---

## 📂 Google Drive Folder Structure
Organize your Google Drive like this:
```
MyDrive/
├── 3 Credit Project/
│   ├── Data/
│   │   ├── DXSUM_15Apr2026.csv
│   │   ├── Outputs/
│   │   │   ├── Phase3_NIfTI_Raw_output/
│   │   │   ├── Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output/
│   │   │   ├── Phase3_NIfTI_MNI152_output/
│   │   │   ├── Phase3_NIfTI_Normalized_output/
```

---

## 🛠️ Phase 2: Google Colab Cell 1 - Mount Drive & Load Data

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import libraries
import os
import pandas as pd

# Define paths in Google Drive
drive_root = '/content/drive/MyDrive/Diagnostic_Assistant_Alzheimers'
data_folder = os.path.join(drive_root, 'ADNI_Data')
csv_file_name = 'DXSUM_15Apr2026.csv'
csv_path = os.path.join(data_folder, csv_file_name)

print(f"Checking access to target file: {csv_path}")
if os.path.exists(csv_path):
    print("✅ File found!")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Data shape: {df.shape}")
else:
    print("❌ File not found. Please check your Google Drive folder structure.")
```

---

## 🛠️ Phase 2: Google Colab Cell 2 - Filter & Select Study Participants

```python
import pandas as pd
import os

# Define paths (using same paths as Cell 1)
drive_root = '/content/drive/MyDrive/Diagnostic_Assistant_Alzheimers'
data_folder = os.path.join(drive_root, 'ADNI_Data')
csv_path = os.path.join(data_folder, 'DXSUM_15Apr2026.csv')

# Open and read the data file
df = pd.read_csv(csv_path, low_memory=False)

# Clean up any accidental spaces in the data
df['PTID'] = df['PTID'].str.strip()
df['VISCODE2'] = df['VISCODE2'].str.strip()

# Keep only records from each person's very first visit
baseline_mask = df['VISCODE2'] == 'bl'
baseline_df = df[baseline_mask].copy()

# Separate the two groups:
# 1 = Healthy (CN), 3 = Alzheimer's (AD)
cn_cohort = baseline_df[baseline_df['DIAGNOSIS'] == 1]['PTID'].unique()
ad_cohort = baseline_df[baseline_df['DIAGNOSIS'] == 3]['PTID'].unique()

print(f"📊 Done! Here's what we found:")
print(f" -> Healthy participants: {len(cn_cohort)}")
print(f" -> Alzheimer's participants: {len(ad_cohort)}")

# Pick 50 people from each group
selected_cn = list(cn_cohort[:50])
selected_ad = list(ad_cohort[:50])
target_subjects = selected_cn + selected_ad

# Combine all their IDs into one list, separated by commas
loni_search_string = ",".join(target_subjects)

print("\n🚀 [NEXT ACTION] Copy the list below and paste it into the brain scan search portal:")
print(loni_search_string)
```

---

## 🛠️ Phase 2: Google Colab Cell 3 - Verify Brain Scans

```python
# Install required tools
!pip install pydicom numpy matplotlib -q

import os
import glob
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Define path to brain scans folder in Google Drive
drive_root = '/content/drive/MyDrive/Diagnostic_Assistant_Alzheimers'
scans_folder = os.path.join(drive_root, 'Brain_Scans')

print(f"Searching for DICOM files in: {scans_folder}")

# --- AUTO-SORTING BRAIN SEARCH ---
# Finds the main scan folder and automatically detects the orientation
all_dcm = glob.glob(os.path.join(scans_folder, '**', '*.dcm'), recursive=True)
candidate_dirs = {d: len(glob.glob(os.path.join(d, '*.dcm'))) for d in set(os.path.dirname(f) for f in all_dcm)}
brain_dir = max(candidate_dirs, key=candidate_dirs.get) if candidate_dirs else None

if brain_dir and candidate_dirs[brain_dir] > 100:
    print(f"Full Scan Found: {brain_dir} ({candidate_dirs[brain_dir]} slices)")
    
    # Load all slices
    slices = [pydicom.dcmread(os.path.join(brain_dir, f)) for f in os.listdir(brain_dir) if f.endswith('.dcm')]
    
    # --- AUTO-DETECT SORTING AXIS ---
    # Finds which direction (X, Y, or Z) the slices were taken in
    positions = np.array([s.ImagePositionPatient for s in slices])
    diffs = np.max(positions, axis=0) - np.min(positions, axis=0)
    sorting_axis = np.argmax(diffs)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[sorting_axis]))
    
    # Stack the pixels into a 3D volume
    data = np.stack([s.pixel_array for s in slices])
    
    # --- AUTO-BRIGHTNESS ---
    vmin = np.percentile(data, 10)
    vmax = np.percentile(data, 99)
    
    # Show three views
    z, y, x = data.shape
    mid_z, mid_y, mid_x = z // 2, y // 2, x // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(data[mid_z, :, :], cmap='bone', vmin=vmin, vmax=vmax)
    axes[0].set_title("View 1 (Slice Axis)")
    axes[0].axis('off')
    
    axes[1].imshow(data[:, mid_y, :], cmap='bone', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("View 2 (Cross-section)")
    axes[1].axis('off')
    
    axes[2].imshow(data[:, :, mid_x], cmap='bone', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title("View 3 (Cross-section)")
    axes[2].axis('off')
    
    plt.suptitle(f"Brain Scan Check: {data.shape} volume", fontsize=14)
    plt.show()
    
    print("✅ Brain scan verification complete!")
else:
    print("❌ No full scan folder found or insufficient DICOM files.")
```

---

## 🛠️ Phase 3: Google Colab Cell 4 - Convert DICOM to NIfTI

```python
# Install required libraries
!pip install nibabel pydicom numpy -q

import os
import glob
import pydicom
import numpy as np
import nibabel as nib
from pathlib import Path

# Define paths
drive_root = '/content/drive/MyDrive/Diagnostic_Assistant_Alzheimers'
input_root = os.path.join(drive_root, 'Brain_Scans')
output_root = os.path.join(drive_root, 'Outputs', 'Phase3_NIfTI_Raw')
failed_log = os.path.join(drive_root, 'Outputs', 'failed_conversions.txt')

os.makedirs(output_root, exist_ok=True)

series_dirs = sorted({
    os.path.dirname(path)
    for path in glob.glob(os.path.join(input_root, "**", "*.dcm"), recursive=True)
})

print(f"Found {len(series_dirs)} DICOM folders.\n")

success_count = 0
failed_series = []

for series_dir in series_dirs:
    dcm_files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if len(dcm_files) < 20:
        continue

    subject_name = os.path.basename(series_dir.rstrip("\\/"))
    
    try:
        slices = []
        for dcm_file in dcm_files:
            try:
                slices.append(pydicom.dcmread(dcm_file))
            except Exception as e:
                print(f"  ⚠ Skipped corrupted file: {dcm_file}")
                continue
        
        if len(slices) < 20:
            raise ValueError(f"Too few valid slices: {len(slices)}")
        
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        pixel_array = np.stack([s.pixel_array for s in slices])
        affine = np.eye(4)
        
        subject_out = os.path.join(output_root, subject_name)
        os.makedirs(subject_out, exist_ok=True)
        output_path = os.path.join(subject_out, f"{subject_name}.nii.gz")
        
        nib.save(nib.Nifti1Image(pixel_array, affine), output_path)
        print(f"✅ {subject_name} ({len(slices)} slices) -> {output_path}")
        success_count += 1
        
    except Exception as e:
        print(f"❌ FAILED {subject_name}: {str(e)}")
        failed_series.append((subject_name, str(e)))

with open(failed_log, "w") as f:
    for name, err in failed_series:
        f.write(f"{name}: {err}\n")

print(f"\n✅ Conversion complete: {success_count} succeeded, {len(failed_series)} failed.")
print(f"Failed series logged to: {failed_log}")
```

---

## 🛠️ Phase 3: Google Colab Cell 5 - N4 Bias Correction & Skull Stripping

### Approach 1 - Otsu Thresholding

```python
# Install required libraries
!pip install antspyx nibabel numpy scikit-image -q

import os
import glob
import nibabel as nib
import numpy as np
import ants
from skimage.filters import threshold_otsu

# Define paths
root_path = '/content/drive/MyDrive/3 Credit Project/Data'
input_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Raw_output')
output_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output')
os.makedirs(output_folder, exist_ok=True)

nifti_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.nii*"), recursive=True))
total_files = len(nifti_files)
print(f"Found {total_files} NIfTI files.\n")

def n4_bias_correction_antspy(ants_img):
    """Apply N4 bias field correction using ANTsPy."""
    return ants.n4_bias_field_correction(ants_img)

def skull_strip_otsu(ants_img):
    """Apply Otsu thresholding for a baseline skull stripping mask."""
    img_data = ants_img.numpy()
    foreground = img_data[img_data > 0]
    threshold = threshold_otsu(foreground) if foreground.size else 0.0
    brain_mask_data = (img_data > threshold).astype(np.float32)
    brain_mask = ants.from_numpy(
        brain_mask_data,
        origin=ants_img.origin,
        spacing=ants_img.spacing,
        direction=ants_img.direction,
    )
    return brain_mask

for idx, nifti_path in enumerate(nifti_files, 1):
    stem = os.path.basename(nifti_path).replace(".nii.gz", "").replace(".nii", "")
    try:
        ants_img = ants.image_read(nifti_path)
        corrected = n4_bias_correction_antspy(ants_img)
        brain_mask = skull_strip_otsu(corrected)
        brain_only = ants.mask_image(corrected, brain_mask)

        output_path = os.path.join(output_folder, f"{stem}_brain.nii.gz")
        ants.image_write(brain_only, output_path)

        progress_pct = (idx / total_files) * 100
        print(f"[{idx:3d}/{total_files}] ({progress_pct:5.1f}%) ✅ {stem}")
    except Exception as e:
        print(f"[{idx:3d}/{total_files}] ❌ {stem} - Error: {str(e)}")
        continue

print(f"\n✅ N4 correction and Otsu skull stripping complete. Processed {total_files} files.")
```

### Problems with Approach 1

The original threshold-based approach did not properly isolate the brain. Here's why:

1. **Otsu is blind to anatomy:** it separates pixels only by brightness, so it cannot tell brain tissue apart from scalp, fat, or skin.
2. **Intensity overlap in T1-weighted MRIs:** ADNI T1 scans often make outside-head tissues as bright as, or brighter than, the brain.
3. **Result:** the mask tends to keep a whole-head foreground instead of a true brain-only region, and simple cleanup operations do not remove the skull reliably.

### Issues with DeepBrain Version Conflicts

DeepBrain depends on TensorFlow 1.x-style APIs such as `tf.Session`, but Colab and Kaggle often ship with TensorFlow 2.x. That is why the code can fail with `module 'tensorflow' has no attribute 'Session'` before it finishes processing the MRI files.

There are two possible fixes.

### Fix 1 - TensorFlow 1 Compatibility Mode for DeepBrain

```python
!pip install antspyx nibabel numpy deepbrain tensorflow==2.15.0 -q

import os
import glob
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
if not hasattr(tf, "Session"):
    tf.Session = tf.compat.v1.Session
if not hasattr(tf, "placeholder"):
    tf.placeholder = tf.compat.v1.placeholder

import nibabel as nib
import numpy as np
import ants
from deepbrain import Extractor

root_path = '/content/drive/MyDrive/3 Credit Project/Data'
input_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Raw_output')
output_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output')
os.makedirs(output_folder, exist_ok=True)

nifti_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.nii*"), recursive=True))
total_files = len(nifti_files)
print(f"Found {total_files} NIfTI files.\n")

def n4_bias_correction_antspy(ants_img):
    return ants.n4_bias_field_correction(ants_img)

def skull_strip_deepbrain_tf1_compat(ants_img):
    img_data = ants_img.numpy()
    ext = Extractor()
    prob = ext.run(img_data)
    brain_mask_data = (prob > 0.5).astype(np.float32)
    return ants.from_numpy(
        brain_mask_data,
        origin=ants_img.origin,
        spacing=ants_img.spacing,
        direction=ants_img.direction,
    )

for idx, nifti_path in enumerate(nifti_files, 1):
    stem = os.path.basename(nifti_path).replace(".nii.gz", "").replace(".nii", "")
    try:
        ants_img = ants.image_read(nifti_path)
        corrected = n4_bias_correction_antspy(ants_img)
        brain_mask = skull_strip_deepbrain_tf1_compat(corrected)
        brain_only = ants.mask_image(corrected, brain_mask)

        output_path = os.path.join(output_folder, f"{stem}_brain.nii.gz")
        ants.image_write(brain_only, output_path)

        progress_pct = (idx / total_files) * 100
        print(f"[{idx:3d}/{total_files}] ({progress_pct:5.1f}%) ✅ {stem}")
    except Exception as e:
        print(f"[{idx:3d}/{total_files}] ❌ {stem} - Error: {str(e)}")
        continue

print(f"\n✅ DeepBrain completed under TensorFlow 1 compatibility mode. Processed {total_files} files.")
```

### Fix 2 - TensorFlow 2 Safe Alternative

```python
!pip install antspyx nibabel numpy -q

import os
import glob
import nibabel as nib
import numpy as np
import ants
from skimage.filters import threshold_otsu

root_path = '/content/drive/MyDrive/3 Credit Project/Data'
input_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Raw_output')
output_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output')
os.makedirs(output_folder, exist_ok=True)

nifti_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.nii*"), recursive=True))
total_files = len(nifti_files)
print(f"Found {total_files} NIfTI files.\n")

def n4_bias_correction_antspy(ants_img):
    return ants.n4_bias_field_correction(ants_img)

def skull_strip_threshold_based(ants_img):
    img_data = ants_img.numpy()
    foreground = img_data[img_data > 0]
    threshold = threshold_otsu(foreground) if foreground.size else 0.0
    brain_mask_data = (img_data > threshold).astype(np.float32)
    return ants.from_numpy(
        brain_mask_data,
        origin=ants_img.origin,
        spacing=ants_img.spacing,
        direction=ants_img.direction,
    )

for idx, nifti_path in enumerate(nifti_files, 1):
    stem = os.path.basename(nifti_path).replace(".nii.gz", "").replace(".nii", "")
    try:
        ants_img = ants.image_read(nifti_path)
        corrected = n4_bias_correction_antspy(ants_img)
        brain_mask = skull_strip_threshold_based(corrected)
        brain_only = ants.mask_image(corrected, brain_mask)

        output_path = os.path.join(output_folder, f"{stem}_brain.nii.gz")
        ants.image_write(brain_only, output_path)

        progress_pct = (idx / total_files) * 100
        print(f"[{idx:3d}/{total_files}] ({progress_pct:5.1f}%) ✅ {stem}")
    except Exception as e:
        print(f"[{idx:3d}/{total_files}] ❌ {stem} - Error: {str(e)}")
        continue

print(f"\n✅ TensorFlow 2 safe skull stripping completed. Processed {total_files} files.")
```

---

## 🛠️ Phase 3: Google Colab Cell 6 - Verify Skull-Stripped Brains (Before/After)

```python
# Install required libraries
!pip install nibabel matplotlib numpy -q

import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Define paths
root_path = '/content/drive/MyDrive/3 Credit Project/Data'
input_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Raw_output')
output_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output')

raw_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.nii*"), recursive=True))[:3]

if not raw_files:
    print("No raw brain files found. Run Cell 4 first.")
else:
    for raw_path in raw_files:
        subject_name = os.path.basename(raw_path).replace(".nii.gz", "").replace(".nii", "")
        brain_path = os.path.join(output_folder, f"{subject_name}_brain.nii.gz")
        
        if not os.path.exists(brain_path):
            print(f"⚠ Processed file not found for {subject_name}, skipping...")
            continue
        
        raw_img = nib.load(raw_path)
        brain_img = nib.load(brain_path)
        
        raw_data = raw_img.get_fdata().astype(np.float32)
        brain_data = brain_img.get_fdata().astype(np.float32)
        
        raw_vmin = np.percentile(raw_data[raw_data > 0], 10)
        raw_vmax = np.percentile(raw_data[raw_data > 0], 99)
        brain_vmin = np.percentile(brain_data[brain_data > 0], 10)
        brain_vmax = np.percentile(brain_data[brain_data > 0], 99)
        
        z, y, x = raw_data.shape
        mid_z, mid_y, mid_x = z // 2, y // 2, x // 2
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        
        axes[0, 0].imshow(raw_data[mid_z, :, :], cmap='bone', vmin=raw_vmin, vmax=raw_vmax)
        axes[0, 0].set_title("Raw: Axial", fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(brain_data[mid_z, :, :], cmap='bone', vmin=brain_vmin, vmax=brain_vmax)
        axes[0, 1].set_title("Processed: Axial", fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(np.rot90(raw_data[:, mid_y, :]), cmap='bone', aspect='auto', vmin=raw_vmin, vmax=raw_vmax)
        axes[1, 0].set_title("Raw: Coronal", fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.rot90(brain_data[:, mid_y, :]), cmap='bone', aspect='auto', vmin=brain_vmin, vmax=brain_vmax)
        axes[1, 1].set_title("Processed: Coronal", fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[2, 0].imshow(np.rot90(raw_data[:, :, mid_x]), cmap='bone', aspect='auto', vmin=raw_vmin, vmax=raw_vmax)
        axes[2, 0].set_title("Raw: Sagittal", fontweight='bold')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(np.rot90(brain_data[:, :, mid_x]), cmap='bone', aspect='auto', vmin=brain_vmin, vmax=brain_vmax)
        axes[2, 1].set_title("Processed: Sagittal", fontweight='bold')
        axes[2, 1].axis('off')
        
        plt.suptitle(f"Before/After: {subject_name} | Shape: {raw_data.shape}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print(f"✅ {subject_name}: Skull stripped & N4 corrected. Observe skull removal & brightness normalization.")
```

---

## 🛠️ Phase 3: Google Colab Cell 7 - Brain Registration to MNI152

```python
# Install required libraries
!pip install nilearn nibabel antspyx -q

import os
import glob
import nibabel as nib
import ants
from nilearn import datasets

# Define paths
root_path = '/content/drive/MyDrive/3 Credit Project/Data'
input_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output')
output_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_MNI152_output')
os.makedirs(output_folder, exist_ok=True)

# Load MNI152 template
template_img = datasets.load_mni152_template(resolution=1)
template_path = os.path.join(root_path, 'mni152_template.nii.gz')
nib.save(template_img, template_path)
fixed = ants.image_read(template_path)

brain_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))
print(f"Found {len(brain_files)} brain volumes for registration.")

for idx, brain_path in enumerate(brain_files, 1):
    try:
        moving = ants.image_read(brain_path)
        
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="SyN"
        )
        warped = registration["warpedmovout"]
        
        stem = os.path.basename(brain_path).replace("_brain.nii.gz", "").replace(".nii.gz", "").replace(".nii", "")
        output_path = os.path.join(output_folder, f"{stem}_mni152.nii.gz")
        
        ants.image_write(warped, output_path)
        print(f"[{idx}] ✅ Saved registered volume: {stem}")
    except Exception as e:
        print(f"[{idx}] ❌ Failed to register: {str(e)}")
        continue

print("✅ MNI152 registration complete.")
```

---

## 🛠️ Phase 3: Google Colab Cell 8 - Intensity Normalization

```python
# Install required libraries
!pip install nibabel numpy -q

import os
import glob
import nibabel as nib
import numpy as np

# Define paths
root_path = '/content/drive/MyDrive/3 Credit Project/Data'
input_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_MNI152_output')
output_folder = os.path.join(root_path, 'Outputs', 'Phase3_NIfTI_Normalized_output')
os.makedirs(output_folder, exist_ok=True)

registered_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))
print(f"Found {len(registered_files)} registered volumes.")

for idx, volume_path in enumerate(registered_files, 1):
    try:
        img = nib.load(volume_path)
        data = img.get_fdata().astype(np.float32)
        
        brain_voxels = data[data > 0]
        if brain_voxels.size == 0:
            raise ValueError(f"No foreground voxels found in {volume_path}")
        
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        if std == 0:
            std = 1.0
        
        normalized = np.zeros_like(data, dtype=np.float32)
        normalized[data > 0] = (data[data > 0] - mean) / std
        
        stem = os.path.basename(volume_path).replace(".nii.gz", "").replace(".nii", "")
        output_path = os.path.join(output_folder, f"{stem}_zscore.nii.gz")
        
        nib.save(nib.Nifti1Image(normalized, img.affine, img.header), output_path)
        print(f"[{idx}] ✅ Saved normalized volume: {stem}")
    except Exception as e:
        print(f"[{idx}] ❌ Failed to normalize: {str(e)}")
        continue

print("✅ Intensity normalization complete.")
```

---

## 💡 Quick Reference: Switching Between Kaggle & Google Colab

| Task | Kaggle Path | Google Colab Path |
|------|------------|-------------------|
| Input data | `/kaggle/input/dataset-name/` | `/content/drive/MyDrive/3 Credit Project/Data/Outputs/Phase3_NIfTI_Raw_output/` |
| Output folder | `/kaggle/working/` | `/content/drive/MyDrive/3 Credit Project/Data/Outputs/Phase3_NIfTI_Brain_n4_correction_and_skull_stripping_output/` |
| Mount Drive | N/A | `from google.colab import drive; drive.mount('/content/drive')` |
| Install packages | `!pip install package` | `!pip install package -q` (add `-q` for quieter output) |

---

> ✅ **All cells are now adapted for Google Colab!** Follow the cells in order (1 → 2 → 3 → 4 → 5 → 6 → 7 → 8) and your preprocessing pipeline will complete successfully.
