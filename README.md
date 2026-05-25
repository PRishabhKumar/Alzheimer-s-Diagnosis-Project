# Diagnostic Assistant for Alzheimer's Prediction

A machine learning-based diagnostic assistant designed to aid in the early prediction and diagnosis of Alzheimer's disease using MRI brain imaging data from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Getting Started](#getting-started)
4. [Step-by-Step Guide](#step-by-step-guide)
   - [Running the Notebooks](#running-the-notebooks)
   - [Downloading Data from ADNI Website](#downloading-data-from-adni-website)
   - [Organizing Downloaded Data](#organizing-downloaded-data)
5. [Folder Structure](#folder-structure)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)

---

## Project Overview

This project aims to build a diagnostic assistant that can predict Alzheimer's disease status using structural MRI brain scans. The project includes:

- **Data Collection & Verification**: Initial scripts to validate and organize ADNI dataset
- **Data Processing**: Skull stripping and N4 bias correction of MRI images
- **Model Development**: Training and testing of diagnostic models (in progress)

The project uses the ADNI dataset, which is one of the most comprehensive neuroimaging datasets available for Alzheimer's research.

---

## Prerequisites

Before you begin, ensure you have the following:

- **Free Kaggle or Google Colab Account**
  - **Kaggle**: Create a free account at [Kaggle.com](https://www.kaggle.com/)
  - **Google Colab**: Use your Google account at [Colab.research.google.com](https://colab.research.google.com/)
  
- **ADNI Account Credentials** - Your team lead will provide login credentials to access the ADNI dataset

- **A file extraction tool** (WinRAR, 7-Zip, or built-in ZIP extraction on Windows/Mac) - for organizing downloaded MRI data

**No local software installation required!** All notebooks run in the cloud.

---

## Getting Started

### 1. Clone or Download the Repository

**Option A - Using Git** (if you have Git installed):
```bash
git clone https://github.com/YOUR_USERNAME/Diagnostic-Assistant-Alzheimers.git
```

**Option B - Manual Download**:
1. Go to the GitHub repository
2. Click **"Code"** → **"Download ZIP"**
3. Extract the ZIP file on your computer

### 2. Upload to Kaggle or Google Colab

#### **For Kaggle**:

1. Log in to [Kaggle.com](https://www.kaggle.com/)
2. Go to **"Code"** in the top menu
3. Click **"New Notebook"**
4. In the notebook, click **"File"** → **"Upload notebook"**
5. Select the `.ipynb` file from the `Notebooks/` folder
6. Attach the dataset (you'll need to add the DXSUM_15Apr2026.csv file as a dataset)
   - Click **"+ Add Data"** 
   - Upload or create a dataset with your ADNI data files

#### **For Google Colab**:

1. Go to [Colab.research.google.com](https://colab.research.google.com/)
2. Click **"File"** → **"Upload notebook"**
3. Select the `.ipynb` file from the `Notebooks/` folder
4. In the notebook, mount your Google Drive to access data:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Upload your ADNI data CSV and MRI files to Google Drive
6. Update the file paths in the notebook to point to your Google Drive location

---

## Step-by-Step Guide

### Running the Notebooks

The project includes Jupyter notebooks that run on **Kaggle** or **Google Colab**. Follow these steps:

#### **Notebook 1: Initial Data Collection and Verification**

**File**: `Notebooks/Initial_data_collection_and_verification.ipynb`

**What this notebook does**:
- Loads the ADNI participant diagnostic data (DXSUM_15Apr2026.csv)
- Selects 100 participants (50 cognitively normal, 50 with Alzheimer's disease)
- Extracts participant IDs for downloading MRI images from ADNI

**How to run it on Kaggle**:

1. Upload the notebook to Kaggle (see Getting Started section)
2. Attach the `DXSUM_15Apr2026.csv` dataset
3. In the first code cell, update the path:
   ```python
   base_path = '/kaggle/input/datasets/YOUR_DATASET_NAME/adni-dataset'  # Replace YOUR_DATASET_NAME
   ```
4. Click the **"Run"** button or press `Shift + Enter` to execute each cell sequentially
5. The notebook will output a **list of Participant IDs** (comma-separated)
   - **Copy this entire list** - you'll need it in the next step to download data from the ADNI website

**How to run it on Google Colab**:

1. Upload the notebook to Google Colab
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update the CSV file path to point to your Google Drive
4. Run each cell by clicking the play button or pressing `Shift + Enter`
5. The notebook will output a **list of Participant IDs** 
   - **Copy this entire list** - you'll need it to download data from ADNI

#### **Notebook 2: Skull Stripping and N4 Correction**

**File**: `Notebooks/after_skull_stripping_and_N4_correction.ipynb`

**What this notebook does**:
- Performs skull stripping (removes non-brain tissue from MRI images)
- Applies N4 bias field correction (corrects for MRI scanner artifacts)
- Prepares images for downstream analysis

**How to run it**:

1. First, ensure you've downloaded and organized your MRI data locally (see section below)
2. Upload this notebook to Kaggle or Google Colab
3. Upload your organized MRI data to the platform
4. Update the file paths in the notebook to match your uploaded data location
5. Run each cell sequentially

---

### Downloading Data from ADNI Website

This section explains how to download the MRI data using the Participant IDs generated by the first notebook.

#### **Step 1: Log in to ADNI**

1. Go to [ADNI Official Website](https://ida.loni.usc.edu/login.jsp)
2. Log in with the credentials provided by your team lead
3. You should now see the ADNI dashboard

#### **Step 2: Access the ADNI Download Tool**

1. After logging in, look for **"Download"** or **"Advanced Search"** in the main menu
2. Click on **"Advanced Search"**
3. You should see a search interface with various filtering options

#### **Step 3: Configure Filters for MRI Search**

Now you'll set up filters to find only the specific MRI scans you need:

**Important Filters to Select**:

| Filter Option | Value to Select |
|---|---|
| **Study** | ADNI (or ADNI2 if available) |
| **Image Type** | MRI |
| **MRI Sequence Type** | **MPRAGE** and **T2 FLAIR** |
| **Visit Type** | Baseline/ADNIGO Initial Visit (first visit only) |
| **Status** | Complete (or any status with data available) |

**Step-by-step filtering instructions**:

1. In the Advanced Search page, look for the filter panel on the left or top
2. Under "Image Type" → Select **MRI**
3. Under "MRI Sequence" → Select **MPRAGE** (this is the main structural scan)
   - Also select **T2 FLAIR** (if available/needed for your analysis)
4. Under "Visit Type" → Select **Baseline** or **ADNIGO Initial Visit** (ensures you get only the first visit)
5. Under "Diagnosis/Status" → If available, select both **CN (Cognitively Normal)** and **AD (Alzheimer's Disease)**
6. Click **"Search"** or **"Apply Filters"**

#### **Step 4: Select Participant IDs and Download**

1. The search results will show many participants
2. Copy the **Participant IDs** generated from your first notebook (from the Jupyter notebook)
3. Look for these specific IDs in the search results
4. **Select the checkbox** next to each participant ID you want to download
5. Click the **"Download"** button
6. You may be asked to:
   - Agree to data usage terms → Click "I Agree"
   - Choose download format → Select **ZIP** or your preferred format
7. The download will start (it may take several minutes to hours depending on number of scans)
8. **Save the ZIP file** to a memorable location on your computer

#### **Alternative: Batch Download Script**

If ADNI provides a batch download option:
1. After selecting your participants, look for **"Batch Download"** or **"Script"** option
2. Download the batch script and follow ADNI's instructions for using it

---

### Organizing Downloaded Data

Once you've downloaded the ZIP files from ADNI, follow these steps to organize them so the notebooks can find them correctly.

#### **Step 1: Locate Your Downloaded ZIP Files**

1. Find where your downloads are saved (usually in your `Downloads` folder or the location you specified)
2. You should have one or more ZIP files (names may look like `ADNI_MRI_[DATE].zip`)

#### **Step 2: Extract the ZIP Files**

**On Windows**:
1. Right-click on the ZIP file
2. Select **"Extract All..."**
3. Choose a destination folder (create a new folder called `Downloaded_ADNI_Data` on your Desktop for clarity)
4. Click **"Extract"**

**On Mac/Linux**:
1. Double-click the ZIP file (it auto-extracts)
2. Or in terminal: `unzip filename.zip -d destination_folder`

#### **Step 3: Understand the ADNI Folder Structure**

After extraction, you'll see a folder structure like:

```
Downloaded_ADNI_Data/
├── 002_S_0295/
│   ├── MRI/
│   │   ├── 2006-08-16/
│   │   │   ├── MPRAGE/
│   │   │   │   └── image_file.nii.gz
│   │   │   └── T2_FLAIR/
│   │   │       └── image_file.nii.gz
│   ├── 003_S_0193/
│   ├── 005_S_0006/
│   └── ... (more participants)
```

#### **Step 4: Rename Folders (Important!)**

The notebooks expect data to be organized in a specific folder structure. Follow these naming conventions:

**Main Data Folder Name**: `Data/ZIP FOLDERS/`

This folder should contain subdirectories named after **Participant IDs** with the following structure:

```
Data/
└── ZIP FOLDERS/
    ├── 002_S_0295/
    │   └── (MRI files in subfolders)
    ├── 003_S_0193/
    │   └── (MRI files in subfolders)
    ├── 005_S_0006/
    │   └── (MRI files in subfolders)
    └── ... (more participants)
```

**Detailed Steps**:

1. After extracting your ZIP files, you should have folders with participant IDs (e.g., `002_S_0295`)
2. Move these participant ID folders to `Data/ZIP FOLDERS/` in your project directory
3. Make sure the folder names **exactly match** your Participant IDs (e.g., `002_S_0295`)
4. Inside each participant folder, ensure there are subfolders for MRI sequences:
   - `MPRAGE/` (contains MPRAGE MRI files)
   - `T2_FLAIR/` (if downloaded, contains T2 FLAIR files)

**Example command** (if using terminal/command prompt):

```bash
# Navigate to your extracted data location
cd path/to/Downloaded_ADNI_Data

# Move participant folders to the project's Data/ZIP FOLDERS directory
# Example for Windows:
move 002_S_0295 "C:\path\to\project\Data\ZIP FOLDERS\"
move 003_S_0193 "C:\path\to\project\Data\ZIP FOLDERS\"

# Example for Mac/Linux:
mv 002_S_0295 /path/to/project/Data/ZIP\ FOLDERS/
mv 003_S_0193 /path/to/project/Data/ZIP\ FOLDERS/
```

#### **Step 5: Verify Your Folder Structure**

Before running the notebooks, verify that:

- [ ] Participant ID folders are in `Data/ZIP FOLDERS/`
- [ ] Each folder is named exactly as the Participant ID (e.g., `002_S_0295`)
- [ ] Inside each folder, there are subfolders like `MPRAGE/` and/or `T2_FLAIR/`
- [ ] MRI image files are in these subfolders (files typically end in `.nii.gz` or `.nii`)

If any of these are incorrect, the notebooks will give **"folder not found"** or **"file not found"** errors.

---

## Folder Structure

Here's the complete project folder structure:

```
Diagnostic-Assistant-Alzheimers/
├── README.md                                    # This file
├── Notebooks/
│   ├── Initial_data_collection_and_verification.ipynb
│   └── after_skull_stripping_and_N4_correction.ipynb
├── Data/
│   ├── DXSUM_15Apr2026.csv                      # ADNI diagnostic data
│   ├── Project Initial data/                    # Initial project documentation
│   └── ZIP FOLDERS/                             # Where you place downloaded MRI data
├── Approach and Architecture Diagram/           # Project design documents
├── Papers/                                      # Reference papers and literature
└── Planning Docs/                               # Project planning documents
```

---

## Troubleshooting

### Common Errors and Solutions

#### **Error: "FileNotFoundError: No such file or directory" (Kaggle)**

**Cause**: The notebook can't find your dataset on Kaggle

**Solution**:
1. Make sure you've attached the dataset in Kaggle:
   - Click **"+ Add Data"** → Select your uploaded CSV file
2. Check the `base_path` in the notebook matches your dataset name:
   ```python
   base_path = '/kaggle/input/datasets/YOUR_DATASET_NAME/...'
   ```
3. Verify the CSV filename is correct: `DXSUM_15Apr2026.csv`

#### **Error: "FileNotFoundError: No such file or directory" (Google Colab)**

**Cause**: Google Drive is not mounted or file path is incorrect

**Solution**:
1. Make sure you've mounted Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Upload your CSV file to Google Drive
3. Update the file path in the notebook:
   ```python
   csv_path = '/content/drive/MyDrive/your_folder/DXSUM_15Apr2026.csv'
   ```

#### **Error: "ModuleNotFoundError: No module named 'nibabel'" or similar**

**Cause**: Python library not available in the environment

**Solution**:
Add this cell at the beginning of your notebook:
```python
!pip install nibabel pandas numpy scikit-image scipy matplotlib
```

#### **Error: "No MPRAGE images found"**

**Cause**: MRI files are not organized correctly or uploaded to the cloud platform

**Solution**:
1. Organize MRI files locally first (see "Organizing Downloaded Data" section)
2. Create the folder structure locally before uploading
3. Upload the entire organized folder to Kaggle or Google Drive
4. Update the notebook path to point to the uploaded folder

#### **Error: "Participant ID not found in DXSUM file"**

**Cause**: The Participant IDs from your download don't match the diagnostic data file

**Solution**:
1. Ensure you downloaded data for the correct participants
2. Check that the ADNI diagnostic CSV file is up-to-date
3. Verify Participant IDs in your file names (should be like `011_S_0002`)

### Getting Help

If you encounter an issue not listed above:

1. **Check the notebook comments** - Each cell includes explanations
2. **Review your folder structure** - Most errors are due to incorrect folder organization
3. **Check file paths** - Ensure they match your Kaggle dataset or Google Drive paths
4. **Restart the kernel** - In Kaggle/Colab, click "Restart kernel" to clear memory issues
5. **Contact the project maintainer** - Open an issue on GitHub

---

## Contributing

If you're working with teammates on this project:

1. **Always pull the latest version** before starting work:
   ```bash
   git pull origin main
   ```

2. **Create a new branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add description of changes"
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** for your team to review

5. **Do not commit the `/Data/ZIP FOLDERS/` folder** to GitHub (it's too large)
   - Use `.gitignore` to exclude it:
   ```
   Data/ZIP FOLDERS/*
   *.nii.gz
   ```

---

## Dataset Attribution

This project uses data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** database (http://adni.loni.usc.edu). The ADNI was launched in 2003 as a public-private partnership.

**When publishing research using this data, please cite**:
> Weiner MW, Veitch DP, Aisen PS, et al. The Alzheimer's Disease Neuroimaging Initiative-3: Continued innovation for clinical trial improvement. Alzheimers Dement. 2017;13(5):561-571.

---

