# 🧠 Detailed Project Roadmap: Multimodal Alzheimer's Classification

This roadmap follows the **Stage I-IV** pipeline defined in your architecture diagram, utilizing hybrid Deep Learning (3D-CNN) and Gradient Boosting (XGBoost) for diagnostic prediction.

---

## 🛠️ Tech Stack
- **Languages:** Python (Google Colab / Local Environment)
- **Medical Imaging:** ANTspy, FSL (via Nipype), PyDicom, Nibabel
- **Deep Learning:** PyTorch, MONAI (Medical Open Network for AI)
- **Classical ML:** Scikit-learn, XGBoost
- **Explainability:** Captum (Grad-CAM), SHAP

---

## 📍 STAGE I: Data Collection & Multimodal Acquisition
*Objective: Extract raw high-dimensional data from the ADNI portal.*

1. **Clinical Filtering:** - Use `DXSUM_PDXCONV_ADNIALL.csv` to identify `CN` (Label 1) and `AD` (Label 3) subjects.
   - Extract raw clinical features: **MMSE**, **APOE-ε4**, and **Age**.
2. **Imaging Acquisition:** - Search LONI-IDA for **3D T1-weighted MRI** (MP-RAGE) matching your filtered subjects.
   - Download in **DICOM** format.
3. **Reference Atlases:** - Download **MNI152** (standard brain template) and **AAL3** (Automated Anatomical Labeling) for region-of-interest mapping.

---

## 📍 STAGE II: Preprocessing & Pipeline Design
*Objective: Transform messy medical data into standardized tensors.*

### **Branch A: Brain MRI Preprocessing (ANTspy / FSL)**
1. **NIfTI Conversion:** Convert raw DICOM to NIfTI using `dcm2niix`.
2. **N4 Bias Field Correction:** Remove magnetic field inhomogeneities.
3. **Skull Stripping:** Use `ANTsPy` (specifically `antspynet.brain_extraction`) to isolate brain tissue from bone/fat.
4. **MNI152 Registration:** Spatially normalize all brains to the MNI152 coordinate system.
5. **Intensity Normalization:** Apply Z-score or Min-Max scaling to voxel intensities.

### **Branch B: Clinical Preprocessing (Scikit-Learn)**
1. **Categorical Encoding:** Convert APOE-ε4 genotypes and Sex into **One-Hot Encoded** vectors.
2. **Z-score Normalization:** Standardize numerical values like MMSE and Age so they share a common scale with the image features.

---

## 📍 STAGE III: Prediction Models & Multimodal Fusion
*Objective: Feature extraction and multi-input classification.*

### **1. Imaging Branch: 3D-SE-DenseNet-121**
- **Architecture:** Implement a **3D Squeeze-and-Excitation DenseNet-121** using the `MONAI` library.
- **Pre-training:** Initialize weights from `MedicalNet` (pre-trained on large-scale medical datasets).
- **Output:** Extract the final layer **Global Average Pooling** (GAP) as the "Visual Clue" vector.

### **2. Clinical Branch: XGBoost Classifier**
- **Architecture:** Train an `XGBoost` model on the normalized clinical vectors.
- **Output:** Extract the probability scores or latent feature vectors as "Logic Clues."

### **3. Multimodal Fusion (PyTorch)**
- **Fusion Layer:** Use **Late Fusion** to concatenate the CNN visual vector and the XGBoost logic vector.
- **Final Classifier:** Fully connected layers followed by Softmax for final **CN vs. AD** classification.

---

## 📍 STAGE IV: Explainable AI (XAI) Integration
*Objective: Validate clinical trust through visualization.*

### **1. Spatial Attribution (Images)**
- Use `Captum` to implement **3D Grad-CAM**.
- Generate 3D heatmaps to ensure the model is focusing on the **Hippocampus** and **Ventricles**.
- Map heatmaps against the **AAL3 Atlas** for a "Quantitative ROI Report."

### **2. Feature Attribution (Clinical)**
- Use `SHAP` (SHapley Additive exPlanations) on the XGBoost component.
- Generate SHAP importance graphs to rank how much MMSE vs. APOE-ε4 contributed to each specific prediction.

---

## 📈 Evaluation Metrics
- **Primary:** Accuracy, AUC-ROC (Area Under Curve).
- **Secondary:** Sensitivity (Recall for AD), Specificity (Correct identification of CN).
- **Validation:** 5-Fold Cross-Validation (split by Subject ID).