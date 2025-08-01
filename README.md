# Lacuna-Solar-Survey-Challenge
Lacuna Solar Survey Challenge - Zindi

Author: Omar Abdelhedi  
Final Ranking: Top 32 on Private Leaderboard  
Public Score: 0.897  
Private Score: 0.912  
Competition Link: https://zindi.africa/competitions/lacuna-solar-survey-challenge

--------------------------------------------------------------------------------
🏆 Challenge Overview:
The objective of this challenge was to accurately predict the number of **boilers** and **panels** in solar installations using satellite images and metadata (e.g., image origin and placement). This solution integrates visual data and structured metadata using a multimodal deep learning pipeline.

--------------------------------------------------------------------------------
📦 Solution Summary:

This approach leverages the following key components:

1. **Image Backbone**:
   - Model: `tf_efficientnet_b7` from `timm` (pretrained)
   - Strategy: Extracts deep visual features from input images

2. **Metadata Encoding**:
   - Encodes `img_origin` (binary) and `placement` (one-hot for 4 classes)
   - Uses a small MLP to process and project metadata features

3. **Fusion Layer**:
   - Applies multi-head attention on metadata embeddings
   - Concatenates metadata features with image embeddings before final regression

4. **Regressor Head**:
   - Fully connected layers with ReLU and Dropout
   - Output: Two values representing the count of boilers and panels
   - Activation: `Softplus` for positive-valued outputs

5. **Loss & Optimization**:
   - Loss Function: `HuberLoss (delta=1.0)`
   - Optimizer: `AdamW` with `CosineAnnealingWarmRestarts` scheduler
   - AMP training enabled for faster training and stability

6. **Data Augmentation**:
   - Heavy augmentations (rotation, color jitter, blur, flipping, resizing)
   - Custom classes for rotation and blur to improve generalization

7. **Cross Validation & Ensembling**:
   - 5-fold K-Fold CV on the training set
   - Final predictions are averaged across all 5 fold models

--------------------------------------------------------------------------------
📊 Results:

- Public Leaderboard MAE: **0.897**
- Private Leaderboard MAE: **0.912**
- Final Rank: **32**

The solution generalized well, particularly on the private set, benefiting from metadata integration and ensembling.

--------------------------------------------------------------------------------
📁 File Structure:

- `main.py`: Training and inference code
- `best_model_fold{i}.pth`: Best checkpoint per fold
- `submission_original_b7.csv`: Raw float predictions
- `submission_integer_b7.csv`: Rounded final predictions

--------------------------------------------------------------------------------
📂 Data Access:

You can download the dataset directly from the **Data** section of the Zindi competition page:

➡️ Dataset Download Link: https://zindi.africa/competitions/lacuna-solar-survey-challenge/data

The following files are provided:
- `Train.csv`
- `Test.csv`
- `TrainImages/` (image folder)
- `TestImages/` (image folder)

Make sure to place the CSV files in the working directory and set `imgs_path` in the script to the correct folder path for the images.

--------------------------------------------------------------------------------
🔧 How to Run:

1. Install dependencies:
pip install -r requirements.txt


2. Prepare dataset:
- Place `Train.csv` and `Test.csv` in the same directory as `main.py`
- Place the corresponding `.jpg` images in folders and update `imgs_path` accordingly
- Each image should be named as `ID.jpg`

3. Train:
python main.py

4. Submissions:
- Two CSVs will be generated in the current directory after inference

--------------------------------------------------------------------------------
📝 Credits:

Developed by Omar Abdelhedi  
GitHub: https://github.com/OmarAbdelhedi

For questions or collaboration, feel free to reach out!
