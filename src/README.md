# RISE-MICCAI LISA 2025 – 3D Hippocampus Segmentation

This repository contains a complete training and inference pipeline for the
RISE‑MICCAI LISA 2025 challenge.  The task involves segmenting the left and
right hippocampi from ultra‑low‑field neonatal brain MR images.  The
segmentation is framed as a three–class problem where class 0 denotes
background, class 1 denotes the left hippocampus and class 2 denotes the
right hippocampus【533959220451368†L6-L9】.  The code provided here performs the following steps:

* **Data loading and preprocessing** – NIfTI images and their corresponding
  segmentations are loaded, resampled to a common voxel spacing and resized
  to a fixed isotropic volume of 96×96×96 voxels.  Intensities are normalised
  into a unit range on a per‑volume basis.  The training
  CSV supplied by the organisers lists both the CISO image and the
  HF/LF segmentation; during training only the hippocampi labels (0, 1, 2)
  are used.
* **Stratified group cross‑validation** – the data are split into a number
  of folds using `GroupKFold` (from scikit‑learn) to guarantee that all
  samples from a single subject (identified by the `ID` field) appear in
  only one split.  This prevents information leakage from the high‑field
  and low‑field images of the same subject into both train and validation
  sets.
* **Model selection** – three different 3D models are provided:
  1. **UNesT (pre‑trained)** – a transformer‑based 3D U‑Net architecture
     pre‑trained on whole brain segmentation.  The last convolutional layer
     is adapted to output three classes.  A helper function downloads and
     loads the weights from the [MONAI](https://monai.io/) model zoo.
  2. **SegResNet** – a 3D residual U‑Net implemented in MONAI.  It
     provides a strong baseline for medical image segmentation and does
     not require pre‑training.
  3. **Autoencoder‑Seg** – a simple 3D convolutional autoencoder used
     here as a lightweight segmentation network.  It consists of three
     down‑sampling convolutions and three transpose convolutions and is
     useful for comparing against the transformer and residual networks.
* **Loss functions and class imbalance** – the hippocampi occupy a very
  small fraction of each MR volume.  To prevent the networks from
  predicting background everywhere, the training loop combines a
  weighted multi‑class cross‑entropy loss with a per‑class Dice loss.
  Class weights are computed once per fold based on the number of
  voxels belonging to each class in the training set.  This encourages
  the model to learn the rare hippocampal voxels instead of over‑fitting
  to background.
* **Training, evaluation and model saving** – for each fold the selected
  model is trained for a configurable number of epochs.  After each
  epoch the model is evaluated on the validation split and the Dice
  score is recorded.  The best performing model (highest mean Dice
  across classes) is saved to disk.  All training hyper‑parameters are
  exposed via command‑line arguments in `train.py`.
* **Inference** – `inference.py` loads a trained model and produces
  NIfTI segmentations for a folder of test images.  The predictions
  retain the same voxel spacing and dimensions as the inputs and are
  saved with the naming convention expected by the challenge.

Users are expected to run this pipeline on their own machines with the
required dependencies installed (PyTorch ≥ 1.10, MONAI ≥ 1.2, nibabel,
scikit‑learn and pandas).  The scripts will not run in the current
environment because these libraries are unavailable, but the code is
organised and documented to facilitate adaptation to your local setup.

## Directory Structure

```
rise_miccai/
├── README.md                # this file
├── dataset.py               # data loading and cross‑validation utilities
├── models.py                # definitions of the three segmentation networks
├── train.py                 # training script with k‑fold cross‑validation
├── inference.py             # script for generating predictions on new data
└── utils.py                 # helper functions (losses, metrics, etc.)
```

## Usage

1. **Prepare the CSVs** – ensure `train.csv` contains the columns
   `filepath` (path to the CISO image), `filepath_label` (path to the
   merged HF/LF hippocampal segmentation) and `ID` (subject identifier).
   The `test.csv` should contain `filepath` and `ID` for the unseen
   validation/test set.
2. **Install dependencies** – in your Python environment install the
   required libraries, e.g.:

   ```bash
   pip install torch==2.2.0 monai nibabel pandas scikit-learn
   ```

3. **Train the models** – run the training script for each model,
   specifying the desired number of folds and other hyper‑parameters:

   ```bash
   python train.py \
     --train_csv /path/to/train.csv \
     --output_dir /path/to/output_models \
     --model unest --folds 5 --epochs 100 --batch_size 2 --gpu 0
   ```

   Replace `unest` with `segresnet` or `autoencoder` to train the other
   architectures.  The script writes one sub‑directory per fold and
   stores the best model and training logs therein.

4. **Generate predictions** – after training, run the inference script
   to segment the validation/test set:

   ```bash
   python inference.py \
     --test_csv /path/to/test.csv \
     --model_dir /path/to/output_models/fold_0 \
     --model unest \
     --output_dir /path/to/predictions
   ```

   The script will write NIfTI files following the naming scheme
   `LISAHF<ID>segprediction.nii.gz` for each subject, retaining the
   original spacing and dimensions as required by the challenge【533959220451368†L6-L9】.

## Notes

* The code uses group‑wise splitting to avoid data leakage between the
  high‑field (HF) and low‑field (LF) volumes of the same subject.  Always
  ensure that when adding additional data (e.g. the LF segmentations) you
  link them by the `ID` column so that both views fall into the same
  training or validation split.
* To avoid over‑fitting to the background class, the loss function
  combines weighted cross‑entropy with Dice loss and computes class
  weights from the training data.  You can further mitigate imbalance by
  sampling sub‑volumes around the hippocampi or by using importance
  sampling; however, the provided implementation uses full volumes for
  simplicity.
* The pretrained UNesT weights are downloaded from the Hugging Face hub
  (repo: `MONAI/wholeBrainSeg_Large_UNEST_segmentation`).  Make sure you
  have an internet connection when running the training script for the
  first time; the weights will be cached locally afterwards.
