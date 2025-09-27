# RISE-MICCAI LISA 2025: Hippocampus Segmentation

Two-stage cascade deep learning pipeline for 3D hippocampus segmentation in ultra-low-field neonatal MRI.

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/rise-miccai-lisa2025.git
cd rise-miccai-lisa2025
pip install -r requirements.txt
```

### Training
```bash
python 0_train.py \
  --model_name eff-b2 \
  --root_dir ./models/exp1 \
  --num_epochs 150 \
  --num_folds 5 \
  --batch_size 2 \
  --dim 192 \
  --device cuda:0
```

### Inference
```bash
# 1. Create test CSV
python 1_csv_creation.py --val_path_dir /input --path_results ./results/

# 2. Run inference
python 2_inference_cascade.py \
  --test_csv ./results/preprocessed_data/df_test.csv \
  --models_dir ./models/exp1/fold_models \
  --output_dir ./predictions \
  --model_name eff-b2 \
  --dim 192 \
  --use_tta 1
```

## Docker Usage

### Build & Run
```bash
# Build
docker build -t rise_task2:latest .

# Development
docker compose run --rm dev bash

# Production
docker compose up run
```

### Synapse Submission
```bash
# 1. Tag
docker tag rise_task2:latest docker.synapse.org/SYN_ID/rise_task2:v1.0

# 2. Login
docker login docker.synapse.org

# 3. Push
docker push docker.synapse.org/SYN_ID/rise_task2:v1.0

# 4. Test locally
docker run --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output:rw \
  docker.synapse.org/SYN_ID/rise_task2:v1.0
```

## Key Features
- **Cascade architecture**: Base model (ROI detection) + Fine model (refinement)
- **3-class segmentation**: Background (0), Left hippocampus (1), Right hippocampus (2)
- **TTA with safe flips**: Preserves laterality
- **5-fold cross-validation**: Group-wise splitting
- **Post-processing**: Morphological operations + component filtering

## Requirements
- GPU: NVIDIA with 4GB+ VRAM
- CUDA: 11.7+
- Python: 3.8+

## License
MIT