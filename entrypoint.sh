#!/usr/bin/env bash
set -euo pipefail

# ----- logging con copia a /output/run.log -----
mkdir -p /output
LOG=/output/run.log
exec > >(tee -a "$LOG") 2>&1

echo "===== LISA Task1 Entrypoint ====="
echo "[INFO] DATE: $(date)"
echo "[INFO] INPUT_DIR=${INPUT_DIR:-/input}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR:-/output}"
echo "[INFO] MODEL_DIR=${MODEL_DIR:-/my_solution/models/new_model_bbox_giou_brain0.1_l0.1}"
echo "[INFO] Python: $(python -V)"

# GPU info
if command -v nvidia-smi &>/dev/null; then
  echo "[INFO] nvidia-smi:"
  nvidia-smi || true
else
  echo "[INFO] nvidia-smi no disponible (CPU o sin GPU visible)"
fi

# ----- info del sistema -----
echo "[INFO] Volúmenes montados relacionados:"
mount | grep -E "/input|/output" || echo "No se detectaron montajes para /input o /output"

echo "[INFO] Contenido del root /:"
ls -la / | head -n 50 || true

echo "[INFO] Uso de disco en /:"
df -h / || true

# Inventario de /input
echo "[INFO] /input primeros archivos:"
ls -la "${INPUT_DIR:-/input}" | head -n 40 || true
echo "[INFO] NII count: $(ls -1 ${INPUT_DIR:-/input}/*.nii.gz 2>/dev/null | wc -l)"

# Inventario de /output inicial
echo "[INFO] /output antes de escribir:"
ls -la "${OUTPUT_DIR:-/output}" || true

# ----- TU PIPELINE -----
echo "[STEP] 1) csv_creation"
python /my_solution/1_csv_creation.py \
  --val_path_dir "${INPUT_DIR:-/input}" \
  --path_results "/my_solution/results/"

echo "[STEP] 2) inference"
python /my_solution/2_inference_cascade.py \
  --test_csv "/my_solution/results/preprocessed_data/df_test.csv" \
  --models_dir /my_solution/models/fold_models \
  --output "${OUTPUT_DIR:-/output}" \
  --model_name eff-b2 \
  --dim 192 \
  --use_tta 1 \
  --use_post 1 \
  --save_folds 0 \
  --format LISAHF \
  --debug 1

# Listado final
echo "[INFO] /output después de ejecutar:"
ls -la "${OUTPUT_DIR:-/output}" || true
echo "===== DONE ====="

