# rise-miccai-challenge2025


https://summer.rise-miccai.org

The challenge of automatic segmentation in low-field MRI environments poses a significant hurdle, particularly in regions with limited resources where high-field MRI systems are scarce. Structural delineation becomes even more challenging due to the lower resolution of accessible systems like the 0.064T Hyperfine scanner. Despite these obstacles, the benefits of utilizing low-field MRI, such as portability and reduced clinical costs, are undeniable, especially in settings where sedation for young patients is impractical.

Addressing this segmentation challenge head-on, the LISA challenge introduces its second task. Participants are called upon to pioneer deep learning methods tailored for automatically segmenting the (A) bilateral hippocampi and (B) basal ganglia in ultra low-field (0.064T) T2-weighted MRI images of early childhood brains. Both structures play a critical role in cognitive functions, making their accurate segmentation essential for understanding abnormal neurodevelopment.


Dataset
Data description
High field T2 data was acquired at Kawempe National Referral Hospital, Makerere University, Kampala, Uganda; CUBIC, University of Cape Town, South Africa and Warren Alpert Medical School at Brown University, Providence, RI, USA, and the Advanced Baby Imaging Lab, Rhode Island Hospital, Providence, RI, USA. All images were collected by MRI techs with experience imaging patients at the institutions listed above. High field scans are synchronized with matching low field Hyperfine scans of the same subjects. All bilateral hippocampi segmentations, supporting ventricle segmentations, and basal ganglia segmentations were reviewed by an expert medical image evaluator.

Participants in this task will have access to combined isometric hyperfine images (in NIFTI .nii.gz format) which have been 9-point linearly registered to their subjects' matching high field scans. Bilateral hippocampi segmentations are in NIFTI .nii.gz format and will be in high field scan space.


# Setup

From root:
```
python codes/1.download_data.py
```

scp va0831@148.100.72.4:/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_01/LISAHF0001segprediction.nii.gz .

scp va0831@148.100.72.4:/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_01/LISAHF0001segprediction.nii.gz C:\Users\TU_USUARIO\Downloads\


scp va0831@148.100.72.4:/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_dynunet_02/LISAHF0001segprediction.nii.gz .

scp va0831@148.100.72.4:/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_03_train/LISAHF0001segprediction.nii.gz .

scp -r va0831@148.100.72.4:/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/efficientnet-b7_mixup_lite_focal_foreground_lovasz_fine/predictions/ .



```bash
python train.py \
  --train_csv results/preprocessed_data/task2/df_train_hipp.csv \
  --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_01 \
  --model unest \
  --folds 3 \
  --epochs 50 \
  --batch_size 4 \
  --gpu 5
```


```bash
python inference.py \
  --test_csv results/preprocessed_data/task2/df_test_hipp.csv \
  --model_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_01/fold_0 \
  --model unest \
  --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_01/
```
"""