# Higher-order Heuristic Distillation (H2D)

This temporary repo is the implementation of H2D for the CVPR2023 double-blind submission

* PAPER NUMBER: 8632

## Requirements

To install requirements:

```
# Install env
conda env create -f env.yml

# Activate env
conda activate foryolo
```

## Pre-training

To pre-train the H2D with custom dataset, run this command:

```
python train_h2d_gnn.py --epochs $NUM_EPOCHS --learning_rate $LR --L $L --beta $BETA --n_layers $N_LAYERS_GNN --gpu $GPU_NUM --root_dir $PATH_FOR_DATA --checkpoint_path $PATH_FOR_SAVE
```

For the pre-training on the MS-COCO, run this command:

```
python train_h2d_gnn_coco.py --root_dir ./coco --seed $TORCH_SEED --batch_size $N_BATCH --epochs $NUM_EPOCHS --learning_rate $LR --checkpoint_path $PATH_FOR_SAVE --gpu $GPU_NUM --numk $QUEUE_SIZE --device_ids $MULTI_GPU_LIST --num_workers $N_WORKERS --L $L --beta $BETA
```

## Fine-tuning

To fine-tune the Faster-RCNN model with FPN Resnet-50 backbone on custom dataset, run this command:

```
python main_twostage.py --config config_two_stage_main.json --GPU_NUM $GPU_NUM --nbepoch $NUM_EPOCHS --pretrained $PATH_FOR_WEIGHTS --lr $LR
```

For the fine-tuning on the Pascal VOC, run this command:

```
python train_on_voc.py --GPU_NUM $GPU_NUM --save_dir $PATH_FOR_SAVE --pretrained $PATH_FOR_WEIGHTS
```

## Results

* mean (std) for the 10-fold cross validation

|Method|mAP|mAP_50|mAR|
|---|---|---|---|
|Rand. Init.|84.9 (11.5)|95.8 (8.2)|89.4 (6.6)|
|MOCOv2|86.5 (12.2)|95.5 (10.0)|89.6 (8.9)|
|BYOL|86.6 (11.0)|96.1 (8.1)|90.5 (6.6)|
|Compress|87.0 (12.1)|95.9 (10.30|90.3 (8.5)|
|ISD|87.6 (11.8)|96.4 (9.6)|90.4 (8.5)|
|**H2D**|**87.6 (10.6)**|**96.7 (7.8)**|**90.9 (7.1)**|
