# Higher-order Heuristic Distillation (H2D)

This temporary repo is the implementation of H2D for the CVPR2023 double-blind submission

* PAPER NUMBER: 

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

|Method|mAP|mAP_50|mAR|
|Rand. Init.|84.9 \pm 11.5|||
