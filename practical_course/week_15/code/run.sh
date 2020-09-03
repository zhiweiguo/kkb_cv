CUDA_VISIBLE_DEVICES=3 \
python train.py --model_def config/face_mask.cfg \
                --data_config config/face.data \
                --batch_size 4 \
                --n_cpu 8 \
                --data_aug_mode 1 \
                --regu_mode 0 \
                --start_epoch 8 \
                --active_mode 2 \
                --pretrained_weights checkpoints/yolov3_ckpt_9_0.7316.pth \
                --multiscale_training False
