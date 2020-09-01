CUDA_VISIBLE_DEVICES=0 \
python train.py --model_def config/face_mask.cfg \
                --data_config config/face.data \
                --batch_size 16 \
                --n_cpu 8 \
                --data_aug_mode 4 \
                --pretrained_weights checkpoints/yolov3_ckpt_11.pth \
                --regu_mode 3 \
                --start_epoch 12 \
                --multiscale_training True
