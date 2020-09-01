python test.py \
               --data_config config/face.data \
               --model_def config/face_mask.cfg \
               --weights_path checkpoints/yolov3_ckpt_1.pth \
               --class_path data/face_mask/face.names \
               --conf_thres 0.001 \
               --nms_thres 0.5 \
               --batch_size 8 \
               --n_cpu 0 \
               --img_size 416 \

