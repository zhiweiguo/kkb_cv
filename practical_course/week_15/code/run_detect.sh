python detect.py \
               --image_folder data/face_samples \
               --model_def config/face_mask.cfg \
               --weights_path checkpoints/yolov3_ckpt_43_0.8610.pth \
               --class_path data/face_mask/face.names \
               --conf_thres 0.8 \
               --nms_thres 0.4 \
               --batch_size 1 \
               --n_cpu 0 \
               --img_size 416 \
               --active_mode 2 \
               --regu_mode 0 \
               --checkpoint_model checkpoints/yolov3_ckpt_43_0.8610.pth

