CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --lr 0.01 --workers 4 --epochs 40 --batch-size 16 --gpu-ids 0  --eval-interval 1 --dataset laneline

