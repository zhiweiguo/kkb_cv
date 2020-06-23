CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet \
                                       --lr 0.001 \
                                       --lr-scheduler poly \
                                       --workers 8 \
                                       --epochs 40 \
                                       --batch-size 32 \
                                       --test-batch-size 32 \
                                       --gpu-ids 0  \
                                       --eval-interval 1 \
                                       --dataset laneline \
                                       --loss-type ce \
                                       --base-size 512 \
                                       --crop-size 512 
                                       #--no-val
                                       #--ft


#--resume ./run/laneline/deeplab-mobilenet/experiment_3/checkpoint.pth.tar \
#--resume ./run/laneline/deeplab-mobilenet/model_best.pth.tar \
# #--resume ./run/laneline/deeplab-mobilenet/model_best.pth.tar \ 
