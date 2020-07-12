CUDA_VISIBLE_DEVICES=0  python train_2.py --backbone xception \
                                       --lr 0.00005 \
                                       --lr-scheduler poly \
                                       --workers 8 \
                                       --epochs 40 \
                                       --batch-size 2 \
                                       --test-batch-size 4 \
                                       --gpu-ids 0  \
                                       --ft \
                                       --eval-interval 1 \
                                       --dataset laneline \
                                       --loss-type ce_focal \
                                       --base-size 570,1128 \
                                       --crop-size 570,1128 \
                                       --resume ./run/laneline/deeplab-xception/model_best.pth.tar 
#--freeze-bn True \
#--resume ./run/laneline/deeplab-mobilenet/model_best.pth.tar \
# #--resume ./run/laneline/deeplab-mobilenet/model_best.pth.tar \ 
