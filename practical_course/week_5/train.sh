CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet \
                                       --lr 0.01 \
                                       --workers 6 \
                                       --epochs 40 \
                                       --batch-size 32 \
                                       --gpu-ids 0  \
                                       --eval-interval 1 \
                                       --dataset laneline \
                                       --loss-type ce \
                                       --resume ./run/laneline/deeplab-mobilenet/model_best.pth.tar \
                                       --base-size 1024 \
                                       --crop-size 513 





# #--resume ./run/laneline/deeplab-mobilenet/model_best.pth.tar \ 
