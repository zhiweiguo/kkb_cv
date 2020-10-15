CUDA_VISIBLE_DEVICES=0 python train_Fusion_CyclicLR.py --model=baseline_18 --batch_size=256 --image_size=48 --cycle_inter=20 --cycle_num=5
# | tee model_A.log
