MACHINE_IDX=0
TOTAL=17020695
EVERY=212758
OFFSET=4
GPUS=$(seq ${OFFSET} 7)
for gpu in ${GPUS}
do
    gpu_idx=`expr ${MACHINE_IDX} '*' 8 + ${gpu}`
    start=`expr ${EVERY} '*' ${gpu_idx}`
    end=`expr ${start} + ${EVERY}`
    device=`expr ${gpu} '-' ${OFFSET}`
    echo "CUDA_VISIBLE_DEVICES=${device} python compute_distance.py --start_bb ${start} --end_bb ${end} --batch_step 12 --log_path logs-ssim/machine_${MACHINE_IDX}_gpu${gpu}.log --img_dir ~/data/FFHQ/images1024x1024 --out_dir dists-ssim/dist-${start}-${end}.pkl"
    CUDA_VISIBLE_DEVICES=${device} python compute_distance.py --start_bb ${start} --end_bb ${end} --batch_step 12 --log_path logs-ssim/machine_${MACHINE_IDX}_gpu${gpu}.log --img_dir ~/data/FFHQ/images1024x1024 --out_dir dists-ssim/dist-${start}-${end}.pkl > /dev/null 2>&1 &
done