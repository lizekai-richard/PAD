cd buffer
nohup python buffer_CL.py \
--dataset=CIFAR10 \
--model=ConvNet \
--train_epochs=100 \
--num_experts=50 \
--zca \
--buffer_path="/home/wangkai/big_space/lzk/buffer_storage/cl_75_30/" \
--data_path="../dataset/" \
--sort_method="CIFAR10_Uncertainty" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
--init_ratio=0.75 \
--end_epoch=30 \
--rm_easy='False' \
--add_hard='True' \
--rm_start=50 \
--rate=0.01 \
--max_ratio=0.2 \
> ../logs/train_teacher_trajectories_cifar10_cl_75_30.log 2>&1 &