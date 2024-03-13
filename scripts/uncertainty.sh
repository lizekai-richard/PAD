CUDA_VISIBLE_DEVICES="1,2" 
cd deepcore

nohup python -u main.py \
--fraction 0.1 \
--dataset CIFAR10 \
--data_path ../dataset \
--num_exp 5 \
--workers 10 \
--optimizer SGD \
-se 10 \
--selection Uncertainty \
--model ResNet18 \
--lr 0.1 -sp ./result \
--batch 128 \
--uncertainty Entropy \
--indices_save_dir "../data_selection/data_indices/" \
--balance False \
> ../logs/deepcore_uncertainty_cifar10.log 2>&1 &