cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC10.yaml"

nohup python3 DATM_param.py --cfg $CFG \
> ../logs/distill_cifar10_ipc10_datm_20exp_param_match_last_25.log 2>&1 &