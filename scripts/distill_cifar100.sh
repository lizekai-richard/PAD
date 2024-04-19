cd distill

CFG="../configs/CIFAR-100/ConvIN/IPC1.yaml"

nohup python3 DATM.py --cfg $CFG \
> ../logs/distill_cifar100_ipc1_rm_hard_20.log 2>&1 &