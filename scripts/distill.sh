cd distill
CFG="../configs/CIFAR-10/ConvIN/IPC500.yaml"

nohup python3 DATM.py --cfg $CFG \
> ../logs/distill_cifar10_ipc500_convnet_cl_v2_convnet_loss.log 2>&1 &