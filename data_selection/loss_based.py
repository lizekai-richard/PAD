import os
import argparse
import sys 
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.utils_gsam import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
from buffer.gsam import GSAM, LinearScheduler, CosineScheduler, ProportionScheduler
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class LossSortDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, indices):
        super().__init__()
        self.images = images
        self.labels = labels
        self.indices = indices

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        index = self.indices[item]

        return image, label, index
    
    def __len__(self):
        return len(self.indices)


def train_models(args):

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)


    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)

    candidate_models = ['ConvNet', 'ResNet18', 'AlexNet', 'VGG11', 'ResNet18BN', 'VGG11BN']

    for model in candidate_models:
        args.model = model
        save_dir = "/home/kwang/zekai/DATM/data_selection/loss_eval_models/{}_CIFAR10.pth".format(model)
        print(save_dir)

        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        teacher_net.train()
        lr = args.lr_teacher

        base_optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer,step_size=args.train_epochs*len(trainloader),gamma=1)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=lr, min_lr=lr,
            max_value=args.rho_max, min_value=args.rho_min)
        teacher_optim = GSAM(params=teacher_net.parameters(), base_optimizer=base_optimizer, 
                model=teacher_net, gsam_alpha=args.alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):

            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, scheduler=scheduler, aug=True)

            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, scheduler=scheduler, aug=False)

            print("Model: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(model, e, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()
        
        torch.save(teacher_net.state_dict(), save_dir)
    

def sort_data_by_loss(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, \
    testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, 
                                                                          args.data_path, 
                                                                          args.batch_real, 
                                                                          args.subset, 
                                                                          args=args)
    images_all = []
    labels_all = []
    indices_all = []
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
        indices_all.append(i)

    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    indices_all = torch.tensor(indices_all, dtype=torch.long, device="cpu")

    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

    # dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()), 
    #                           copy.deepcopy(indices_all.detach()))
    dst_train = LossSortDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()), 
                                copy.deepcopy(indices_all.detach()))
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    candidate_models = ['ConvNet', 'ResNet18', 'AlexNet', 'VGG11', 'ResNet18BN', 'VGG11BN']
    indice_to_losses = {k: [] for k in range(images_all.size(0))}
    print(len(indice_to_losses.keys()))
    for model in candidate_models:
        
        save_path = "/home/kwang/zekai/DATM/data_selection/loss_eval_models/{}_CIFAR10.pth".format(model)
        model_state_dict = torch.load(save_path)
        model = get_network(model, channel, num_classes, im_size).to(args.device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(train_loader):
                images = batch[0].float().to(args.device)
                labels = batch[1].long().to(args.device)
                indices = batch[2].numpy()

                output = model(images)
                losses = criterion(output, labels).cpu().numpy()
                
                for idx, loss in zip(indices, losses):
                    indice_to_losses[idx].append(loss)
    
    for i in indice_to_losses.keys():
        indice_to_losses[i] = np.mean(indice_to_losses[i])
    
    sorted_indices_with_loss = sorted(indice_to_losses.items(), key=lambda x: -x[1])
    sorted_indices = torch.tensor([x[0] for x in sorted_indices_with_loss], dtype=torch.long)

    indices_save_path = "/home/kwang/zekai/DATM/data_selection/data_indices/CIFAR10_AvgModels.pt"
    torch.save(sorted_indices, indices_save_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument("--rho_max", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--rho_min", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--indices_save_dir', type=str, default="data_indices")
    parser.add_argument('--model_save_path', type=str, default="loss_eval_models")

    args = parser.parse_args()
    # train_models(args)
    sort_data_by_loss(args)
