from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import torch

def check():

    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = datasets.CIFAR10("./dataset/", train=True, download=True, transform=transform)  # no augmentation
    dst_test = datasets.CIFAR10("./dataset", train=False, download=True, transform=transform)

    indices_file_path = "./data_selection/data_indices/CIFAR10_Uncertainty.pt"
    sorted_diff_indices = torch.load(indices_file_path)
    assert torch.unique(sorted_diff_indices).size(0) == 50000

    easy_images = []
    hard_images = []
    # for c in range(num_classes):
    easiest, hardest = sorted_diff_indices[4][:10], sorted_diff_indices[4][-10:]
    for e_i, h_i in zip(easiest, hardest):
        easy_images.append(dst_train[e_i][0])
        hard_images.append(dst_train[h_i][0])
        # e_imgs, h_imgs = dst_train[easiest][0], dst_train[hardest][0]
    # easy_images.append(e_img)
    # hard_images.append(h_img)
    # images = easy_images + hard_images
    save_image(easy_images, 'easy_visual.png')
    save_image(hard_images, 'hard_visual.png')


if __name__ == '__main__':
    check()