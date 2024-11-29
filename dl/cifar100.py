import torch, torchvision
import torch.utils.data.dataloader
import torchvision.transforms as tt

class cifar100() :
    def __init__(self, data_dir):
        self.dir = data_dir

    def load_cifar100(self, batch_size=128, is_train=True, num_workers=4) :
        if is_train :
            transform = tt.Compose([tt.Resize(256),
                        tt.RandomHorizontalFlip(),
                        tt.RandomVerticalFlip(),
                        tt.RandomRotation(degrees=45),
                        tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        tt.CenterCrop(224),
                        tt.ToTensor(),
                        tt.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
        else :
            transform = tt.Compose([tt.Resize(224),
                                    tt.ToTensor(),
                                    tt.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])

        dset = torchvision.datasets.CIFAR100(self.dir, train=is_train, download=True, transform=transform);

        dloader = torch.utils.data.DataLoader(
            dset, batch_size, shuffle=is_train,
            num_workers=num_workers, pin_memory=True,
            drop_last=is_train)

        return dloader