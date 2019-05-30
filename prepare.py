import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import datasets, transforms, utils

def prepare(data_dir):
    class ImageFolderWithPaths(datasets.ImageFolder):
        """Extends ImageFolder to include image file paths"""
        def __getitem__(self, index):
            try:
                original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
                path = self.imgs[index][0]
                tuple_with_path = (original_tuple + (path,))
            except:
                tuple_with_path = None
            return tuple_with_path

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'), 
            transform=train_transform
        )

    val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'), 
            transform=val_transform
        )

    test_dataset = ImageFolderWithPaths(
            os.path.join(data_dir, 'test'), 
            transform=test_transform
        )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)

    image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    image_dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    class_names = image_datasets['train'].classes
    return image_dataloaders, dataset_sizes, class_names


if __name__ == "__main__":
    action = sys.argv[1] # prepare, train, eval, predict
    data_dir = sys.argv[2] # file of train, test data
    dataloader_file = sys.argv[4] # model path to save or load
    prepare(action, data_dir, csv_file, dataloader_file)

