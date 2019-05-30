import sys
import re
import time
import csv
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import datasets, transforms, utils

from model import *

def predict(model_path, data_dir):
    data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

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

    image_dataset = ImageFolderWithPaths(data_dir, transform=data_transform)
    class_names = ['face', 'retina']
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, 
        collate_fn=(lambda x: torch.utils.data.dataloader.default_collate(list(filter(lambda y: y is not None, x)))))
    dataset_size = len(image_dataset)
    print("Predicting on {} images".format(dataset_size))

    num_classes = len(class_names)
    model = vgg16(num_classes)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.train(False)
    model.eval()
    start_time = time.time()
    
    csv_file_name = 'face_retina_predictions.csv'
    with open(csv_file_name, mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image', 'prediction'])

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print("\rEvaluating batch {}/{}".format(i+1, len(dataloader)), end='', flush=True)

                inputs, _, img_paths = data

                if CUDA:
                    inputs = Variable(inputs.cuda())
                    model.cuda()
                else:
                    inputs = Variable(inputs)

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                predicted_labels = [preds[j] for j in range(inputs.size()[0])]

                for i in range(len(img_paths)):
                    writer.writerow([img_paths[i], class_names[predicted_labels[i].item()]])
                
                del inputs, outputs, preds, predicted_labels
                torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    print("Prediction completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    model_path = 'models/vgg16.epoch10'
    data_dir = sys.argv[1] # data directory
    predict(model_path, data_dir)