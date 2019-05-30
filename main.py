import sys

from utils import *
from prepare import *
from model import *
from train import *
from evaluate import *
from predict import *

if __name__ == "__main__":
    action = sys.argv[1] # prepare, train, eval, predict
    data_dir = sys.argv[2] # file of train, test data
    model_path = sys.argv[3] # model path save or load
    # TODO: batch_size, num_epochs

    if action == 'train':
        image_dataloaders, dataset_sizes, class_names = prepare(data_dir)
        train(model_path, image_dataloaders, dataset_sizes)
    elif action == 'eval':
        image_dataloaders, dataset_sizes, class_names = prepare(data_dir)
        evaluate(model_path, image_dataloaders, dataset_sizes)
    elif action == 'predict':
        predict(model_path, data_dir)