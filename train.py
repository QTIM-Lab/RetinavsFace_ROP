import sys
import re
import time
import csv
import copy
from os.path import isfile
from sklearn.metrics import classification_report

from model import *
from utils import *

def train(model_path, dataloaders, dataset_sizes):
    num_epochs = 10

    avg_loss_train = 0.
    avg_acc_train = 0.
    avg_loss_val = 0.
    avg_acc_val = 0.
    best_acc = 0.
    
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    num_batches_train = len(dataloaders['train'])
    num_batches_val = len(dataloaders['val'])

    num_classes = 2
    model = vgg16(num_classes)
    best_model_weights = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr = 1e-3, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    epoch = load_checkpoint(model_path, model) if isfile(model_path) else 0
    filename = re.sub("\.epoch[0-9]+$", "", model_path)
    print("Training model on {} images".format(dataset_sizes['train']))
    print("Validating model on {} images".format(dataset_sizes['val']))
    start_time = time.time()

    for ei in range(epoch + 1, epoch + num_epochs + 1):
        print("Epoch {}/{}".format(ei, num_epochs))
        scheduler.step()

        # train
        model.train()
        loss_train = 0.
        acc_train = 0.
        y_true = []
        y_pred = []
        timer = time.time()
        for i, data in enumerate(train_dataloader):
            print("\rTraining batch {}/{}".format(i+1, num_batches_train), end='', flush=True)
            inputs, labels = data
            if CUDA:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                model.cuda()
            else:
                inputs = Variable(inputs), Variable(labels)
            model.zero_grad()
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            # out, aux_out = outputs[0], outputs[1]
            # _, preds = torch.max(out.data, 1)
            # loss = sum((criterion(o, labels) for o in outputs))
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data).double()
            y_true.extend(labels.data.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        timer = time.time() - timer
        loss_train /= dataset_sizes['train']
        acc_train /= dataset_sizes['train']

        print()
        print("Epoch {} result: ".format(ei))
        print("Avg loss (train): {:.4f}".format(loss_train))
        print("Avg acc (train): {:.4f}".format(acc_train))
        print(classification_report(y_true, y_pred))

        # validation
        model.train(False)
        model.eval()
        loss_val = 0.
        acc_val = 0.
        y_true = []
        y_pred = []

        for i, data in enumerate(val_dataloader):
            print("\rValidating batch {}/{}".format(i+1, num_batches_val), end='', flush=True)
            inputs, labels = data
            if CUDA:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                model.cuda()
            else:
                inputs = Variable(inputs), Variable(labels)
            model.zero_grad()
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data).double()
            y_true.extend(labels.data.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        timer = time.time() - timer
        loss_val /= dataset_sizes['val']
        acc_val /= dataset_sizes['val']

        print()
        print("Avg loss (val): {:.4f}".format(loss_val))
        print("Avg acc (val): {:.4f}".format(acc_val))
        print(classification_report(y_true, y_pred))

        if acc_val > best_acc:
            best_acc = acc_val
            best_model_weights = copy.deepcopy(model.state_dict())

        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_val, timer)
        else:
            save_checkpoint(filename, model, ei, loss_val, timer)

    elapsed_time = time.time() - start_time
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    model_path = sys.argv[1] # model_path
    dataloader = sys.argv[2] # dataloader file
    train(model_path, dataloader)
