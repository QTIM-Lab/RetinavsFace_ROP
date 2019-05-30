import sys
import re
import time
import csv
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from model import *
from utils import *

def evaluate(model_path, dataloaders, dataset_sizes):
    num_classes = 2
    num_batches = len(dataloaders['test'])
    data_size = dataset_sizes['test']
    criterion = nn.CrossEntropyLoss()
    model = vgg16(num_classes)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.train(False)
    model.eval()
    print("Evaluating model on {} images".format(data_size))
    start_time = time.time()

    loss_test = 0.
    acc_test = 0.
    y_true = []
    y_pred = []
    
    csv_file_name = model_path.split('/')[-1] + '_' + str(data_size) + '_predictions.csv'
    with open(csv_file_name, mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image', 'true', 'prediction'])

        with torch.no_grad():
            for i, data in enumerate(dataloaders['test']):
                print("\rEvaluating batch {}/{}".format(i+1, num_batches), end='', flush=True)
                inputs, labels, img_names = data
                if CUDA:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    model.cuda()
                else:
                    inputs = Variable(inputs).float(), Variable(labels)
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss_test += loss.item()
                acc_test += torch.sum(preds == labels.data).double()
                y_true.extend(labels.data.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                for i in range(len(img_names)):
                    writer.writerow([img_names[i], labels.data[i].item(), preds[i].item()])
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

    loss_test /= data_size
    acc_test /= data_size
    elapsed_time = time.time() - start_time
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(loss_test))
    print("Avg acc (test): {:.4f}".format(acc_test))
    print("ROC AUC (test): {:.4f}".format(roc_auc_score(y_true, y_pred)))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    model_path = sys.argv[1] # model_path
    dataloader = sys.argv[2] # dataloader file
    evaluate(model_path, dataloader)