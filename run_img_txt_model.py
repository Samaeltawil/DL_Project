import torch
import sys
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import time
import os
from PIL import Image
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARENT_DIR)

from custom_dataset import CustomDataset
from ResNet_Bert_Model import ResNet_Bert

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
        for i in range(len(metrics_names)):
            curr_metric_name = metrics_names[i]
            # print('curr_metric_name', curr_metric_name)
            # print('new_metrics_dict[curr_metric_name]', new_metrics_dict[curr_metric_name])
            # print('metrics_log[i]', metrics_log[i])
            # metrics_log[i].append(new_metrics_dict[curr_metric_name])
            metrics_log[i].append(new_metrics_dict[curr_metric_name])
        return metrics_log

def train_model(model, train_loader, test_loader, metrics, num_epochs=1, learning_rate=0.001):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    
    train_loss_log,  eval_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    eval_metrics_log = [[] for i in range(len(metrics))]

    criterion = nn.BCELoss() # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    all_labels = []
    all_outputs = []

    for epoch in range(num_epochs):
        print("\n----------------------------------------------------------------------------")
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"epoch {epoch+1}: {current_time}")
        # Training phase
        model.train()
        epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))
        epoch_loss = 0.0
        for i,batch in enumerate(train_loader):
            # print(f"Batch {i}")
            batch = [b.to(device) for b in batch]
            text, mask, img, labels = batch
            optimizer.zero_grad()
            outputs = model(text, mask, img)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                predicted = (outputs.detach() > 0.5)
                epoch_loss += loss.item()
                for name,metric in metrics.items():
                    epoch_metrics[name] += metric(predicted.float(), labels.float())

        epoch_loss /= len(train_loader)
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(train_loader)

        # print('train Loss: {:.4f}, '.format(epoch_loss),
        #     ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

        train_loss_log.append(epoch_loss)
        train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, epoch_metrics)
        print(f"\ntrain metrics log: {train_metrics_log}")
        
        # Validation phase
        model.eval()
        epoch_metrics_eval = dict(zip(metrics.keys(), torch.zeros(len(metrics))))
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                batch = [b.to(device) for b in batch]
                text, mask, img, labels = batch
                outputs = model(text, mask)
                # print(outputs)
                loss = criterion(outputs, labels.float())
                predicted = (outputs.detach() > 0.5)
                epoch_loss += loss.item()
                for name,metric in metrics.items():
                    epoch_metrics_eval[name] += metric(predicted.float(), labels.float())

                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

            epoch_loss /= len(test_loader)
            for k in epoch_metrics_eval.keys():
                # print(f"\nDEBUG: update div epoch metrics {epoch_metrics_eval[k] / len(test_loader)}")
                epoch_metrics_eval[k] /= len(test_loader)
        
            # print(f"\nDEBUG: {eval_metrics_log}")

            eval_loss_log.append(epoch_loss)
            eval_metrics_log = update_metrics_log(metrics_names, eval_metrics_log, epoch_metrics_eval)
            print(f"\ntest metrics log: {eval_metrics_log}")

    return train_loss_log, train_metrics_log, eval_metrics_log, all_labels, all_outputs

def acc(preds, target):
    return accuracy_score(target.detach().cpu(), preds.detach().cpu())

def f1(preds, target):
    return f1_score(target.detach().cpu(), preds.detach().cpu(), zero_division=1)

def save_metrics_log(train_loss_log, train_metrics_log, eval_metrics_log, metrics, test_results_path):
    # Convert lists to pandas DataFrame
    # Convert tensors to numbers and transpose the list of lists
    train_metrics_log = [[t.item() for t in sublist] for sublist in train_metrics_log]
    eval_metrics_log = [[t.item() for t in sublist] for sublist in eval_metrics_log]

    train_metrics_log_transposed = list(map(list, zip(*train_metrics_log)))
    eval_metrics_log_transposed = list(map(list, zip(*eval_metrics_log)))
    
    train_loss_df = pd.DataFrame(train_loss_log, columns=['train_loss'])
    train_metrics_df = pd.DataFrame(train_metrics_log_transposed, columns=['train_' + metric for metric in metrics.keys()])
    test_metrics_df = pd.DataFrame(eval_metrics_log_transposed, columns=['test_' + metric for metric in metrics.keys()])

    # Concatenate DataFrames
    metrics_df = pd.concat([train_loss_df, train_metrics_df, test_metrics_df], axis=1)

    # Save DataFrame to CSV file
    metrics_df.to_csv(test_results_path, index=False)

    print("\n INFO: metrics saved")

# ==================================================================================================================================

def run_text_model():

    # Define hyperparameters -------------------------------------------------------

    batch_size = 20
    
    # create the splits from ratio
    train_split = 0.8
    test_split = 0.1
    val_split = 0.1
    
    # ------------------------------------------------------------------------------

    print("\n===============================================================================================")
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"start time: {current_time}")
    print("===============================================================================================")

    dataset_path = PARENT_DIR
    # Load dataset
    image_path = os.path.join(dataset_path, 'dataset/img_tens')
    img_text_path = os.path.join(dataset_path, 'dataset/img_txt')
    json_path = os.path.join(dataset_path, 'dataset/MMHS150K_GT.json')
    GT_path = os.path.join(dataset_path, 'dataset/MMHS150K_Custom.csv')

    # open the csv
    GT_data = pd.read_csv(GT_path)

    # create the splits with ratios
    train_data = GT_data.sample(frac=train_split, random_state=42)
    val_data = GT_data.drop(train_data.index).sample(frac=val_split/(val_split + test_split), random_state=42)
    test_data = GT_data.drop(train_data.index).drop(val_data.index)

    train_set = CustomDataset(train_data, image_path, img_text_path)
    test_set = CustomDataset(test_data, image_path, img_text_path)    
    val_set = CustomDataset(val_data, image_path, img_text_path)

    print("\n")
    print(f"test_set ratio hate: {sum(test_set.labels)/len(test_set.labels)}")
    print("\n")

    # balancing the train loader

    # Create a WeightedRandomSampler to balance the training dataset
    train_hate_ratio = sum(train_set.labels)/len(train_set.labels)
    class_weights = [train_hate_ratio, 1-train_hate_ratio]  # Inverse of number of samples per class

    weights = []
    for element in zip(train_set.ids, train_set.labels):
        try:
            label = element[1]
            according_weights = class_weights[int(label)]
            weights.append(according_weights)
        except:
            print(f"Error with idx: {element[0]}")
            print(f"Label: {element[1]}")

    # weights = [class_weights[int(dataset[idx]['label'])] for idx in train_indices]
    sampler_train = WeightedRandomSampler(weights, len(weights))

    # balancing the test loader

    # Create a WeightedRandomSampler to balance the test dataset
    test_hate_ratio = sum(test_set.labels)/len(test_set.labels)
    class_weights = [test_hate_ratio, 1-test_hate_ratio]  # Inverse of number of samples per class

    weights = []
    for element in zip(test_set.ids, test_set.labels):
        try:
            label = element[1]
            according_weights = class_weights[int(label)]
            weights.append(according_weights)
        except:
            print(f"Error with idx: {element[0]}")
            print(f"Label: {element[1]}")

    # weights = [class_weights[int(dataset[idx]['label'])] for idx in train_indices]
    sampler_test = WeightedRandomSampler(weights, len(weights))

    # Create data loader for balanced training set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler_train)

    # Create data loaders for validation and test sets
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=sampler_test)

    print("\nINFO: loader loaded")

    model = ResNet_Bert()

    metrics = {
        'ACC': acc,
        'F1': f1
    }

    print("\nINFO: training start")
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"training start time: {current_time}\n")
    train_loss_log, train_metrics_log, eval_metrics_log, all_labels, all_outputs = train_model(model, train_loader, test_loader, metrics, num_epochs=3, learning_rate=0.0005)
    
    print("\n===============================================================================================")
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"training end: {current_time}")
    print("===============================================================================================")

    print(f"\ncalculating ROC")
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the ROC plot
    results_dir = os.path.join(PARENT_DIR, "results/plots")
    os.makedirs(results_dir, exist_ok=True)
    roc_plot_path = os.path.join(results_dir, "roc_curve_" + time.strftime("%y%m%d_%H%M") + ".png")
    plt.savefig(roc_plot_path)
    plt.close()

    print(f"\nINFO: ROC curve saved at {roc_plot_path}")
    print(f"\ncalculating confusion matrix")

    # Calculate and plot confusion matrix
    predicted_labels = (np.array(all_outputs) > 0.5).astype(int)
    cm = confusion_matrix(all_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)

    # Save the confusion matrix plot
    cm_plot_path = os.path.join(results_dir, "confusion_matrix_" + time.strftime("%y%m%d_%H%M") + ".png")
    plt.savefig(cm_plot_path)
    plt.close()

    print(f"\nINFO: Confusion matrix saved at {cm_plot_path}")

    # print(f"DEBUG: train_metrics_log shape: {np.shape(train_metrics_log)}")

    print("\nINFO: saving results")
    test_results_path = os.path.join(PARENT_DIR, "results", "fcm_test_results_" + time.strftime("%y%m%d_%H%M") + ".csv",)
    save_metrics_log(train_loss_log, train_metrics_log, eval_metrics_log, metrics, test_results_path)

if __name__ == "__main__":
    run_text_model()
    print("Done running text model.")