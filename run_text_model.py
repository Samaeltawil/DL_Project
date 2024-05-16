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

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARENT_DIR)

from custom_dataset import CustomDataset
from vilbert_adapt import CustomBert

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
        for i in range(len(metrics_names)):
            curr_metric_name = metrics_names[i]
            print('curr_metric_name', curr_metric_name)
            print('new_metrics_dict[curr_metric_name]', new_metrics_dict[curr_metric_name])
            print('metrics_log[i]', metrics_log[i])
            metrics_log[i].append(new_metrics_dict[curr_metric_name])
        return metrics_log

def train_model(model, train_loader, val_loader, metrics, num_epochs=1, learning_rate=0.001):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)

    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))
    
    train_loss_log,  test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    test_metrics_log = [[] for i in range(len(metrics))]

    criterion = nn.BCELoss() # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i,batch in enumerate(train_loader):
            # print(f"Batch {i}")
            batch = [b.to(device) for b in batch]
            text, mask, img, labels = batch
            optimizer.zero_grad()
            outputs = model(text, mask)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            predicted = (outputs.detach() > 0.5)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            # compute metrics
            # no gradients should be propagated at this step
            with torch.no_grad():
                for k in epoch_metrics.keys():
                    epoch_metrics[k] += metrics[k](predicted, labels)

        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(train_loader)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total

        train_metrics_log[0].append(train_accuracy)

        # clear_output() #clean the prints from previous epochs
        print('train Loss: {:.4f}, '.format(train_loss),
            ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))


        train_loss_log.append(train_loss)
        # train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, epoch_metrics)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                text, mask, img, labels = batch
                outputs = model(text, mask)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item() * text.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        test_metrics_log[0].append(val_accuracy)

        # plot_training(train_loss_log, metrics_names, train_metrics_log)

        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return train_loss_log, train_metrics_log, test_metrics_log

def acc(preds, target):
    return accuracy_score(target.detach().cpu(), preds.detach().cpu())

def save_metrics_log(train_loss_log, train_metrics_log, test_metrics_log, metrics, test_results_path):
    # Convert lists to pandas DataFrame
    train_loss_df = pd.DataFrame(train_loss_log, columns=['train_loss'])
    train_metrics_df = pd.DataFrame(train_metrics_log, columns=['train_' + metric for metric in metrics.keys()])
    test_metrics_df = pd.DataFrame(test_metrics_log, columns=['test_' + metric for metric in metrics.keys()])

    # Concatenate DataFrames
    metrics_df = pd.concat([train_loss_df, train_metrics_df, test_metrics_df], axis=1)

    # Save DataFrame to CSV file
    metrics_df.to_csv(test_results_path, index=False)

def run_text_model():

    dataset_path = PARENT_DIR
    # Load dataset
    image_path = os.path.join(dataset_path, 'dataset/img_tens')
    img_text_path = os.path.join(dataset_path, 'dataset/img_txt')
    json_path = os.path.join(dataset_path, 'dataset/MMHS150K_GT.json')
    GT_path = os.path.join(dataset_path, 'dataset/MMHS150K_Custom.csv')

    # open the csv
    GT_data = pd.read_csv(GT_path)

    # create the splits from ratio
    train_split = 0.1
    val_split = 0.8
    test_split = 0.1

    # create the splits with ratios
    train_data = GT_data.sample(frac=train_split, random_state=42)
    val_data = GT_data.drop(train_data.index).sample(frac=val_split/(val_split + test_split), random_state=42)
    test_data = GT_data.drop(train_data.index).drop(val_data.index)

    train_set = CustomDataset(train_data, image_path, img_text_path)
    val_set = CustomDataset(val_data, image_path, img_text_path)
    test_set = CustomDataset(test_data, image_path, img_text_path)    

    # Define hyperparameters -------------------------------------------------------

    batch_size = 10

    # ------------------------------------------------------------------------------

    print("\n")
    print(f"train_set ratio hate: {sum(train_set.labels)/len(train_set.labels)}")

    # train_set, test_set, val_set = torch.utils.data.dataset.random_split(dataset, [0.1, 0.8, 0.1])
    # Create data loader for training set
    not_hate_indices = []
    hate_indices = []
    for element in zip(train_set.ids, train_set.labels):
        if element[1] == 1:
            hate_indices.append(element[0])
        else:
            not_hate_indices.append(element[0])

    num_not_hate = len(not_hate_indices)
    num_hate = len(hate_indices)
    total_samples = num_not_hate + num_hate

    # Create a WeightedRandomSampler to balance the training dataset
    class_weights = [1-num_hate/total_samples, 1-num_not_hate/ total_samples]  # Inverse of number of samples per class

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
    sampler = WeightedRandomSampler(weights, len(weights))

    # Create data loader for balanced training set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler)

    # Create data loaders for validation and test sets
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    model = CustomBert()

    metrics = {'ACC': acc}
    # Example usage
    print("training start")
    train_loss_log, train_metrics_log, test_metrics_log = train_model(model, train_loader, validation_loader, metrics, num_epochs=3, learning_rate=0.0001)

    print("saving results")
    test_results_path = os.path.join(PARENT_DIR, "results", "fcm_test_results_" + time.strftime("%y%m%d_%H%M") + ".csv",)
    save_metrics_log(train_loss_log, train_metrics_log, test_metrics_log, metrics, test_results_path)

if __name__ == "__main__":
    run_text_model()
    print("Done running text model.")