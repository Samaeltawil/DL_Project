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
from ultralytics import YOLO

MODEL_NAME = "CustomBert"

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARENT_DIR)

from custom_dataset import CustomDataset
from custom_models import CustomBert, ResNet_Bert, ResNet50

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
        for i in range(len(metrics_names)):
            curr_metric_name = metrics_names[i]
            metrics_log[i].append(new_metrics_dict[curr_metric_name])
        return metrics_log

def plot_loss(train_loss_log, test_loss_log, name="loss_plot_unamed", save_path=os.path.join(PARENT_DIR, "results/plots")):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure()
    plt.plot(train_loss_log, label='Train Loss')
    plt.plot(test_loss_log, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.close()

def save_labels_outputs(labels, outputs, save_path):
    """
    Save the labels and outputs to a CSV file.

    Parameters:
    - labels (list): List of labels.
    - outputs (list): List of outputs.
    - save_path (str): Path to save the CSV file.
    """
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert lists to a DataFrame
    data = {
        'labels': labels,
        'outputs': outputs
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(save_path, index=False)

    print(f"\nINFO: Labels and outputs saved at {save_path}")

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
            if model.name == "CustomBert":
                outputs = model(text, mask)
            elif model.name == "ResNet_Bert":
                outputs = model(text, mask, img)
            elif model.name == "ResNet50":
                outputs = model(img)
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

        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for batch in test_loader:
                batch = [b.to(device) for b in batch]
                text, mask, img, labels = batch
                if model.name == "CustomBert":
                    outputs = model(text, mask)
                elif model.name == "ResNet_Bert":
                    outputs = model(text, mask, img)
                elif model.name == "ResNet50":
                    outputs = model(img)
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
    
    plot_loss(train_loss_log, eval_loss_log, name=(f"loss_plot_{model.name}_" + time.strftime("%y%m%d_%H%M")))

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

def save_split(loader, split_name):
    # create a folder saving the images used for training
    images_path = os.path.join(PARENT_DIR, "dataset", split_name)
    os.makedirs(images_path, exist_ok=True)

    # create a folder for hate and non-hate images
    hate_path = os.path.join(images_path, "hate")
    non_hate_path = os.path.join(images_path, "non_hate")
    os.makedirs(hate_path, exist_ok=True)
    os.makedirs(non_hate_path, exist_ok=True)

    # clear the folders if they are not empty
    for file in os.listdir(hate_path):
        os.remove(os.path.join(hate_path, file))
    for file in os.listdir(non_hate_path):
        os.remove(os.path.join(non_hate_path, file))

    # copy the images to the folder
    # for batch in train_loader:
    for j, batch in enumerate(loader):
        # print(f"Batch {j}/{len(loader)}")
        _, _, img, labels = batch
        for i in range(len(img)):
            if labels[i] == 1:
                img_path = os.path.join(hate_path, f"{j}_{i}.jpg")
            else:
                img_path = os.path.join(non_hate_path, f"{j}_{i}.jpg")
            img_pil = transforms.ToPILImage()(img[i])
            img_pil.save(img_path)

    print(f"\nINFO: {split_name} split saved")

    return os.path.join(PARENT_DIR, "dataset")



# ==================================================================================================================================

def run_model(model_name="CustomBert"):

    # Define hyperparameters -------------------------------------------------------

    batch_size = 32
    
    # create the splits from ratio
    train_split = 0.93 # 93% train, 7% test
    
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
    test_data = GT_data.drop(train_data.index)

    train_set = CustomDataset(train_data, image_path, img_text_path)
    test_set = CustomDataset(test_data, image_path, img_text_path)    

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

    sampler_test = WeightedRandomSampler(weights, len(weights))

    # Create data loader for balanced training set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler_train)

    # Create data loaders for validation and test sets
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=sampler_test)

    print("\nINFO: loader loaded")

    if model_name == "CustomBert":
        model = CustomBert()
    elif model_name == "ResNet_Bert":
        model = ResNet_Bert()
    elif model_name == "ResNet50":
        model = ResNet50()
    elif model_name == "Yolov8":
        model = YOLO('yolov8-cls.yaml')
        save_split(train_loader, "train")
        dt_path = save_split(test_loader, "test")

    # print(f"\nINFO: model used: {model.name}")

    metrics = {
        'ACC': acc,
        'F1': f1
    }

    print("\nINFO: training start")
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"training start time: {current_time}\n")
    if model_name == "Yolov8":
        results = model.train(data = dt_path, epochs = 15, batch=32, imgsz = 64)   ## Train the Model
        print(f"\nINFO: Yolov8 training results: {results}")
        return
    else:
        train_loss_log, train_metrics_log, eval_metrics_log, all_labels, all_outputs = train_model(model, train_loader, test_loader, metrics, num_epochs=15, learning_rate=0.0005)
    
    print("\n===============================================================================================")
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"training end: {current_time}")
    print("===============================================================================================")

    # Saving all labels, all outputs
    labels_outputs_path = os.path.join(PARENT_DIR, "results", f"labels_outputs_{model.name}_" + time.strftime("%y%m%d_%H%M") + ".csv")
    save_labels_outputs(all_labels, all_outputs, labels_outputs_path)

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
    plt.title(f'Receiver Operating Characteristic {model.name}')
    plt.legend(loc="lower right")

    # Save the ROC plot
    results_dir = os.path.join(PARENT_DIR, "results/plots")
    os.makedirs(results_dir, exist_ok=True)
    roc_plot_path = os.path.join(results_dir, f"roc_curve_{model.name}_" + time.strftime("%y%m%d_%H%M") + ".png")
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
    cm_plot_path = os.path.join(results_dir, f"confusion_matrix_{model.name}_" + time.strftime("%y%m%d_%H%M") + ".png")
    plt.savefig(cm_plot_path)
    plt.close()

    print(f"\nINFO: Confusion matrix saved at {cm_plot_path}")

    # print(f"DEBUG: train_metrics_log shape: {np.shape(train_metrics_log)}")

    print("\nINFO: saving results")
    test_results_path = os.path.join(PARENT_DIR, "results", "fcm_test_results_" + time.strftime("%y%m%d_%H%M") + ".csv",)
    save_metrics_log(train_loss_log, train_metrics_log, eval_metrics_log, metrics, test_results_path)

if __name__ == "__main__":
    run_model(MODEL_NAME)
    print("Done running model.")
