#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter # Added for TensorBoard logging
from torch.optim.lr_scheduler import ReduceLROnPlateau # Added for learning rate scheduling
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import datetime

from utils import EarlyStopping
from dataset import KeypointDataset
from models import KeypointResnetModel
from visualize import KeypointVisualizer
from loss import LaplaceNLLLoss
from loss import GeneralGaussianNLLLoss


IMG_SIZE = 256
CHECKPOINT_PTH = 'resnet.pth'
LOSS_CHOICES = ['gaussian', 'laplace', 'generalized_gaussian']

# Your existing KeypointDataset class
# Assuming a simple model for demonstration

# Main training script
def train_model(loss_function, note, with_outliers=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Get the current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"resent_{loss_function}_{IMG_SIZE}_with_outliers_{str(with_outliers)}_{timestamp}"
    print ("Starting run ", run_name)
    writer = SummaryWriter(log_dir=os.path.join('runs', run_name))
    
    # The note you want to add for the experiment
    experiment_note = f"""
    imag size {IMG_SIZE} loss {loss_function}
    """ + note
    writer.add_text('Experiment Notes', experiment_note, 0)

    # Hyperparameters
    num_epochs = 300
    learning_rate = 0.001
    batch_size = 64

    # Define your Albumentations transformations
    # A.Compose combines multiple augmentations
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),  # Resize to a fixed size
        A.HorizontalFlip(p=0.5), # Randomly flip the image horizontally
        A.RandomBrightnessContrast(p=0.2), # Adjust brightness and contrast
        A.Affine(
         scale=0.8,  # Single scalar value for scale
         rotate=15,  # Single scalar value for rotation (degrees)
         translate_px=10,  # Single scalar value for translation (pixels)
         p=1.0),
        A.ShotNoise(scale_range=(9.0, 10.0), p=0.1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
         A.RandomGridShuffle(grid=(3, 3), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalize pixel values
    ], keypoint_params=A.KeypointParams(format='xy'))

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),  # Resize to a fixed size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalize pixel values
    ], keypoint_params=A.KeypointParams(format='xy'))


    dataset = KeypointDataset(root_dir='./data', image_size=IMG_SIZE, transform=transform)
    val_dataset = KeypointDataset(root_dir='./data', image_size=IMG_SIZE, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
   
    print (" len dataset ", len(val_dataset))
    print (" len Dalaloader  ", len(val_dataloader))
    # Model, loss function, and optimizer
    if loss_function == 'generalized_gaussian':
        model = KeypointResnetModel(additional_output=True).to(device)
    else:
        model = KeypointResnetModel().to(device)
   
    # Early stopping
    early_stopping = EarlyStopping(patience=20, 
            verbose=True, filename=loss_function+"_with_outliers_"+str(with_outliers)+"_"+CHECKPOINT_PTH)
   
    if loss_function == 'gaussian':
        criterion = torch.nn.GaussianNLLLoss()
    elif loss_function == 'laplace':
        criterion = LaplaceNLLLoss
    elif loss_function == 'generalized_gaussian':
        criterion = GeneralGaussianNLLLoss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Pass only the trainable parameters to the optimizer
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Training loop
    for epoch in range(0, num_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0
        if ((epoch % 21) == 0 ) or (epoch == 1) :
            print (f"{epoch}UnFreezed ")
            model.unfreeze()
            optimizer.param_groups.clear()
            optimizer.state.clear()
            # Get all parameters that are trainable
            all_trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer.add_param_group({'params' : all_trainable_params})

        if (epoch % 20) == 0:
            #Pause the learning of output layer to train uncertainty
            print ("Freezed only output ")
            model.freeze_for_uncertainty()
            optimizer.param_groups.clear()
            optimizer.state.clear()
            # Get all parameters that are trainable
            all_trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer.add_param_group({'params' : all_trainable_params})

        for i, (images, keypoints) in enumerate(dataloader):
            images = images.to(device)
            keypoints = keypoints.to(device)
            keypoints = keypoints.view(-1, 8)
            if with_outliers:
                keypoints = add_outliers(keypoints, 0.05) # Adding 10% outliers

            optimizer.zero_grad()
            if loss_function == 'generalized_gaussian':
                pred_mean, pred_scale, pred_beta = model(images)
                loss = criterion(pred_mean, keypoints, pred_scale, pred_beta)   
            else:
                pred_mean, pred_scale = model(images)
                loss = criterion(pred_mean, keypoints, pred_scale)   
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            # Calculate RMSE
            rmse = torch.sqrt(torch.mean((pred_mean - keypoints) ** 2))
            running_rmse += rmse.item() * images.size(0)
            # Calculate the mean of the predicted variances
            mean_variance = torch.mean(pred_scale)
            max_variance = torch.max(pred_scale)

            # Log loss to TensorBoard
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Metrics/RMSE', rmse.item(), global_step)
            writer.add_scalar('Metrics/Mean Variance', mean_variance.item(), global_step)
            writer.add_scalar('Metrics/Max Variance', max_variance.item(), global_step)
            if loss_function == 'generalized_gaussian':
                mean_beta = torch.mean(pred_beta)
                max_beta = torch.max(pred_beta)
                writer.add_scalar('Metrics/Mean Beta', mean_variance.item(), global_step)
                writer.add_scalar('Metrics/Max Beta', max_variance.item(), global_step)

        epoch_loss = running_loss / len(dataset)
        epoch_rmse = running_rmse / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}")

        # Update learning rate based on validation loss
        scheduler.step(epoch_loss)

        # Check for early stopping
        early_stopping(epoch_loss, model)

        if ((epoch % 10) == 0 ) :
            val_data(val_dataloader, model, epoch, writer, device, loss_function, len(dataset))

        if early_stopping.train_uncertainty:
            model.load_state_dict(torch.load(loss_function+"_with_outliers_"+str(with_outliers)+"_"+CHECKPOINT_PTH, weights_only=True))
            model.to(device)
            print("Early stopping triggered training uncertainty ")
            print ("Freezed only output ")
            model.freeze_for_uncertainty()
            optimizer.param_groups.clear()
            optimizer.state.clear()
            # Get all parameters that are trainable
            all_trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer.add_param_group({'params' : all_trainable_params})

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Finished Training ", timestamp)
    writer.close()
    model.load_state_dict(torch.load(loss_function+"_with_outliers_"+str(with_outliers)+"_"+CHECKPOINT_PTH, weights_only=True))
    model.to(device)


    # Get a batch of data from the dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    images, keypoints = next(iter(dataloader))

    # Set the model to evaluation mode
    model.eval()

    val_data(val_dataloader, model, epoch, writer, device, loss_function, len(dataset))
    # Perform inference
    with torch.no_grad():
        images = images.to(device)
        if loss_function == 'generalized_gaussian':
            pred_mean, pred_scale, pred_beta = model(images)
        else:
            pred_mean, pred_scale = model(images)

    # Reshape predicted keypoints to match the expected format (batch_size, num_keypoints, 2)
    pred_mean = pred_mean.view(-1, 4, 2)
    pred_scale = pred_scale.view(-1, 4, 2)
    if loss_function == 'generalized_gaussian':
        pred_beta = pred_beta.view(-1, 4, 2)
        # The batch size and the middle dimension remain the same.
        pred_scale = torch.cat((pred_scale, pred_beta), dim=2)
        
    visualizer = KeypointVisualizer(distribution=loss_function, img_size=IMG_SIZE)
    visualizer.visualize_batch(
        images.cpu(), keypoints, pred_mean.cpu(), 
        pred_scale.cpu(), show_contours=False
    )

def val_data(dataloader, model, epoch, writer, device, loss_function, len_dataset):
    # Set the model to evaluation mode
    model.eval()
    running_rmse = 0
    for i, (images, keypoints) in enumerate(dataloader):
        images = images.to(device)
        keypoints = keypoints.to(device)
        keypoints = keypoints.view(-1, 8)

        with torch.no_grad():
            if loss_function == 'generalized_gaussian':
                pred_mean, pred_scale, pred_beta = model(images)
            else:
                pred_mean, pred_scale = model(images)
        
            # Calculate RMSE
            rmse = torch.sqrt(torch.mean((pred_mean - keypoints) ** 2))
            running_rmse += rmse.item() * images.size(0)

            # Log loss to TensorBoard
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Metrics/val_RMSE', rmse.item(), global_step)

    epoch_rmse = running_rmse / len_dataset
    print(f"Validation Loss:  RMSE: {epoch_rmse:.4f}")


def load_device_model_data(test_ood: bool, loss_function: str, with_outliers: bool):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Define your Albumentations transformations
    # A.Compose combines multiple augmentations
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),  # Resize to a fixed size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalize pixel values
    ], keypoint_params=A.KeypointParams(format='xy'))
    
    if test_ood:
        dataset = KeypointDataset(root_dir='./ood_dataset', image_size=IMG_SIZE, transform=transform)
    else:
        dataset = KeypointDataset(root_dir='./data', image_size=IMG_SIZE, transform=transform)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model, loss function, and optimizer
    if loss_function == 'generalized_gaussian':
        model = KeypointResnetModel(additional_output=True).to(device)
    else:
        model = KeypointResnetModel().to(device)
   
    model.load_state_dict(torch.load(loss_function+"_with_outliers_"+str(with_outliers)+"_"+CHECKPOINT_PTH, weights_only=True))
    model.to(device)

    return device, model, dataloader

def test_near_ood(loss_function, with_outliers):
    device, model, dataloader = load_device_model_data(test_ood=True, 
                                                       loss_function=loss_function,
                                                       with_outliers=with_outliers)
    # Get a batch of data from the dataloader
    images, keypoints = next(iter(dataloader))

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        images = images.to(device)
        if loss_function == 'generalized_gaussian':
            pred_mean, pred_scale, pred_beta = model(images)
        else:
            pred_mean, pred_scale = model(images)

    # Reshape predicted keypoints to match the expected format (batch_size, num_keypoints, 2)
    pred_mean = pred_mean.view(-1, 4, 2)
    pred_scale = pred_scale.view(-1, 4, 2)
    print ("pred_scale ", pred_scale)
    if loss_function == 'generalized_gaussian':
        pred_beta = pred_beta.view(-1, 4, 2)
        # The batch size and the middle dimension remain the same.
        pred_scale = torch.cat((pred_scale, pred_beta), dim=2)
        
    print (pred_scale)
    visualizer = KeypointVisualizer(distribution=loss_function, img_size=IMG_SIZE)
    visualizer.visualize_batch(
        images.cpu(), keypoints, pred_mean.cpu(), 
        pred_scale.cpu(), show_contours=False, figname='OOD'
    )
    
def add_outliers(keypoint, percentage):
    # Calculate the total number of values
    total_elements = keypoint.numel()

    # Calculate the number of values to change (10%)
    num_to_change = int(total_elements * percentage)

    # Get all possible indices
    all_indices = torch.arange(total_elements)

    # Randomly select the indices to change
    indices_to_change = torch.randperm(total_elements)[:num_to_change]

    # Create a new tensor with the same shape as keypoint
    # This makes sure the new values are in the same range as the old ones
    random_data = torch.randn_like(keypoint.flatten())

    # Use the selected indices to replace values in the original tensor
    keypoint.flatten()[indices_to_change] = random_data[indices_to_change]
    
    return keypoint

def test_adversarial_attack(loss_function):
    device, model, dataloader = load_device_model_data(test_ood=False)
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, dataloader, eps)
        accuracies.append(acc)
        examples.append(ex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss", type=str, choices=LOSS_CHOICES,
                    help="loss function")
    parser.add_argument("--only_test", help="dont train only OOD")
    parser.add_argument("-n", "--note", type=str,
                    help="note for training")
    parser.add_argument("--with_outliers", type=bool, help="train with outliers OOD")
    args = parser.parse_args()

    if not args.only_test:
        train_model(loss_function=args.loss, 
                note=args.note, 
                with_outliers=args.with_outliers)

    torch.cuda.empty_cache() 
    test_near_ood(loss_function=args.loss, with_outliers=args.with_outliers)
    #test_adversarial_attack(loss_function=args.loss)
    torch.cuda.empty_cache() 
