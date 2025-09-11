import argparse
import cv2
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns

import scipy.stats
from scipy.stats import norm, laplace, gennorm

import torch
import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from dataset import KeypointDataset
from models import KeypointResnetModel
from visualize import KeypointVisualizer
import tqdm

cm = 1/2.54  # centimeters in inches
sns.set()
sns.set_style("white")
sns.set_style("ticks")
sns.despine(trim=True)
sns.set_context("paper")
sns.color_palette("tab10")

parser = argparse.ArgumentParser()
parser.add_argument("--load-pkl", action='store_true',
                    help="Load predictions for a cached pickle file or \
                        recompute from scratch by feeding the data through \
                        trained models")
args = parser.parse_args()

IMG_SIZE = 256

class Model(Enum):
    GroundTruth = "GroundTruth"
    Gaussian = "Gaussian"
    Evidential = "Evidential"
    Laplace = "Laplace"
    Generalized = "Generalized"


save_dir = "trained_models"
output_dir = "figs/keypoint"

trained_models = {
    Model.Gaussian: [
        ["gaussian/gaussian_with_outliers_None_resnet.pth", 0.0],
        ["gaussian/gaussian_with_outliers_True_resnet.pth", 5.0],
    ],
    Model.Evidential: [
        ["evidential/evidential_with_outliers_None_resnet.pth", 0.0],
        ["evidential/evidential_with_outliers_True_resnet.pth", 5.0],
    ],
    Model.Laplace: [
        ["laplace/laplace_with_outliers_None_resnet.pth", 0.0],
        ["laplace/laplace_with_outliers_True_resnet.pth", 5.0],
    ],
    Model.Generalized: [
        ["generalized/generalized_gaussian_with_outliers_None_resnet.pth", 0.0],
        ["generalized/generalized_gaussian_with_outliers_True_resnet.pth", 5.0],
    ],
}

def load_model(method, check_point_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == Model.Generalized:
        model = KeypointResnetModel(additional_output=True).to(device)
    elif method == Model.Evidential:
        model = KeypointResnetModel(is_evidential=True).to(device)
    else:
        model = KeypointResnetModel().to(device)
    model.load_state_dict(torch.load(check_point_path, weights_only=True))
    model.to(device)

    return model 


def compute_predictions(batch_size=16, n_adv=3):
     # Set devic3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define your Albumentations transformations
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),  # Resize to a fixed size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalize pixel values
    ], keypoint_params=A.KeypointParams(format='xy'))
    
    ood_dataset = KeypointDataset(root_dir='./ood_dataset', image_size=IMG_SIZE, transform=transform)
    dataset = KeypointDataset(root_dir='./data', image_size=IMG_SIZE, transform=transform)

    # dataloade for data
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # dataloader for OOD data
    ood_dataloader = DataLoader(ood_dataset, batch_size=32, shuffle=True)

    adv_eps = np.linspace(0, 0.04, n_adv)
    adv_eps = [0.0]
    all_summaries = []
    for method, model_path_list in trained_models.items():
        for model_path, noise_percentage in model_path_list:
            print (f'noisy percentage {noise_percentage}')
            full_path = os.path.join(save_dir, model_path)
            model = load_model(method, full_path)
            for epsilon in adv_eps:
                batch_summary = get_prediction_summary(model, dataloader, device, method, model_path, eps=epsilon, outliers=noise_percentage)
                all_summaries.extend(batch_summary)
            batch_summary = get_prediction_summary(model, ood_dataloader, device, method, model_path, ood=True, outliers=noise_percentage)
            all_summaries.extend(batch_summary)

    df_pred_image = pd.DataFrame(all_summaries)
    print (df_pred_image.head())
    return df_pred_image

def get_prediction_summary(model, dataloader, device, method, model_path, eps=0.0, ood=False, outliers=0.0):
    # Set the model to evaluation mode
    model.eval()
    batch_summaries = []
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i, (images, keypoints) in enumerate(dataloader):
        images = images.to(device)
        images.requires_grad = True
        keypoints = keypoints.view(-1, 8)
        keypoints = keypoints.to(device)
        # Perform inference
        if method == Model.Generalized:
            pred_mean, pred_scale, beta = model(images)
        elif method == Model.Evidential:
            pred_mean, pred_scale, alpha, beta = model(images)
            pred_scale = beta/(pred_scale*(alpha-1)) #variance
        else:
            pred_mean, pred_scale = model(images)
            beta = torch.ones_like(pred_scale)

        # Ieps is higher then do adversarial attack
        if (eps > 0.0 ) :
            loss = torch.nn.functional.mse_loss(pred_mean, keypoints)
            model.zero_grad()
            loss.backward()
            data_grad = images.grad.data
            #denorm image 
            #image = image_tensor.permute(1, 2, 0)
            # Convert mean and std to tensors for easy computation
            mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
            std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
            images = images * std_tensor + mean_tensor
            images = torch.clip(images, 0, 1)
            # Collect the element-wise sign of the data gradient
            sign_data_grad = data_grad.sign()
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_images = images + eps*sign_data_grad
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            # Reapply normalization
            perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_images)
            # Re-classify the perturbed image
            if method == Model.Generalized:
                pred_mean, pred_scale, beta = model(perturbed_data_normalized)
            elif method == Model.Evidential:
                pred_mean, pred_scale, alpha, beta = model(images)
                pred_scale = beta/(pred_scale*(alpha-1)) #variance
            else:
                pred_mean, pred_scale = model(perturbed_data_normalized)
                beta = torch.ones_like(pred_scale)
 
        pred_mean = pred_mean.detach()
        pred_scale = pred_scale.detach()
        beta = beta.detach()
        ### Save the predictions to some dataframes for later analysis
        summary = [{"Method": method.value, "Model Path": model_path,
            "Keypoint": y, "Mu": mu, "Var": var,"Beta": beta, 
            "Epsilon": eps, "OOD": ood, "Outliers": outliers}
            for y,mu,var,beta in zip(keypoints.cpu().numpy(), pred_mean.cpu().numpy(), pred_scale.cpu().numpy(), beta.cpu().numpy())]
        batch_summaries.extend(summary)
    return batch_summaries

def gen_interval_score_plot(df_image):
    print(f"Generating Interval score")
    df_pixel = df_image[(df_image["OOD"]==False) & ((df_image["Epsilon"]==0.0) | (df_image["Epsilon"]==0.02) | (df_image["Epsilon"]==0.04))]

    df_pixel = df_pixel[['Method', 'Epsilon', 'Keypoint', 'Mu', 'Var', 'Beta', 'Outliers']]
    df_pixel = df_pixel.explode(['Keypoint', 'Mu', 'Var', 'Beta'])
    df_pixel = df_pixel.astype({"Keypoint": float, "Mu": float, "Var": float, "Beta": float, "Outliers": float})
    df_pixel = df_pixel.replace({'Outliers': {0.0: 'Clean', 5.0: 'Outliers'}})
    print ("Unique Adv : ", df_pixel["Epsilon"].unique())
    print ("Unique Adv : ", df_pixel["Outliers"].unique())

    print ("Generating RMSE Score")
    df_pixel["RMSE"] = (df_pixel["Mu"] - df_pixel["Keypoint"])**2
    #g.set(yscale="log")
    #plt.savefig(os.path.join(output_dir, f"RMSE_Adv_box_Keypoint_logscale.pdf"))
    #plt.show()

    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=(14.2*cm/2.0,14.2*cm/2.0))
    #g = sns.catplot(hue="Outliers", y="RMSE", x="Method", data=df_pixel, kind="box", whis=0.5, showfliers=False)
    g = sns.boxplot(hue="Outliers", x="RMSE", y="Method", data=df_pixel, whis=0.5, showfliers=False)
    plt.legend(fontsize='xx-small')
    plt.savefig(os.path.join(output_dir, f"RMSE_Comparison_Keypoint.pdf"), bbox_inches='tight')
    plt.show()

    print (f"Generating Interval Score")

    df_pixel["Mu"] = df_pixel["Mu"] * IMG_SIZE
    df_pixel["Keypoint"] = df_pixel["Keypoint"] * IMG_SIZE
    # Calculate the 95% interval regions
    # This corresponds to the 2.5th and 97.5th percentiles of the distributions.
    lower_percentile = 0.025
    upper_percentile = 0.975

    df_pixel["lower"] = df_pixel['Beta']
    df_pixel["lower"].mask(df_pixel["Method"]=="Gaussian", 
            norm.ppf(lower_percentile , loc=df_pixel['Mu'], scale=np.sqrt(df_pixel['Var'])), inplace=True )
    df_pixel["lower"].mask(df_pixel["Method"]=="Evidential", 
            norm.ppf(lower_percentile , loc=df_pixel['Mu'], scale=np.sqrt(df_pixel['Var'])), inplace=True )
    df_pixel["lower"].mask(df_pixel["Method"]=="Laplace", 
            laplace.ppf(lower_percentile , loc=df_pixel['Mu'], scale=df_pixel['Var']), inplace=True)
    df_pixel["lower"].mask(df_pixel["Method"]=="Generalized", 
            gennorm.ppf(lower_percentile , loc=df_pixel['Mu'], scale=df_pixel['Var'], beta=df_pixel['Beta']), inplace=True)
    df_pixel["upper"] = df_pixel['Beta']
    df_pixel["upper"].mask(df_pixel["Method"]=="Gaussian", 
            norm.ppf(upper_percentile , loc=df_pixel['Mu'], scale=np.sqrt(df_pixel['Var'])), inplace=True)
    df_pixel["upper"].mask(df_pixel["Method"]=="Evidential", 
            norm.ppf(upper_percentile , loc=df_pixel['Mu'], scale=np.sqrt(df_pixel['Var'])), inplace=True)
    df_pixel["upper"].mask(df_pixel["Method"]=="Laplace", 
            laplace.ppf(upper_percentile , loc=df_pixel['Mu'], scale=df_pixel['Var']), inplace=True)
    df_pixel["upper"].mask(df_pixel["Method"]=="Generalized", 
            gennorm.ppf(upper_percentile , loc=df_pixel['Mu'], scale=df_pixel['Var'], beta=df_pixel['Beta']), inplace=True)
    
    df_pixel["Interval Score"] = df_pixel["upper"] - df_pixel["lower"] \
     + (2/0.95)*(df_pixel["lower"]-df_pixel["Keypoint"])*(df_pixel["Keypoint"]<df_pixel["lower"]) \
     + (2/0.95)*(df_pixel["Keypoint"] - df_pixel["upper"])*(df_pixel["Keypoint"]>df_pixel["upper"])
    
    df_pixel["Entropy"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_pixel["Var"]))
    print ("Entropy inf count :",np.sum(np.isinf(df_pixel['Entropy'])))
    df_pixel["Entropy"].mask(df_pixel["Method"]=="Gaussian", norm.entropy(loc=df_pixel["Mu"], scale=np.sqrt(df_pixel["Var"])) ) #  entropy for laplace distirbution
    df_pixel["Entropy"].mask(df_pixel["Method"]=="Laplace",  laplace.entropy(loc=df_pixel["Mu"], scale=df_pixel["Var"]) ) #  entropy for laplace distirbution
    df_pixel["Entropy"].mask(df_pixel["Method"]=="Generalized",  gennorm.entropy(loc=df_pixel["Mu"], scale=df_pixel["Var"], beta=df_pixel["Beta"]) ) #  entropy for laplace distirbution


    print ('#################Interval Score ###################')
    print (df_pixel[df_pixel['Epsilon'] == 0.0].groupby(["Method", "Outliers"])['Interval Score'].describe() )
    print ('################# RMSE ###################')
    print (df_pixel[df_pixel['Epsilon'] == 0.0].groupby(["Method", "Outliers"])['RMSE'].describe() )
    print ('################# Var ###################')
    print (df_pixel.groupby(["Method", "Outliers"])['Var'].describe() )
    print ('################# Beta  ###################')
    print (df_pixel.groupby(["Method", "Outliers"])['Beta'].describe() )
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=(14.2*cm/2.0,14.2*cm/2.0))
    g = sns.boxplot(hue="Outliers", x="Interval Score", y="Method", data=df_pixel, whis=0.5, showfliers=False)
    #g = sns.catplot(x="Epsilon", y="Interval Score", hue="Method", data=df_pixel, kind="box", whis=0.5, showfliers=False)
    #g.ax.tick_params(labelsize=15)
    plt.legend(fontsize='xx-small')
    plt.savefig(os.path.join(output_dir, f"Interval_score_Comparison_Keypoint.pdf"), bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(14.2*cm, 14.2*cm/2.5))
    gs = fig.add_gridspec(1, 2)
    gs.update(wspace=0.3, hspace=0.5) # set the spacing between axes. 
    ax1 = fig.add_subplot(gs[0, 0])
    g = sns.boxplot(hue="Outliers", 
                    x="RMSE", 
                    y="Method", 
                    data=df_pixel, 
                    whis=0.5, showfliers=False, dodge=True, ax=ax1)
    sns.despine()
    g.get_legend().remove()
    handles, labels = g.get_legend_handles_labels()
    fig.legend(handles, labels, 
              #bbox_to_anchor=(0.55, 0.98), 
               loc='upper center', ncol=3, fancybox=True, shadow=True)

    #plt.savefig(os.path.join(output_dir, f"comparison_rmse.pdf"))

    ax2 = fig.add_subplot(gs[0, 1])
    g = sns.boxplot(hue="Outliers", 
                    x="Interval Score", 
                    y="Method", 
                    data=df_pixel, 
                    whis=0.5, showfliers=False, dodge=True, ax=ax2)
    sns.despine()
    g.set(ylabel=None)
    g.set(yticklabels=[])
    g.get_legend().remove()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"comparison_IS_rmse.pdf"), bbox_inches='tight')
    plt.show()
    #g.set(yscale="log")

    plt.figure(figsize=(14.2*cm/2.0,14.2*cm/2.0))
    g = sns.boxplot(hue="Outliers", y="Entropy", x="Method", data=df_pixel, whis=0.5, showfliers=False)
    #g.set(yscale="log")
    plt.legend(fontsize='xx-small')
    plt.savefig(os.path.join(output_dir, f"Entropy_Comparison_Keypoint.pdf"), bbox_inches='tight')
    plt.show()


if args.load_pkl:
    print("Loading!")
    df_image = pd.read_pickle("cached_keypoint_noisy_results.pkl")
else:
    df_image = compute_predictions()
    df_image.to_pickle("cached_keypoint_noisy_results.pkl")

#df_image["Mu"] = df_image["Mu"] * IMG_SIZE
#df_image["Keypoint"] = df_image["Keypoint"] * IMG_SIZE
gen_interval_score_plot(df_image)
