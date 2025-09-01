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
import models
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from dataset import KeypointDataset
from models import KeypointResnetModel
import tqdm

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
    Laplace = "Laplace"
    Generalized = "GeneralizedGaussian"


save_dir = "trained_models"
output_dir = "figs/keypoint"

trained_models = {
    Model.Gaussian: [
        "gaussian/trial1.pth",
    ],
    Model.Laplace: [
        "laplace/trial1.pth",
    ],
    Model.Generalized: [
        "generalized/trial1.pth",
    ],
}

def load_model(method, check_point_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == Model.Generalized:
        model = KeypointResnetModel(additional_output=True).to(device)
    else:
        model = KeypointResnetModel().to(device)
    model.load_state_dict(torch.load(check_point_path, weights_only=True))
    model.to(device)

    return model 


def compute_predictions(batch_size=32, n_adv=9):
     # Set device
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

    df_pred_image = pd.DataFrame(
            columns=["Method", "Model_path", "Input", 
                "Keypoint", "Mu", "Var", "Beta", "Adv. Mask", "Epsilon", "OOD"])

    adv_eps = np.linspace(0, 0.04, n_adv)
    all_summaries = []
    for method, model_path_list in trained_models.items():
        for model_i, model_path in enumerate(model_path_list):
            full_path = os.path.join(save_dir, model_path)
            model = load_model(method, full_path)
            batch_summary = get_prediction_summary(model, dataloader, device, method, model_path)
            all_summaries.extend(batch_summary)

    df_pred_image = pd.DataFrame(all_summaries)
    print (df_pred_image.head())
    return df_pred_image
       
def get_prediction_summary(model, dataloader, device, method, model_path, eps=0.0, ood=False):
    # Set the model to evaluation mode
    model.eval()
    batch_summaries = []
    for i, (images, keypoints) in enumerate(dataloader):
        images = images.to(device)
        keypoints = keypoints.view(-1, 8)
        # Perform inference
        with torch.no_grad():
            if method == Model.Generalized:
                pred_mean, pred_scale, beta = model(images)
            else:
                pred_mean, pred_scale = model(images)
                beta = torch.ones_like(pred_scale)
        ### Save the predictions to some dataframes for later analysis
        summary = [{"Method": method.value, "Model Path": model_path,
            "Input": x, "Keypoint": y, "Mu": mu, "Var": var,"Beta": beta, 
            "Epsilon": eps, "OOD": ood}
            for x,y,mu,var,beta in zip(images.cpu().numpy(), keypoints.numpy(), pred_mean.cpu().numpy(), pred_scale.cpu().numpy(), beta.cpu().numpy())]
        batch_summaries.extend(summary)
    return batch_summaries

def df_image_to_pixels(df, keys=["Keypoint", "Mu", "Var"]):
    required_keys = ["Method", "Model Path"]
    keys = required_keys + keys
    key_types = {key: type(df[key].iloc[0]) for key in keys}
    max_shape = max([np.prod(np.shape(df[key].iloc[0])) for key in keys])

    contents = {}
    for key in keys:
        if np.prod(np.shape(df[key].iloc[0])) == 1:
            contents[key] = np.repeat(df[key], max_shape)
        else:
            contents[key] = np.stack(df[key], axis=0).flatten()

    df_pixel = pd.DataFrame(contents)
    return df_pixel
    

def gen_calibration_plot(df_image, eps=0.0, ood=False, plot=True):
    print(f"Generating calibration plot with eps={eps}, ood={ood}")
    df_pixel = df_image[(df_image["Epsilon"]==eps) & (df_image["OOD"]==ood)]
    # df = df.iloc[::10]

    df_calibration = list()

    for method, model_path_list in trained_models.items():
        print (model_path_list)
        for model_i, model_path in enumerate(model_path_list):

            df_model = df_pixel[(df_pixel["Method"]==method.value) & (df_pixel["Model Path"]==model_path)]
            df_model = df_model[['Method', 'Keypoint', 'Mu', 'Var', 'Beta']]
            df_model = df_model.explode(['Keypoint', 'Mu', 'Var', 'Beta'])
            df_model = df_model.astype({"Keypoint": float, "Mu": float, "Var": float, "Beta": float})
            print (df_model.head())
            expected_p = np.arange(41)/40.

            observed_p = list()
            for p in expected_p:
                if method == Model.Generalized:
                    ppf = scipy.stats.gennorm.ppf(p, loc=df_model["Mu"], scale=df_model["Var"], beta=df_model["Beta"])
                elif method == Model.Laplace:
                    ppf = scipy.stats.laplace.ppf(p, loc=df_model["Mu"], scale=df_model["Var"])
                elif method == Model.Gaussian:
                    ppf = scipy.stats.norm.ppf(p, loc=df_model["Mu"], scale=np.sqrt(df_model["Var"]))

                obs_p = (df_model["Keypoint"] < ppf).mean()
                observed_p.append(obs_p)

            df_single = {'Method': method.value, 'Model Path': model_path,
                'Expected Conf.': expected_p, 'Observed Conf.': observed_p}
            df_calibration.append(df_single)

    df_truth = {'Method': Model.GroundTruth.value, 'Model Path': "",
        'Expected Conf.': expected_p, 'Observed Conf.': expected_p}
    df_calibration.append(df_truth)
    df_calibration = pd.DataFrame(df_calibration)
    df_calibration = df_calibration.explode(['Expected Conf.', 'Observed Conf.'])

    df_calibration['Calibration Error'] = np.abs(df_calibration['Expected Conf.'] - df_calibration['Observed Conf.'])
    df_calibration["Epsilon"] = eps
    table = df_calibration.groupby(["Method", "Model Path"])["Calibration Error"].mean().reset_index()
    table = pd.pivot_table(table, values="Calibration Error", index="Method", aggfunc=[np.mean, np.std, scipy.stats.sem])

    if plot:
        print(table)
        table.to_csv(os.path.join(output_dir, "calib_errors.csv"))

        print("Plotting confidence plots")
        cm = 1/2.54  # centimeters in inches
        plt.figure(figsize=(14.2*cm/3.0,14.2*cm/3.0))
        sns.lineplot(x="Expected Conf.", y="Observed Conf.", hue="Method", data=df_calibration)
        plt.legend(fontsize='xx-small')
        plt.savefig(os.path.join(output_dir, f"calib_eps-{eps}_ood-{ood}.pdf"), bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(14.2*cm,14.2*cm/2.0))
        g = sns.FacetGrid(df_calibration, col="Method", legend_out=False)
        g = g.map_dataframe(sns.lineplot, x="Expected Conf.", y="Observed Conf.", hue="Model Path")#.add_legend()
        plt.savefig(os.path.join(output_dir, f"calib_eps-{eps}_ood-{ood}_panel.pdf"))
        plt.show()

    return df_calibration, table

def gen_interval_score_plot(df_image):
    print(f"Generating Interval score")
    df_pixel = df_image[(df_image["OOD"]==False) & ((df_image["Epsilon"]==0.0) | (df_image["Epsilon"]==0.02) | (df_image["Epsilon"]==0.04))]

    df_pixel = df_pixel[['Method', 'Keypoint', 'Mu', 'Var', 'Beta']]
    df_pixel = df_pixel.explode(['Keypoint', 'Mu', 'Var', 'Beta'])
    df_pixel = df_pixel.astype({"Keypoint": float, "Mu": float, "Var": float, "Beta": float})

    #print ("Unique Adv : ", df_pixel["Epsilon"].unique())

    print ("Generating RMSE Score")
    df_pixel["RMSE"] = (df_pixel["Mu"] - df_pixel["Keypoint"])**2
    g = sns.catplot(x="Epsilon", y="RMSE", hue="Method", data=df_pixel, kind="box", whis=0.5, showfliers=False)
    g.set(yscale="log")
    plt.savefig(os.path.join(output_dir, f"RMSE_Adv_box_Keypoint_logscale.pdf"))
    plt.show()

    g = sns.catplot(x="Epsilon", y="RMSE", hue="Method", data=df_pixel, kind="box", whis=0.5, showfliers=False)
    plt.savefig(os.path.join(output_dir, f"RMSE_Adv_box_Keypoint.pdf"))
    plt.show()

    print (f"Generating Interval Score")
    df_pixel["lower"] = df_pixel["Mu"] - 2*df_pixel["Sigma"]
    df_pixel["lower"].mask(df_pixel["Method"]=="Laplace", df_pixel["Mu"] - 3*df_pixel["Sigma"], inplace=True)
    df_pixel["upper"] = df_pixel["Mu"] + 2*df_pixel["Sigma"]
    df_pixel["upper"].mask(df_pixel["Method"]=="Laplace", df_pixel["Mu"] + 3*df_pixel["Sigma"], inplace=True)
    
    df_pixel["Interval Score"] = df_pixel["upper"] - df_pixel["lower"] \
     + (2/0.95)*(df_pixel["lower"]-df_pixel["Target"])*(df_pixel["Target"]<df_pixel["lower"]) \
     + (2/0.95)*(df_pixel["Target"] - df_pixel["upper"])*(df_pixel["Target"]>df_pixel["upper"])
    
    g = sns.catplot(x="Epsilon", y="Interval Score", hue="Method", data=df_pixel, kind="box", whis=0.5, showfliers=False)
    g.set(yscale="log")
    plt.savefig(os.path.join(output_dir, f"Interval_score_Adv_box_Keypoint.pdf"))
    plt.show()
 
if args.load_pkl:
    print("Loading!")
    df_image = pd.read_pickle("cached_keypoint_results.pkl")
else:
    df_image = compute_predictions()
    df_image.to_pickle("cached_keypoint_results.pkl")


gen_calibration_plot(df_image)
            
