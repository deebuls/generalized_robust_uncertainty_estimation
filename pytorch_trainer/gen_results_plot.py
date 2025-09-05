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
    Generalized = "Generalized"


save_dir = "trained_models"
output_dir = "figs/keypoint"

trained_models = {
    Model.Gaussian: [
        ["gaussian/gaussian_with_outliers_None_resnet.pth", 0],
        ["gaussian/gaussian_with_outliers_True_resnet.pth", 5],
    ],
    Model.Laplace: [
        ["laplace/laplace_with_outliers_None_resnet.pth", 0],
        ["laplace/laplace_with_outliers_True_resnet.pth", 0],
    ],
    Model.Generalized: [
        ["generalized/generalized_gaussian_with_outliers_None_resnet.pth", 0],
        ["generalized/generalized_gaussian_with_outliers_True_resnet.pth", 5],
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
    all_summaries = []
    for method, model_path_list in trained_models.items():
        for model_i, model_path, outlier_percentage in enumerate(model_path_list):
            full_path = os.path.join(save_dir, model_path)
            model = load_model(method, full_path)
            for epsilon in adv_eps:
                batch_summary = get_prediction_summary(model, dataloader, device, method, model_path, eps=epsilon)
                all_summaries.extend(batch_summary)
            batch_summary = get_prediction_summary(model, ood_dataloader, device, method, model_path, ood=True)
            all_summaries.extend(batch_summary)

    df_pred_image = pd.DataFrame(all_summaries)
    print (df_pred_image.head())
    return df_pred_image
       
def get_prediction_summary(model, dataloader, device, method, model_path, eps=0.0, ood=False):
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
            else:
                pred_mean, pred_scale = model(perturbed_data_normalized)
                beta = torch.ones_like(pred_scale)
 
        pred_mean = pred_mean.detach()
        pred_scale = pred_scale.detach()
        beta = beta.detach()
        if i == 0:
            images = images.detach()
            keypoints = keypoints.detach()
            visualize_batch(images.cpu(), keypoints.cpu(), pred_mean.cpu(), 
                    pred_scale.cpu(), beta.cpu(), method, msg=f'eps{eps}_ood_{ood}')

        ### Save the predictions to some dataframes for later analysis
        summary = [{"Method": method.value, "Model Path": model_path,
            "Keypoint": y, "Mu": mu, "Var": var,"Beta": beta, 
            "Epsilon": eps, "OOD": ood}
            for y,mu,var,beta in zip(keypoints.cpu().numpy(), pred_mean.cpu().numpy(), pred_scale.cpu().numpy(), beta.cpu().numpy())]
        batch_summaries.extend(summary)
    return batch_summaries

def visualize_batch(images, keypoints, pred_mean, pred_scale, pred_beta, method, msg=''):
    pred_mean = pred_mean.view(-1, 4, 2)
    pred_scale = pred_scale.view(-1, 4, 2)
    keypoints = keypoints.view(-1, 4, 2)
    if method == Model.Generalized:
        pred_beta = pred_beta.view(-1, 4, 2)
        # The batch size and the middle dimension remain the same.
        pred_scale = torch.cat((pred_scale, pred_beta), dim=2)
        loss_function = "generalized_gaussian"
    elif method == Model.Gaussian:
        loss_function = "gaussian"
    elif method == Model.Laplace:
        loss_function = "laplace"
        
    visualizer = KeypointVisualizer(distribution=loss_function, img_size=IMG_SIZE)
    visualizer.visualize_batch(
        images.cpu(), keypoints, pred_mean.cpu(), 
        pred_scale.cpu(), show_contours=False, figname=msg
    )

def gen_calibration_plot(df_image, eps=0.0, ood=False, plot=True):
    print(f"Generating calibration plot with eps={eps}, ood={ood}")
    df_pixel = df_image[(df_image["Epsilon"]==eps) & (df_image["OOD"]==ood)]
    # df = df.iloc[::10]

    df_calibration = list()

    for method, model_path_list in trained_models.items():
        for model_i, model_path in enumerate(model_path_list):

            df_model = df_pixel[(df_pixel["Method"]==method.value) & (df_pixel["Model Path"]==model_path)]
            df_model = df_model[['Method', 'Keypoint', 'Mu', 'Var', 'Beta']]
            df_model = df_model.explode(['Keypoint', 'Mu', 'Var', 'Beta'])
            df_model = df_model.astype({"Keypoint": float, "Mu": float, "Var": float, "Beta": float})
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

    df_pixel = df_pixel[['Method', 'Epsilon', 'Keypoint', 'Mu', 'Var', 'Beta']]
    df_pixel = df_pixel.explode(['Keypoint', 'Mu', 'Var', 'Beta'])
    df_pixel = df_pixel.astype({"Keypoint": float, "Mu": float, "Var": float, "Beta": float})

    print ("Unique Adv : ", df_pixel["Epsilon"].unique())

    print ("Generating RMSE Score")
    df_pixel["RMSE"] = (df_pixel["Mu"] - df_pixel["Keypoint"])**2
    g = sns.catplot(x="Epsilon", y="RMSE", hue="Method", data=df_pixel, kind="box", whis=0.5, showfliers=False)
    #g.set(yscale="log")
    #plt.savefig(os.path.join(output_dir, f"RMSE_Adv_box_Keypoint_logscale.pdf"))
    #plt.show()

    g = sns.catplot(x="Epsilon", y="RMSE", hue="Method", data=df_pixel, kind="box", whis=0.5, showfliers=False)
    plt.savefig(os.path.join(output_dir, f"RMSE_Adv_box_Keypoint.pdf"))
    plt.show()

    print (f"Generating Interval Score")

    # Calculate the 95% interval regions
    # This corresponds to the 2.5th and 97.5th percentiles of the distributions.
    lower_percentile = 0.025
    upper_percentile = 0.975

    df_pixel["lower"] = df_pixel['Beta']
    df_pixel["lower"].mask(df_pixel["Method"]=="Gaussian", 
            norm.ppf(lower_percentile , loc=df_pixel['Mu'], scale=np.sqrt(df_pixel['Var'])), inplace=True )
    df_pixel["lower"].mask(df_pixel["Method"]=="Laplace", 
            laplace.ppf(lower_percentile , loc=df_pixel['Mu'], scale=df_pixel['Var']), inplace=True)
    df_pixel["lower"].mask(df_pixel["Method"]=="Generalized", 
            gennorm.ppf(lower_percentile , loc=df_pixel['Mu'], scale=df_pixel['Var'], beta=df_pixel['Beta']), inplace=True)
    df_pixel["upper"] = df_pixel['Beta']
    df_pixel["upper"].mask(df_pixel["Method"]=="Gaussian", 
            norm.ppf(upper_percentile , loc=df_pixel['Mu'], scale=np.sqrt(df_pixel['Var'])), inplace=True)
    df_pixel["upper"].mask(df_pixel["Method"]=="Laplace", 
            laplace.ppf(upper_percentile , loc=df_pixel['Mu'], scale=df_pixel['Var']), inplace=True)
    df_pixel["upper"].mask(df_pixel["Method"]=="Generalized", 
            gennorm.ppf(upper_percentile , loc=df_pixel['Mu'], scale=df_pixel['Var'], beta=df_pixel['Beta']), inplace=True)
    
    df_pixel["Interval Score"] = df_pixel["upper"] - df_pixel["lower"] \
     + (2/0.95)*(df_pixel["lower"]-df_pixel["Keypoint"])*(df_pixel["Keypoint"]<df_pixel["lower"]) \
     + (2/0.95)*(df_pixel["Keypoint"] - df_pixel["upper"])*(df_pixel["Keypoint"]>df_pixel["upper"])
    
    g = sns.catplot(x="Epsilon", y="Interval Score", hue="Method", data=df_pixel, kind="box", whis=0.5, showfliers=False)
    #g.set(yscale="log")
    plt.savefig(os.path.join(output_dir, f"Interval_score_Adv_box_Keypoint.pdf"))
    plt.show()
 
def gen_ood_comparison(df_image, unc_key="Entropy"):
    print(f"Generating OOD plots with unc_key={unc_key}")

    df_pixel = df_image[df_image["Epsilon"]==0.0] # Remove adversarial noise experiments
    df_pixel = df_pixel[['Method', 'Model Path', 'OOD', 'Keypoint', 'Mu', 'Var', 'Beta']]
    df_pixel = df_pixel.explode(['Keypoint', 'Mu', 'Var', 'Beta'])
    df_pixel = df_pixel.astype({"Keypoint": float, "Mu": float, "Var": float, "Beta": float})

    print ("sigma inf count :",np.sum(np.isinf(df_pixel['Var'])))
    #inf_id = df_pixel[df_pixel.isin([np.nan, np.inf, -np.inf]).any(1)]
    #print (inf_id.head())
    df_pixel["Entropy"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_pixel["Var"]))
    print ("Entropy inf count :",np.sum(np.isinf(df_pixel['Entropy'])))
    df_pixel["Entropy"].mask(df_pixel["Method"]=="Gaussian", norm.entropy(loc=df_pixel["Mu"], scale=np.sqrt(df_pixel["Var"])) ) #  entropy for laplace distirbution
    df_pixel["Entropy"].mask(df_pixel["Method"]=="Laplace",  laplace.entropy(loc=df_pixel["Mu"], scale=df_pixel["Var"]) ) #  entropy for laplace distirbution
    df_pixel["Entropy"].mask(df_pixel["Method"]=="Generalized",  gennorm.entropy(loc=df_pixel["Mu"], scale=df_pixel["Var"], beta=df_pixel["Beta"]) ) #  entropy for laplace distirbution

    df_by_method = df_pixel.groupby(["Method","Model Path", "OOD"])
    df_by_image = df_pixel.groupby([df_pixel.index, "Method","Model Path", "OOD"])

    df_mean_unc = df_by_method[unc_key].mean().reset_index() #mean of all pixels per method
    df_mean_unc_img = df_by_image[unc_key].mean().reset_index() #mean of all pixels in every method and image


    ### Grab some sample images of most and least uncertainty
    #for method in df_mean_unc_img["Method"].unique():
    #    imgs_max = dict()
    #    imgs_min = dict()
    #    for ood in df_mean_unc_img["OOD"].unique():
    #        df_subset = df_mean_unc_img[
    #            (df_mean_unc_img["Method"]==method) &
    #            (df_mean_unc_img["OOD"]==ood)]
    #        if len(df_subset) == 0:
    #            continue
    #        def get_imgs_from_idx(idx):
    #            i_img = df_subset.loc[idx]["level_0"]
    #            img_data = df_image.loc[i_img]
    #            sigma = np.array(img_data["Sigma"])
    #            entropy = np.log(sigma**2)

    #            ret = [img_data["Input"], img_data["Mu"], entropy, img_data["Target"]]
    #            return list(map(trim, ret))

    #        def idxquantile(s, q=0.5, *args, **kwargs):
    #            qv = s.quantile(q, *args, **kwargs)
    #            return (s.sort_values()[::-1] <= qv).idxmax()

    #        imgs_max[ood] = get_imgs_from_idx(idx=idxquantile(df_subset["Entropy"], 0.95))
    #        imgs_min[ood] = get_imgs_from_idx(idx=idxquantile(df_subset["Entropy"], 0.05))

    #    all_entropy_imgs = np.array([ [d[ood][2] for ood in d.keys()] for d in (imgs_max, imgs_min)])
    #    entropy_bounds = (all_entropy_imgs.min(), all_entropy_imgs.max())

    #    Path(os.path.join(output_dir, "images")).mkdir(parents=True, exist_ok=True)
    #    for d in (imgs_max, imgs_min):
    #        for ood, (x, y, entropy, target) in d.items():
    #            id = os.path.join(output_dir, f"images/ood_{ood}_method_{method}_entropy_{entropy.mean()}")
    #            cv2.imwrite(f"{id}_0.png", 255*x)
    #            cv2.imwrite(f"{id}_mu.png", apply_cmap(y, cmap=cv2.COLORMAP_JET))
    #            cv2.imwrite(f"{id}_target.png", apply_cmap(target, cmap=cv2.COLORMAP_JET))
    #            entropy = (entropy - entropy_bounds[0]) / (entropy_bounds[1]-entropy_bounds[0])
    #            cv2.imwrite(f"{id}_unc.png", apply_cmap(entropy))

    #cm = 1/2.54  # centimeters in inches
    #sns.catplot(x="Method", y=unc_key, hue="OOD", data=df_mean_unc_img, kind="violin")
    #plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_violin.pdf"))
    #plt.show()

    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(14.2*cm/2.0,14.2*cm/2.0))
    #sns.catplot(x="Method", y=unc_key, hue="OOD", data=df_mean_unc_img, kind="box", whis=0.5, showfliers=False)
    g = sns.boxplot(x="Method", y=unc_key, hue="OOD", data=df_mean_unc_img, whis=0.5, showfliers=False)
    g.get_legend().remove()
    handles, labels = g.get_legend_handles_labels()
    print ("OOD Labels ", labels)
    plt.legend(handles, ['ID','OOD'], bbox_to_anchor=(0.01, 0.99), loc='upper left', ncol=1)
    plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_box.pdf"), bbox_inches='tight')
    plt.show()


    ### Plot PDF for each Method on both OOD and IN
    cm = 1/2.54  # centimeters in inches
    g = sns.FacetGrid(df_mean_unc_img, col="Method", hue="OOD", height=14.2*cm/2.0, aspect=0.8, legend_out=False)
    g.map(sns.kdeplot, "Entropy")#.add_legend()
    g.axes[0][2].legend()
    plt.legend(['ID','OOD'], fontsize='small')
    plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_pdf_per_method.pdf"), bbox_inches='tight')
    plt.show()

    exit()
    ## FIX BEELOW
    ### Plot CDFs for every method on both OOD and IN
    df_cumdf = list()
    unc_ = np.linspace(df_mean_unc_img[unc_key].min(), df_mean_unc_img[unc_key].max(), 200)

    for method in df_mean_unc_img["Method"].unique():
        for model_path in df_mean_unc_img["Model Path"].unique():
            for ood in df_mean_unc_img["OOD"].unique():
                df = df_mean_unc_img[
                    (df_mean_unc_img["Method"]==method) &
                    (df_mean_unc_img["Model Path"]==model_path) &
                    (df_mean_unc_img["OOD"]==ood)]
                if len(df) == 0:
                    continue
                unc = np.sort(df[unc_key])
                prob = np.linspace(0,1,unc.shape[0])
                f_cdf = scipy.interpolate.interp1d(unc, prob, fill_value=(0.,1.), bounds_error=False)
                prob_ = f_cdf(unc_)

                df_single = {'Method': method, 'Model Path': model_path,
                    'OOD': ood, unc_key: unc_, 'CDF': prob_}
                df_cumdf.append(df_single)

    df_cumdf = pd.DataFrame(df_cumdf)
    df_cumdf = df_cumdf.explode(['CDF'])
    print (df_cumdf)
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=(14.2*cm/2.0,14.2*cm/2.0))
    g = sns.lineplot(data=df_cumdf, x=unc_key, y="CDF", hue="Method", style="OOD")
    handles, labels = g.get_legend_handles_labels()
    print ("OOD Labels ", labels)
    labels[-2] = 'ID';labels[-1] = 'OOD';
    print ("OOD Labels ", labels)
    plt.legend( handles, labels, fontsize='x-small')
    plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_cdfs.pdf"), bbox_inches='tight')
    plt.show()

if args.load_pkl:
    print("Loading!")
    df_image = pd.read_pickle("cached_keypoint_results.pkl")
else:
    df_image = compute_predictions()
    df_image.to_pickle("cached_keypoint_results.pkl")

quit()
gen_calibration_plot(df_image)
df_image["Mu"] = df_image["Mu"] * IMG_SIZE
df_image["Keypoint"] = df_image["Keypoint"] * IMG_SIZE
gen_interval_score_plot(df_image)
gen_ood_comparison(df_image)
            
