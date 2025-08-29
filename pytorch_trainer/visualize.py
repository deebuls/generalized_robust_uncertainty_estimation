import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from abc import ABC, abstractmethod
from scipy.stats import gennorm, norm

class BaseDistribution(ABC):
    """Base class for probability distributions used in keypoint visualization."""
    
    @abstractmethod
    def create_uncertainty_patch(self, center_x, center_y, params, confidence_level=2.0):
        """Create a matplotlib patch representing the uncertainty region."""
        pass
    
    @abstractmethod
    def create_contour_patches(self, center_x, center_y, params, levels=[1, 1.5, 2]):
        """Create multiple contour patches at different confidence levels."""
        pass
    
    @property
    @abstractmethod
    def param_names(self):
        """Return the parameter names for this distribution."""
        pass


class GaussianDistribution(BaseDistribution):
    """Gaussian distribution for keypoint uncertainty visualization."""
    
    @property
    def param_names(self):
        return "variance"
    
    def create_uncertainty_patch(self, center_x, center_y, params, confidence_level=3.0):
        """Create ellipse for Gaussian uncertainty."""
        if params.shape[0] == 2:  # Independent variances (var_x, var_y)
            var_x, var_y = params[0], params[1]
            width = 2 * confidence_level * np.sqrt(var_x)
            height = 2 * confidence_level * np.sqrt(var_y)
            angle = 0
        elif params.shape == (2, 2):  # Full covariance matrix
            cov = params
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width = 2 * confidence_level * np.sqrt(eigenvals[0])
            height = 2 * confidence_level * np.sqrt(eigenvals[1])
        else:
            raise ValueError(f"Invalid parameter shape for Gaussian: {params.shape}")
        
        return Ellipse(
            (center_x, center_y),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor='green',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7
        )
    
    def create_contour_patches(self, center_x, center_y, params, levels=[1, 1.5, 2]):
        """Create multiple elliptical contours."""
        patches = []
        colors = ['lightgreen', 'green', 'darkgreen']
        alphas = [0.3, 0.5, 0.7]
        linewidths = [0.8, 1.0, 1.2]
        
        for level, color, alpha, lw in zip(levels, colors, alphas, linewidths):
            if params.shape[0] == 2:
                var_x, var_y = params[0], params[1]
                width = 2 * level * np.sqrt(var_x)
                height = 2 * level * np.sqrt(var_y)
                angle = 0
            elif params.shape == (2, 2):
                cov = params
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width = 2 * level * np.sqrt(eigenvals[0])
                height = 2 * level * np.sqrt(eigenvals[1])
            
            patch = Ellipse(
                (center_x, center_y),
                width=width,
                height=height,
                angle=angle,
                fill=False,
                edgecolor=color,
                linestyle='--',
                linewidth=lw,
                alpha=alpha
            )
            patches.append(patch)
        
        return patches


class LaplaceDistribution(BaseDistribution):
    """Laplace distribution for keypoint uncertainty visualization."""
    
    @property
    def param_names(self):
        return "scale"
    
    def create_uncertainty_patch(self, center_x, center_y, params, confidence_level=2.0):
        """Create diamond for Laplace uncertainty."""
        if params.shape[0] != 2:
            raise ValueError(f"Laplace distribution expects 2D scale parameters, got shape: {params.shape}")
        
        scale_x, scale_y = params[0], params[1]
        half_width = confidence_level * scale_x
        half_height = confidence_level * scale_y
        
        diamond_vertices = np.array([
            [center_x, center_y + half_height],  # top
            [center_x + half_width, center_y],   # right
            [center_x, center_y - half_height],  # bottom
            [center_x - half_width, center_y]    # left
        ])
        
        return Polygon(
            diamond_vertices,
            fill=False,
            edgecolor='green',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7
        )
    
    def create_contour_patches(self, center_x, center_y, params, levels=[1, 1.5, 2]):
        """Create multiple diamond contours."""
        patches = []
        colors = ['lightgreen', 'green', 'darkgreen']
        alphas = [0.3, 0.5, 0.7]
        linewidths = [0.8, 1.0, 1.2]
        
        scale_x, scale_y = params[0], params[1]
        
        for level, color, alpha, lw in zip(levels, colors, alphas, linewidths):
            half_width = level * scale_x
            half_height = level * scale_y
            
            diamond_vertices = np.array([
                [center_x, center_y + half_height],
                [center_x + half_width, center_y],
                [center_x, center_y - half_height],
                [center_x - half_width, center_y]
            ])
            
            patch = Polygon(
                diamond_vertices,
                fill=False,
                edgecolor=color,
                linestyle='--',
                linewidth=lw,
                alpha=alpha
            )
            patches.append(patch)
        
        return patches


class StudentTDistribution(BaseDistribution):
    """Student's t-distribution for keypoint uncertainty visualization."""
    
    @property
    def param_names(self):
        return "scale_and_dof"
    
    def create_uncertainty_patch(self, center_x, center_y, params, confidence_level=2.0):
        """Create ellipse for Student's t uncertainty (similar to Gaussian but wider)."""
        if params.shape[0] != 3:  # scale_x, scale_y, degrees_of_freedom
            raise ValueError(f"Student's t distribution expects [scale_x, scale_y, dof], got shape: {params.shape}")
        
        scale_x, scale_y, dof = params[0], params[1], params[2]
        
        # t-distribution is wider than Gaussian, adjust by sqrt((dof)/(dof-2))
        if dof > 2:
            t_factor = np.sqrt(dof / (dof - 2))
        else:
            t_factor = 2.0  # fallback for low dof
        
        width = 2 * confidence_level * scale_x * t_factor
        height = 2 * confidence_level * scale_y * t_factor
        
        return Ellipse(
            (center_x, center_y),
            width=width,
            height=height,
            angle=0,
            fill=False,
            edgecolor='purple',
            linestyle='-.',
            linewidth=1.5,
            alpha=0.7
        )
    
    def create_contour_patches(self, center_x, center_y, params, levels=[1, 1.5, 2]):
        """Create multiple elliptical contours for t-distribution."""
        patches = []
        colors = ['plum', 'purple', 'indigo']
        alphas = [0.3, 0.5, 0.7]
        linewidths = [0.8, 1.0, 1.2]
        
        scale_x, scale_y, dof = params[0], params[1], params[2]
        
        if dof > 2:
            t_factor = np.sqrt(dof / (dof - 2))
        else:
            t_factor = 2.0
        
        for level, color, alpha, lw in zip(levels, colors, alphas, linewidths):
            width = 2 * level * scale_x * t_factor
            height = 2 * level * scale_y * t_factor
            
            patch = Ellipse(
                (center_x, center_y),
                width=width,
                height=height,
                angle=0,
                fill=False,
                edgecolor=color,
                linestyle='-.',
                linewidth=lw,
                alpha=alpha
            )
            patches.append(patch)
        
        return patches

class GeneralizedGaussianDistribution(BaseDistribution):
    """Generalized Gaussian distribution for keypoint uncertainty visualization."""
    
    @property
    def param_names(self):
        return "scale_and_shape"

    def calculate_sigmax_sigmay(self, params):
        alpha_x, alpha_y, beta_x, beta_y = params[0], params[1], params[2], params[3]
        # The 90% confidence region is centered, so we need the 95th percentile
        # to find the upper bound (1 - 0.10/2).
        percentile = 0.95

        # Find the PPF (Percent Point Function or inverse CDF) for a standard
        # Generalized Gaussian distribution with the given beta at the 95th percentile.
        # We use a standard GGD (loc=0, scale=1) to get the scaling factor.
        ppf_gennorm_standard_x = gennorm.ppf(percentile, beta_x, loc=0, scale=1)
        ppf_gennorm_standard_y = gennorm.ppf(percentile, beta_y, loc=0, scale=1)

        # Find the PPF for a standard Normal distribution at the 95th percentile.
        # This is the z-score for a 90% confidence interval.
        ppf_norm_standard = norm.ppf(percentile, loc=0, scale=1)

        # The scaling factor is the ratio of these two values.
        scaling_factor_x = ppf_gennorm_standard_x / ppf_norm_standard
        scaling_factor_y = ppf_gennorm_standard_y / ppf_norm_standard

        # The sigma of the equivalent Gaussian is the alpha of the GGD multiplied
        # by this scaling factor.
        sigma_x = alpha_x * scaling_factor_x
        sigma_y = alpha_y * scaling_factor_y
        return sigma_x, sigma_y


    def create_uncertainty_patch(self, center_x, center_y, params, confidence_level=1.0):
        """Create ellipse for generalized Gaussian uncertainty."""
        if params.shape[0] != 4:  # alpha_x, beta_x, alpha_y, beta_y
            raise ValueError(f"Generalized Gaussian distribution expects [alpha_x, beta_x, alpha_y, beta_y], got shape: {params.shape}")
        sigma_x, sigma_y = self.calculate_sigmax_sigmay( params) 
        
        width = 2.0 * confidence_level * sigma_x
        height = 2.0 *  confidence_level * sigma_y
        angle = 0
        
        return Ellipse(
            (center_x, center_y),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor='green',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7
        )
    
    def create_contour_patches(self, center_x, center_y, params, levels=[1, 1.5, 2]):
        """Create multiple elliptical contours."""
        patches = []
        colors = ['lightgreen', 'green', 'darkgreen']
        alphas = [0.3, 0.5, 0.7]
        linewidths = [0.8, 1.0, 1.2]
        
        for level, color, alpha, lw in zip(levels, colors, alphas, linewidths):
            sigma_x, sigma_y = self.calculate_sigmax_sigmay(params) 
            
            width = 2.0 * level * sigma_x
            height = 2.0 *  level * sigma_y
            angle = 0

            patch = Ellipse(
                (center_x, center_y),
                width=width,
                height=height,
                angle=angle,
                fill=False,
                edgecolor=color,
                linestyle='--',
                linewidth=lw,
                alpha=alpha
            )
            patches.append(patch)
        
        return patches


class KeypointVisualizer:
    """
    Flexible keypoint visualization class supporting multiple probability distributions.
    """
    
    def __init__(self, distribution='gaussian', img_size=224):
        """
        Initialize the visualizer.
        
        Args:
            distribution (str or BaseDistribution): Type of distribution ('gaussian', 'laplace') 
                                                   or custom distribution object
            img_size (int): Image size for coordinate scaling
        """
        self.img_size = img_size
        self.available_distributions = {
            'gaussian': GaussianDistribution(),
            'laplace': LaplaceDistribution(),
            'student_t': StudentTDistribution(),
            'generalized_gaussian': GeneralizedGaussianDistribution(),
        }
        
        if isinstance(distribution, str):
            if distribution not in self.available_distributions:
                raise ValueError(f"Unknown distribution: {distribution}. "
                               f"Available: {list(self.available_distributions.keys())}")
            self.distribution = self.available_distributions[distribution]
        elif isinstance(distribution, BaseDistribution):
            self.distribution = distribution
        else:
            raise ValueError("Distribution must be a string or BaseDistribution instance")
    
    def add_distribution(self, name, distribution_obj):
        """
        Add a new distribution type.
        
        Args:
            name (str): Name for the new distribution
            distribution_obj (BaseDistribution): Distribution implementation
        """
        if not isinstance(distribution_obj, BaseDistribution):
            raise ValueError("Distribution must inherit from BaseDistribution")
        
        self.available_distributions[name] = distribution_obj
    
    def set_distribution(self, distribution):
        """
        Change the current distribution.
        
        Args:
            distribution (str or BaseDistribution): New distribution to use
        """
        if isinstance(distribution, str):
            if distribution not in self.available_distributions:
                raise ValueError(f"Unknown distribution: {distribution}")
            self.distribution = self.available_distributions[distribution]
        elif isinstance(distribution, BaseDistribution):
            self.distribution = distribution
        else:
            raise ValueError("Distribution must be a string or BaseDistribution instance")
    
    def visualize_batch(self, image_batch, keypoints_batch, 
                       pred_keypoints_batch=None, uncertainty_params_batch=None,
                       batch_indices=None, show_contours=True, figsize=(15, 10), figname=''):
        """
        Visualize a batch of images with keypoints and uncertainty.
        
        Args:
            image_batch (torch.Tensor): Shape (B, C, H, W)
            keypoints_batch (torch.Tensor): Shape (B, N, 2)
            pred_keypoints_batch (torch.Tensor): Shape (B, N, 2), optional
            uncertainty_params_batch (torch.Tensor): Parameters for uncertainty visualization
            batch_indices (list): Specific indices to visualize
            show_contours (bool): Whether to show multiple contour levels
            figsize (tuple): Figure size
        """
        batch_size = image_batch.shape[0]
        
        if batch_indices is None:
            batch_indices = list(range(batch_size))
        elif isinstance(batch_indices, int):
            batch_indices = [batch_indices]
        
        num_samples = len(batch_indices)
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_samples == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, batch_idx in enumerate(batch_indices):
            ax = axes[idx] if num_samples > 1 else axes[0]
            
            image_tensor = image_batch[batch_idx]
            keypoints_tensor = keypoints_batch[batch_idx]
            
            pred_keypoints_tensor = None
            uncertainty_params_tensor = None
            
            if pred_keypoints_batch is not None:
                pred_keypoints_tensor = pred_keypoints_batch[batch_idx]
            if uncertainty_params_batch is not None:
                uncertainty_params_tensor = uncertainty_params_batch[batch_idx]
            
            self._visualize_single_sample(ax, image_tensor, keypoints_tensor,
                                        pred_keypoints_tensor, uncertainty_params_tensor,
                                        show_contours)
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'({type(self.distribution).__name__})', fontsize=10)
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'{type(self.distribution).__name__}_{figname}.png')

    
    def visualize_single(self, image_tensor, keypoints_tensor, 
                        pred_keypoints_tensor=None, uncertainty_params_tensor=None,
                        show_contours=False, figsize=(8, 6)):
        """
        Visualize a single image with keypoints.
        """
        # Add batch dimension if needed
        if len(image_tensor.shape) == 3:
            image_batch = image_tensor.unsqueeze(0)
            keypoints_batch = keypoints_tensor.unsqueeze(0)
            
            pred_keypoints_batch = None
            uncertainty_params_batch = None
            
            if pred_keypoints_tensor is not None:
                pred_keypoints_batch = pred_keypoints_tensor.unsqueeze(0)
            if uncertainty_params_tensor is not None:
                uncertainty_params_batch = uncertainty_params_tensor.unsqueeze(0)
            
            self.visualize_batch(image_batch, keypoints_batch,
                               pred_keypoints_batch, uncertainty_params_batch,
                               batch_indices=[0], show_contours=show_contours,
                               figsize=figsize)
        else:
            self.visualize_batch(image_tensor, keypoints_tensor,
                               pred_keypoints_tensor, uncertainty_params_tensor,
                               show_contours=show_contours, figsize=figsize)
    
    def _visualize_single_sample(self, ax, image_tensor, keypoints_tensor,
                               pred_keypoints_tensor=None, uncertainty_params_tensor=None,
                               show_contours=True):
        """Helper function to visualize a single sample."""
        # Convert and normalize image
        image = image_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Convert keypoints
        keypoints = keypoints_tensor.numpy() * self.img_size
        
        ax.imshow(image)
        
        # Plot predicted keypoints
        for x, y in keypoints:
            ax.plot(x, y, 'r.', markersize=8)
        
        if pred_keypoints_tensor is not None:
            true_keypoints = pred_keypoints_tensor.numpy() * self.img_size
            
            # Plot true keypoints
            for x, y in true_keypoints:
                ax.plot(x, y, 'g.', markersize=8)
        
        if pred_keypoints_tensor is not None and uncertainty_params_tensor is not None:
            true_keypoints = pred_keypoints_tensor.numpy() * self.img_size
            uncertainty_params = uncertainty_params_tensor.numpy() * self.img_size
            
            for i, (x, y) in enumerate(true_keypoints):
                params = uncertainty_params[i]
                
                if show_contours:
                    patches = self.distribution.create_contour_patches(x, y, params)
                    for patch in patches:
                        ax.add_patch(patch)
                else:
                    patch = self.distribution.create_uncertainty_patch(x, y, params)
                    ax.add_patch(patch)
    
    def get_available_distributions(self):
        """Return list of available distribution names."""
        return list(self.available_distributions.keys())
    
    def get_current_distribution(self):
        """Return current distribution name and object."""
        for name, dist_obj in self.available_distributions.items():
            if dist_obj is self.distribution:
                return name, dist_obj
        return "custom", self.distribution


# Example usage and demonstration
def demo_visualizer():
    """Demonstrate the KeypointVisualizer with different distributions."""
    
    # Create synthetic data
    batch_size = 4
    num_keypoints = 3
    img_size = 224
    
    image_batch = torch.ones(batch_size, 3, img_size, img_size)
    keypoints_batch = torch.rand(batch_size, num_keypoints, 2)
    pred_keypoints_batch = keypoints_batch + 0.02 * torch.randn_like(keypoints_batch)
    
    # Test different distributions
    distributions_to_test = ['gaussian', 'laplace', 'student_t', 'generalized_gaussian']
    
    for dist_name in distributions_to_test:
        print(f"\nTesting {dist_name} distribution:")
        
        # Create appropriate uncertainty parameters
        if dist_name == 'gaussian':
            # Variance parameters (var_x, var_y)
            uncertainty_params = 0.001 + 0.0005 * torch.rand(batch_size, num_keypoints, 2)
        elif dist_name == 'laplace':
            # Scale parameters (scale_x, scale_y)
            uncertainty_params = 0.1 + 0.005 * torch.rand(batch_size, num_keypoints, 2)
        elif dist_name == 'student_t':
            # Scale and degrees of freedom (scale_x, scale_y, dof)
            uncertainty_params = torch.zeros(batch_size, num_keypoints, 3)
            uncertainty_params[..., :2] = 0.01 + 0.005 * torch.rand(batch_size, num_keypoints, 2)
            uncertainty_params[..., 2] = 3 + 5 * torch.rand(batch_size, num_keypoints)  # dof between 3-8
        elif dist_name == 'generalized_gaussian':
            # Scale and degrees of freedom (scale_x, scale_y, dof)
            uncertainty_params = torch.ones(batch_size, num_keypoints, 4)
            uncertainty_params[..., :] = 1 + 1 * torch.rand(batch_size, num_keypoints, 4)  # dof between 1-2
        # Create visualizer and display
        visualizer = KeypointVisualizer(distribution=dist_name, img_size=img_size)
        visualizer.visualize_batch(
            image_batch, keypoints_batch, pred_keypoints_batch, 
            uncertainty_params, batch_indices=[0, 1], show_contours=True
        )
        
        print(f"Available distributions: {visualizer.get_available_distributions()}")
        current_name, _ = visualizer.get_current_distribution()
        print(f"Current distribution: {current_name}")


if __name__ == "__main__":
    demo_visualizer()
