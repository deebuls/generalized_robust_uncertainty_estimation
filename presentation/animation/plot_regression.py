import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Distribution Loss Functions
# ==========================================

class GeneralizedGaussian(dist.Distribution):
    """Custom Generalized Gaussian (Sub-Gaussian/Super-Gaussian) Distribution."""
    arg_constraints = {'loc': dist.constraints.real, 'scale': dist.constraints.positive}
    
    def __init__(self, loc, scale, beta=1.5, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.beta = beta
        super().__init__(batch_shape=loc.shape, validate_args=validate_args)

    def log_prob(self, value):
        # log p(y) = log(beta / (2 * scale * gamma(1/beta))) - (|y - loc| / scale)^beta
        gamma_const = torch.lgamma(torch.tensor(1.0 / self.beta))
        norm_const = torch.log(torch.tensor(self.beta / 2.0)) - torch.log(self.scale) - gamma_const
        return norm_const - (torch.abs(value - self.loc) / self.scale) ** self.beta

def get_distribution(dist_type, mu, scale):
    """Instantiates the requested PyTorch distribution."""
    dist_type = dist_type.lower()
    if dist_type in ["gaussian", "normal"]:
        return dist.Normal(loc=mu, scale=scale)
        dist_type = "gaussian"
    elif dist_type == "laplace":
        return dist.Laplace(loc=mu, scale=scale)
    elif dist_type == "cauchy":
        return dist.Cauchy(loc=mu, scale=scale)
    elif dist_type == "student_t":
        # Fixed degrees of freedom df=2.0 for heavy tails
        return dist.StudentT(df=2.0, loc=mu, scale=scale)
    elif dist_type in ["generalized_gaussian", "gen_gaussian"]:
        return GeneralizedGaussian(loc=mu, scale=scale, beta=1.5)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

def get_95_ci_factor(dist_type):
    """Returns the quantile scale multiplier for a 95% two-sided confidence region (~2.5% to 97.5%)."""
    dist_type = dist_type.lower()
    if dist_type in ["gaussian", "normal"]:
        return 1.96
    elif dist_type == "laplace":
        return 2.99  # -ln(2 * 0.025)
    elif dist_type == "student_t":
        return 4.30  # Student-t with df=2
    elif dist_type == "cauchy":
        return 12.71 # Cauchy has extreme tails (95% quantile is tan(0.475 * pi))
    else:
        return 2.0

# ==========================================
# 2. MLP Model Definition
# ==========================================

class RegressionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # Output 0: mu, Output 1: log_scale
        )

    def forward(self, x):
        out = self.net(x)
        mu = out[:, 0:1]
        # Softplus ensures positive scale with smooth gradients
        scale = nn.functional.softplus(out[:, 1:2]) + 1e-4
        return mu, scale

# ==========================================
# 3. Data Generation
# ==========================================

def generate_data(with_outliers=True):
    n_samples = 20
    x = np.linspace(-1.0, 1.0, n_samples)
    m, c = 0.8, 0.0
    
    # Base linear relationship with small Gaussian noise
    y = m * x + c + np.random.normal(0, 0.08, size=n_samples)
    
    is_outlier = np.zeros(n_samples, dtype=bool)
    
    if with_outliers:
        # Add 3 outliers: 2 above the line, 1 below the line
        outlier_indices = [4, 11, 16]
        y[outlier_indices[0]] += 0.8  # Top outlier 1
        y[outlier_indices[1]] += 0.9  # Top outlier 2
        y[outlier_indices[2]] -= 0.85 # Bottom outlier 1
        is_outlier[outlier_indices] = True

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    return x_tensor, y_tensor, is_outlier

# ==========================================
# 4. Main Training and Animation
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Plot MLP regression with uncertainty.")
    parser.add_argument("distribution", type=str, choices=["gaussian", "laplace", "cauchy", "student_t", "generalized_gaussian"],
                        help="Loss distribution type.")
    parser.add_argument("outliers", type=str, choices=["with_outliers", "without_outliers", "no_outliers"],
                        help="Whether to generate outliers.")
    args = parser.parse_args()

    has_outliers = (args.outliers == "with_outliers")
    dist_name = args.distribution

    # Generate Dataset
    x_train, y_train, is_outlier = generate_data(with_outliers=has_outliers)
    
    # Test grid for plotting fine curves
    x_grid = torch.linspace(-1.2, 1.2, 200).unsqueeze(1)

    # Initialize Model & Optimizer
    model = RegressionMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Logging structures for animation
    epochs = 1200
    save_interval = 15
    history = []

    print(f"Training MLP with {dist_name.upper()} loss | Outliers: {has_outliers}...")

    # Training Loop
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        mu, scale = model(x_train)
        distribution = get_distribution(dist_name, mu, scale)
        
        # Pointwise NLL Loss
        pointwise_nll = -distribution.log_prob(y_train)
        loss = pointwise_nll.mean()

        loss.backward()
        optimizer.step()

        # Record history periodically
        if epoch % save_interval == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                # Grid predictions for uncertainty bounds
                grid_mu, grid_scale = model(x_grid)
                
                # Training set predictions
                train_mu, train_scale = model(x_train)
                train_dist = get_distribution(dist_name, train_mu, train_scale)
                train_nll = -train_dist.log_prob(y_train)
                
                residuals = torch.abs(y_train - train_mu).squeeze().numpy()
                pointwise_losses = train_nll.squeeze().numpy()

            history.append({
                'epoch': epoch,
                'grid_x': x_grid.squeeze().numpy(),
                'grid_mu': grid_mu.squeeze().numpy(),
                'grid_scale': grid_scale.squeeze().numpy(),
                'residuals': residuals,
                'pointwise_losses': pointwise_losses,
                'total_loss': loss.item()
            })

    # Set up Matplotlib Figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"MLP Regression Uncertainty Fitting ({dist_name.upper()} Loss)", fontsize=14, fontweight='bold')

    ci_factor = get_95_ci_factor(dist_name)
    x_np = x_train.squeeze().numpy()
    y_np = y_train.squeeze().numpy()

    def update(frame_idx):
        ax_left.clear()
        ax_right.clear()

        frame = history[frame_idx]
        grid_x = frame['grid_x']
        grid_mu = frame['grid_mu']
        grid_scale = frame['grid_scale']
        residuals = frame['residuals']
        losses = frame['pointwise_losses']

        # ----------------------------------------------------
        # Left Subplot: Scatter Plot & 95% Confidence Interval
        # ----------------------------------------------------
        # Normal data points vs Outliers
        ax_left.scatter(x_np[~is_outlier], y_np[~is_outlier], color='blue', label='Inliers', zorder=4)
        if has_outliers:
            ax_left.scatter(x_np[is_outlier], y_np[is_outlier], color='red', s=60, label='Outliers', zorder=5)

        # Mean Line
        ax_left.plot(grid_x, grid_mu, color='black', lw=2, label='Predicted Mean (μ)')

        # 95% Confidence Region
        upper_bound = grid_mu + ci_factor * grid_scale
        lower_bound = grid_mu - ci_factor * grid_scale
        ax_left.fill_between(grid_x, lower_bound, upper_bound, color='deepskyblue', alpha=0.3, label='95% Confidence Region')

        ax_left.set_xlim([-1.2, 1.2])
        ax_left.set_ylim([-2.0, 2.0])
        ax_left.set_title(f"Epoch {frame['epoch']} - Fit & Uncertainty")
        ax_left.set_xlabel("x")
        ax_left.set_ylabel("y")
        ax_left.legend(loc='upper left')
        ax_left.grid(True, linestyle='--', alpha=0.5)

        # ----------------------------------------------------
        # Right Subplot: Per-point Residual Error vs. Pointwise Loss
        # ----------------------------------------------------
        indices = np.arange(len(x_np))
        
        # Color code bars by outlier status
        colors = ['red' if is_outlier[i] else 'blue' for i in range(len(x_np))]
        
        ax_right.scatter(residuals, losses, c=colors, s=50, alpha=0.8, zorder=3)
        for i in range(len(x_np)):
            ax_right.annotate(f"{i}", (residuals[i], losses[i]), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

        ax_right.set_title(f"Residual |y - μ| vs Pointwise Loss (NLL = {frame['total_loss']:.3f})")
        ax_right.set_xlabel("Absolute Residual |y - μ|")
        ax_right.set_ylabel("Pointwise NLL Loss")
        ax_right.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

    # Generate Animation
    anim = FuncAnimation(fig, update, frames=len(history), interval=100, repeat=False)
    
    output_filename = f"regression_{dist_name}_{'outliers' if has_outliers else 'no_outliers'}.gif"
    print(f"Saving animation to '{output_filename}'...")
    anim.save(output_filename, writer=PillowWriter(fps=15))
    print("Done!")

if __name__ == "__main__":
    main()
