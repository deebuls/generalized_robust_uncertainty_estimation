import torch

def LaplaceNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):

    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of scale must match that of input or be equal to 1
    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = (torch.log(2*scale) + torch.abs(input - target) / scale).view(input.size(0), -1).sum(dim=1)


    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    


def CauchyNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):

    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of scale must match that of input or be equal to 1
    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = (torch.log(3.14*scale) + 
            torch.log(1 + ((input - target)**2)/scale**2)) .view(input.size(0), -1).sum(dim=1)


    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    

class EvidentialLoss(torch.nn.Module):
    def __init__(self, mu, alpha, beta, lamda, targets, weight=None, size_average=True):
        super(EvidentialLoss, self).__init__()
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.targets = targets

    def forward(self,  mu, alpha, beta, lamda, targets,smooth=1):
        targets = targets.view(-1)
        y = self.mu.view(-1) #first column is mu,delta, predicted value
        loga = self.alpha.view(-1) #alpha
        logb = self.beta.view(-1) #beta
        logl = self.lamda.view(-1) #lamda

        a = torch.exp(loga)
        b = torch.exp(logb)
        l = torch.exp(logl)


        term1 = (torch.exp(torch.lgamma(a - 0.5)))/(4 * torch.exp(torch.lgamma(a)) * l * torch.sqrt(b))

        term2 = 2 * b *(1 + l) + (2*a - 1)*l*(y - targets)**2


        J = term1 * term2
 
        Kl_divergence = torch.abs(y - targets) * (2*a + l)
  

        loss = J + Kl_divergence

 
        return loss.mean()


def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)
    nll = 0.5 * torch.log(torch.pi / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    if reduce:
        return torch.mean(nll)
    else:
        return nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1)**2) \
        + 0.5 * v2 / v1 \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 \
        + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1)

    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    error = torch.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return torch.mean(reg) if reduce else reg

def EvidentialRegression(input, target, v, alpha, beta,  coeff=0.01):
    loss_nll = NIG_NLL(target, input, v, alpha, beta)
    loss_reg = NIG_Reg(target, input, v, alpha, beta)
    return loss_nll + coeff * loss_reg

def GeneralGaussianNLLLoss(input, target, alpha, beta, eps=1e-06, reduction='mean'): 
  
  # Inputs and targets much have same shape
  input = input.view(input.size(0), -1)
  target = target.view(target.size(0), -1)
  if input.size() != target.size():
      raise ValueError("input and target must have same size")

  # Second dim of scale must match that of input or be equal to 1
  alpha = alpha.view(input.size(0), -1)
  if alpha.size(1) != input.size(1) and alpha.size(1) != 1:
      raise ValueError("alpha is of incorrect size")

# Second dim of scale must match that of input or be equal to 1
  beta = beta.view(input.size(0), -1)
  if beta.size(1) != beta.size(1) and beta.size(1) != 1:
      raise ValueError("beta is of incorrect size")


  # Check validity of reduction mode
  if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
      raise ValueError(reduction + " is not valid")

  # Entries of var must be non-negative
  if torch.any(alpha < 0):
      raise ValueError("alpha has negative entry/entries")
  # Entries of var must be non-negative
  if torch.any(beta < 0):
      raise ValueError("beta has negative entry/entries")

  # Clamp for stability
  alpha = alpha.clone()
  beta = beta.clone()
  with torch.no_grad():
      alpha.clamp_(min=eps)
      beta.clamp_(min=eps)

  # Calculate loss (without constant)
  loss = (torch.abs(input - target)/alpha)**beta - torch.log(beta) + torch.log(2 * alpha ) + torch.lgamma(1/beta)


  # Apply reduction
  if reduction == 'mean':
      return loss.mean()
  elif reduction == 'sum':
      return loss.sum()
  else:
      return loss
