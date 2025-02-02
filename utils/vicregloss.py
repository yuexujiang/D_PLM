import torch
import torch.nn.functional as F

class VICRegLoss(torch.nn.Module):
    def __init__(self, lambda_var=25.0, mu=25.0, nu=1.0,pairdist_weight=0.0):
        super(VICRegLoss, self).__init__()
        self.lambda_var = lambda_var
        self.mu = mu
        self.nu = nu
        self.pairdist_weight = pairdist_weight
    
    def forward(self, z_a, z_b):
        batch_size, feature_dim = z_a.shape
        
        # Invariance Loss (MSE)
        invariance_loss = F.mse_loss(z_a, z_b)
        
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        # Variance Regularization
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        variance_loss = torch.mean(F.relu(1.0 - std_z_a)) + torch.mean(F.relu(1.0 - std_z_b))
        
        # Covariance Regularization
        cov_z_a = (z_a.T @ z_a) / (batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (batch_size - 1)
        identity = torch.eye(feature_dim, device=z_a.device)
        covariance_loss = (cov_z_a - identity).pow(2).sum() / feature_dim + (cov_z_b - identity).pow(2).sum() / feature_dim
        
        pair_dist1 = torch.cdist(z_a, z_a)
        pair_dist2 = torch.cdist(z_b, z_b)
        pairwise_distances = torch.cdist(pair_dist1, pair_dist2, p=2)
        pairdist_loss = pairwise_distances.mean()
        
        return self.lambda_var * invariance_loss + self.mu * variance_loss + self.nu * covariance_loss + self.pairdist_weight*pairdist_loss



class VICRegLoss_byol(torch.nn.Module):
    def __init__(self, lambda_var=25.0, mu=25.0, nu=1.0,pairdist_weight=0.0):
        super(VICRegLoss_byol, self).__init__()
        self.lambda_var = lambda_var
        self.mu = mu
        self.nu = nu
        self.pairdist_weight = pairdist_weight
    
    def forward(self, online, target):
        batch_size, feature_dim = online.shape
        
        # Invariance Loss (MSE)
        invariance_loss = F.mse_loss(online, target)
        
        online = online - online.mean(dim=0)
        target = target - target.mean(dim=0)
        # Variance Regularization
        std_online = torch.sqrt(online.var(dim=0) + 1e-4)
        std_target = torch.sqrt(target.var(dim=0) + 1e-4)
        variance_loss = torch.mean(F.relu(1.0 - std_online)) + torch.mean(F.relu(1.0 - std_target))
        
        # Covariance Regularization
        cov_z_a = (online.T @ online) / (batch_size - 1)
        cov_z_b = (target.T @ target) / (batch_size - 1)
        identity = torch.eye(feature_dim, device=online.device)
        covariance_loss = (cov_z_a - identity).pow(2).sum() / feature_dim + (cov_z_b - identity).pow(2).sum() / feature_dim
        
        pair_dist1 = torch.cdist(online, online)**2
        pair_dist2 = torch.cdist(target, target)**2
        #pairwise_distances = torch.cdist(pair_dist1, pair_dist2, p=2)
        #pairdist_loss = pairwise_distances.mean()
        pairdist_loss = torch.mean(F.relu(torch.logsumexp(-pair_dist1, dim=1))) + torch.mean(F.relu(torch.logsumexp(-pair_dist2, dim=1)))
        lossinfo=(invariance_loss.item(),variance_loss.item(),covariance_loss.item(),pairdist_loss.item())
        #print(f"(invar:{lossinfo[0]:.2f},var:{lossinfo[1]:.2f},covar:{lossinfo[2]:.2f},pdist{lossinfo[3]:.3f})")
        return self.lambda_var * invariance_loss + self.mu * variance_loss + self.nu * covariance_loss + self.pairdist_weight*pairdist_loss



import torch.nn as nn
import torchvision.models as models

class VICRegModel(nn.Module):
    def __init__(self, feature_dim=2048, projection_dim=128):
        super(VICRegModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections
