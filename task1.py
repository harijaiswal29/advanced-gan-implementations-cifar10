import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F
from scipy.stats import entropy
from torchvision.models import inception_v3
from scipy import linalg
import os
from tqdm import tqdm
import logging
import json
from datetime import datetime

# Set up logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_improved.log'),
        logging.StreamHandler()
    ]
)

class Config:
    def __init__(self):
        # Model parameters - increased capacity
        self.latent_dim = 128  # Increased from 100
        self.num_classes = 10
        self.embedding_dim = 128  # Increased from 100
        self.ngf = 96  # Increased from 64 (Generator filters)
        self.ndf = 96  # Increased from 64 (Critic filters)
        
        # Training parameters - improved stability
        self.num_epochs = 150  # Increased training time
        self.batch_size = 64
        self.n_critic = 3  # Reduced from 5 for better balance
        self.lambda_gp = 15  # Increased from 10 for stronger gradient penalty
        self.lr_g = 0.0001  # Separate learning rates
        self.lr_c = 0.0003  # Higher LR for critic
        self.beta1 = 0.0  # Changed from 0.5 for better stability
        self.beta2 = 0.9  # Changed from 0.999
        
        # Evaluation parameters
        self.n_eval_samples = 10000
        self.n_eval_batches = self.n_eval_samples // self.batch_size
        self.n_splits = 10
        self.eval_frequency = 10  # Evaluate every 10 epochs
        
        # Directories
        self.sample_dir = 'samples_improved'
        self.checkpoint_dir = 'checkpoints_improved'
        self.log_dir = 'logs_improved'
        
        # Create directories
        for directory in [self.sample_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)

config = Config()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# Improved Generator with residual connections and self-attention
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        residual = self.bypass(x)
        if hasattr(self, 'upsample'):
            residual = self.upsample(residual)
            x = self.upsample(x)
            
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.activation(x)
        return x

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width*height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(config.num_classes, config.embedding_dim)
        
        # Initial dense layer
        self.init_size = 4
        self.l1 = nn.Sequential(
            nn.Linear(config.latent_dim + config.embedding_dim, config.ngf * 8 * self.init_size ** 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Improved architecture with residual blocks and self-attention
        self.res_blocks = nn.Sequential(
            nn.BatchNorm2d(config.ngf * 8),
            ResBlockGenerator(config.ngf * 8, config.ngf * 4),  # 4x4 -> 8x8
            ResBlockGenerator(config.ngf * 4, config.ngf * 2),  # 8x8 -> 16x16
            SelfAttention(config.ngf * 2),  # Self-attention at 16x16 resolution
            ResBlockGenerator(config.ngf * 2, config.ngf),      # 16x16 -> 32x32
        )
        
        # Final convolution to get 3 channels
        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(config.ngf),
            nn.Conv2d(config.ngf, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat((noise, label_embedding), -1)
        x = self.l1(x)
        x = x.view(x.shape[0], -1, self.init_size, self.init_size)
        x = self.res_blocks(x)
        return self.final_conv(x)

# Spectral Normalization for Critic stability
def spectral_norm(module):
    nn.utils.spectral_norm(module)
    return module

class ResBlockCritic(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResBlockCritic, self).__init__()
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1))
        self.bypass = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.bypass = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0))
            
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        residual = self.bypass(x)
        if self.downsample:
            residual = F.avg_pool2d(residual, 2)
            x = F.avg_pool2d(x, 2)
            
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = x + residual
        x = self.activation(x)
        return x

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.label_embedding = nn.Embedding(config.num_classes, config.embedding_dim)
        
        # Improved critic with resblocks and spectral normalization
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(3, config.ndf, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.blocks = nn.Sequential(
            ResBlockCritic(config.ndf, config.ndf * 2, downsample=True),   # 32x32 -> 16x16
            SelfAttention(config.ndf * 2),  # Self-attention
            ResBlockCritic(config.ndf * 2, config.ndf * 4, downsample=True),  # 16x16 -> 8x8
            ResBlockCritic(config.ndf * 4, config.ndf * 8, downsample=True),  # 8x8 -> 4x4
        )
        
        # Layer normalization is still useful for overall stability
        self.final_ln = nn.LayerNorm([config.ndf * 8, 4, 4])
        
        # Final fully connected layer
        self.adv_layer = spectral_norm(nn.Linear(config.ndf * 8 * 4 * 4 + config.embedding_dim, 1))

    def forward(self, img, labels):
        features = self.initial(img)
        features = self.blocks(features)
        features = self.final_ln(features)
        features = features.view(features.shape[0], -1)
        label_embedding = self.label_embedding(labels)
        d_in = torch.cat((features, label_embedding), -1)
        return self.adv_layer(d_in)

def get_inception_model():
    """Initialize and prepare inception model for feature extraction"""
    inception = inception_v3(pretrained=True, transform_input=True)
    inception.fc = nn.Identity()
    inception.eval()
    return inception.to(device)

def compute_gradient_penalty(critic, real_samples, fake_samples, labels):
    """Compute gradient penalty for WGAN-GP with improved implementation"""
    batch_size = real_samples.size(0)
    alpha = torch.rand((batch_size, 1, 1, 1)).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Flatten gradients to compute norm per sample
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_class_statistics(class_idx=1):
    """Compute statistics of real images for a specific class"""
    logging.info(f"Computing real image statistics for class {class_idx}...")
    inception = get_inception_model()
    
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
    indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
    class_dataset = Subset(dataset, indices)
    dataloader = DataLoader(class_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    real_features = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f"Processing class {class_idx} images"):
            try:
                images = images.to(device)
                features = inception(images).cpu().numpy()
                real_features.append(features)
                
                if len(real_features) * 32 >= config.n_eval_samples:
                    break
            except RuntimeError as e:
                logging.error(f"Error processing batch: {e}")
                continue
    
    real_features = np.concatenate(real_features, axis=0)[:config.n_eval_samples]
    logging.info(f"Using {len(real_features)} real samples for FID")
    mu = np.mean(real_features, axis=0)
    sigma = np.cov(real_features, rowvar=False)
    
    return mu, sigma

def calculate_inception_score(generator, class_idx=1):
    """Calculate Inception Score for a specific class"""
    logging.info(f"Calculating Inception Score for class {class_idx}...")
    generator.eval()
    inception_model = get_inception_model()
    
    predictions = []
    with torch.no_grad():
        for _ in tqdm(range(config.n_eval_batches), desc="Generating samples"):
            z = torch.randn(config.batch_size, config.latent_dim).to(device)
            labels = torch.full((config.batch_size,), class_idx, dtype=torch.long).to(device)
            images = generator(z, labels)
            
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            pred = F.softmax(inception_model(images), dim=1).cpu().numpy()
            predictions.append(pred)
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Split predictions and calculate scores
    split_scores = []
    predictions_per_split = predictions.shape[0] // config.n_splits
    for k in range(config.n_splits):
        part = predictions[k * predictions_per_split:(k + 1) * predictions_per_split]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Calculate Fréchet Inception Distance with proper numerical handling"""
    # Ensure matrices are symmetric and positive semi-definite
    sigma1 = (sigma1 + sigma1.T) / 2
    sigma2 = (sigma2 + sigma2.T) / 2
    
    # Add small diagonal term to ensure positive definiteness
    eps = 1e-6
    sigma1 += eps * np.eye(sigma1.shape[0])
    sigma2 += eps * np.eye(sigma2.shape[0])
    
    diff = mu1 - mu2
    
    # Calculate sqrt(a*b) term
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Numerical stability check
    if np.iscomplexobj(covmean):
        if not np.allclose(np.zeros_like(covmean.imag), covmean.imag, rtol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    # Calculate FID
    fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    
    # Ensure non-negative
    if fid < 0:
        logging.warning(f"Negative FID value encountered: {fid}")
        fid = 0
    
    return float(fid)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = device
        
        # Initialize networks
        self.generator = Generator(config).to(device)
        self.critic = Critic(config).to(device)
        
        # Optimizers with different learning rates
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=config.lr_g, 
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_C = optim.Adam(
            self.critic.parameters(), 
            lr=config.lr_c, 
            betas=(config.beta1, config.beta2)
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G, T_max=config.num_epochs, eta_min=config.lr_g/10)
        self.scheduler_C = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_C, T_max=config.num_epochs, eta_min=config.lr_c/10)
        
        # Training history
        self.history = {
            'g_losses': [],
            'c_losses': [],
            'inception_scores': [],
            'fid_scores': []
        }
        
        # Data augmentation for improved training
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

    def train_epoch(self, epoch):
        self.generator.train()
        self.critic.train()
        
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for i, (real_imgs, labels) in pbar:
            batch_size = real_imgs.shape[0]
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            # Train Critic
            self.optimizer_C.zero_grad()
            
            # Generate fake images with noise
            z = torch.randn(batch_size, self.config.latent_dim).to(device)
            fake_imgs = self.generator(z, labels)
            
            # Add instance noise for stability (decreases over epochs)
            noise_std = max(0, 0.1 * (1 - epoch / 50))  # Decrease from 0.1 to 0
            if noise_std > 0:
                real_imgs = real_imgs + torch.randn_like(real_imgs) * noise_std
                fake_imgs = fake_imgs + torch.randn_like(fake_imgs) * noise_std
            
            real_validity = self.critic(real_imgs, labels)
            fake_validity = self.critic(fake_imgs.detach(), labels)
            
            gradient_penalty = compute_gradient_penalty(
                self.critic, real_imgs, fake_imgs.detach(), labels
            )
            
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + \
                     self.config.lambda_gp * gradient_penalty
            
            c_loss.backward()
            self.optimizer_C.step()
            
            self.history['c_losses'].append(c_loss.item())

            # Train Generator
            if i % self.config.n_critic == 0:
                self.optimizer_G.zero_grad()                
                fake_validity = self.critic(fake_imgs, labels)
                g_loss = -torch.mean(fake_validity)                
                g_loss.backward()
                self.optimizer_G.step()                
                self.history['g_losses'].append(g_loss.item())
                
                desc = f"[Epoch {epoch}/{self.config.num_epochs}] [C loss: {c_loss.item():.4f}]"
                if i % self.config.n_critic == 0:
                    desc += f" [G loss: {g_loss.item():.4f}]"
                # Update progress bar
                pbar.set_description(desc)
                
        # Update learning rates
        self.scheduler_G.step()
        self.scheduler_C.step()

    def evaluate(self, class_idx=1):
        """Evaluate the model for a specific class"""
        logging.info(f"Evaluating model for class {class_idx}...")
        
        # Generate samples for evaluation
        self.generator.eval()
        inception_model = get_inception_model()
        
        # Get real statistics
        real_mu, real_sigma = compute_class_statistics(class_idx)
        
        # Generate features for fake images
        fake_features = []
        with torch.no_grad():
            for _ in tqdm(range(config.n_eval_batches), desc="Generating evaluation samples"):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                labels = torch.full((config.batch_size,), class_idx, dtype=torch.long).to(device)
                fake_imgs = self.generator(z, labels)
                
                # Resize for inception
                fake_imgs = F.interpolate(fake_imgs, size=(299, 299), mode='bilinear', align_corners=False)
                features = inception_model(fake_imgs).cpu().numpy()
                fake_features.append(features)
        
        fake_features = np.concatenate(fake_features, axis=0)[:config.n_eval_samples]
        logging.info(f"Using {len(fake_features)} fake samples for FID")
        fake_mu = np.mean(fake_features, axis=0)
        fake_sigma = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        fid_score = calculate_fid(real_mu, real_sigma, fake_mu, fake_sigma)
        
        # Calculate Inception Score
        is_mean, is_std = calculate_inception_score(self.generator, class_idx)
        
        metrics = {
            'class_idx': class_idx,
            'n_samples': config.n_eval_samples,
            'inception_score': {
                'mean': float(is_mean),
                'std': float(is_std)
            },
            'fid_score': float(fid_score),
            'timestamp': str(datetime.now())
        }
        
        # Explicitly log the metrics
        logging.info(f"Class {class_idx} Metrics: IS = {is_mean:.4f} ± {is_std:.4f}, FID = {fid_score:.4f}")
        
        return metrics

    def save_checkpoint(self, epoch, metrics=None):
        """Save model checkpoint and metrics"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_C_state_dict': self.optimizer_C.state_dict(),
            'history': self.history
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

    def generate_samples(self, n_samples=10, label=1):
        """Generate samples for a specific class"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim).to(device)
            labels = torch.full((n_samples,), label, dtype=torch.long).to(device)
            samples = self.generator(z, labels)
            return samples

def save_metrics(metrics, filename):
    """Save evaluation metrics to JSON"""
    metrics_dir = os.path.join(config.log_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    filepath = os.path.join(metrics_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
    
    logging.info(f"Saved metrics to {filepath}")

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    """Main training loop"""
    # Save configuration
    with open(os.path.join(config.log_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=4)
    
    # Initialize trainer
    trainer = Trainer(config)
    logging.info("Starting training with improved WGAN-GP...")
    
    # Training loop
    for epoch in range(config.num_epochs):
        trainer.train_epoch(epoch)
        
        # Evaluate and save samples periodically
        if epoch % config.eval_frequency == 0 or epoch == config.num_epochs - 1:
            # Generate and save automobile samples
            samples = trainer.generate_samples(n_samples=10, label=1)
            save_image(
                samples,
                os.path.join(config.sample_dir, f'automobile_samples_epoch_{epoch}.png'),
                nrow=5,
                normalize=True
            )
            
            # Evaluate model
            metrics = trainer.evaluate(class_idx=1)
            save_metrics(metrics, f'metrics_epoch_{epoch}.json')
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, metrics)
            
            # Log results with prominent metrics display
            logging.info(
                f"======= EVALUATION EPOCH {epoch} =======\n"
                f"Inception Score = {metrics['inception_score']['mean']:.4f} ± "
                f"{metrics['inception_score']['std']:.4f}\n"
                f"FID Score = {metrics['fid_score']:.4f}\n"
                f"======================================="
            )
    
    # Final evaluation
    logging.info("\n\n==== TRAINING COMPLETED: FINAL EVALUATION ====")
    final_metrics = trainer.evaluate(class_idx=1)
    save_metrics(final_metrics, 'final_metrics.json')
    
    # Generate final samples
    final_samples = trainer.generate_samples(n_samples=10, label=1)
    save_image(
        final_samples,
        os.path.join(config.sample_dir, 'final_automobile_samples.png'),
        nrow=5,
        normalize=True
    )
    
    # Save final model
    trainer.save_checkpoint(config.num_epochs, final_metrics)
    
    # Display final metrics prominently
    logging.info(
        f"\n\n========== FINAL RESULTS ==========\n"
        f"Model: Improved Conditional WGAN-GP\n"
        f"Class: Automobile\n"
        f"Inception Score: {final_metrics['inception_score']['mean']:.4f} ± "
        f"{final_metrics['inception_score']['std']:.4f}\n"
        f"FID Score: {final_metrics['fid_score']:.4f}\n"
        f"====================================="
    )

if __name__ == "__main__":
    main()