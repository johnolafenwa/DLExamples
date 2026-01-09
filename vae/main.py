import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 32):
        super().__init__()

        self.enc1 = nn.Linear(input_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, output_dim)
        self.log_var = nn.Linear(hidden_dim, output_dim)

        self.dec = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):

        # return z = mu + std * eps

        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std)

        z = mu + (std * noise)

        return z

    def encode(self, x: torch.Tensor):

        enc1 = F.relu(self.enc1(x))
        mu, log_var = self.mu(enc1), self.log_var(enc1)

        return mu, log_var 
    
    def decode(self, z: torch.Tensor):

        output = self.dec(z)

        return output
    
    def forward(self, input: torch.Tensor):

        mu, log_var = self.encode(input)

        z = self.reparameterize(mu, log_var)

        reconstructed = self.decode(z)

        return reconstructed, mu, log_var
    

def elbo_loss(input: torch.Tensor, reconstructed: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):

    bce_loss = F.binary_cross_entropy(reconstructed, input, reduction="sum")

    kl_loss = -0.5 * torch.sum( 1 + log_var - mu.pow(2) - log_var.exp())

    return bce_loss, kl_loss


def visualize_reconstructions(model, train_dataset, test_dataset, epoch, output_dir, device):
    model.eval()

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))

    # Get 8 samples from train and test
    train_images = torch.stack([train_dataset[i][0] for i in range(8)])
    test_images = torch.stack([test_dataset[i][0] for i in range(8)])

    with torch.no_grad():
        train_inputs = train_images.view(-1, 784).to(device)
        test_inputs = test_images.view(-1, 784).to(device)

        train_recon, _, _ = model(train_inputs)
        test_recon, _, _ = model(test_inputs)

    # Row 0: Train originals, Row 1: Train reconstructions
    # Row 2: Test originals, Row 3: Test reconstructions
    for i in range(8):
        axes[0, i].imshow(train_images[i].squeeze().cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Train Original', fontsize=10)

        axes[1, i].imshow(train_recon[i].view(28, 28).cpu(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Train Reconstructed', fontsize=10)

        axes[2, i].imshow(test_images[i].squeeze().cpu(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Test Original', fontsize=10)

        axes[3, i].imshow(test_recon[i].view(28, 28).cpu(), cmap='gray')
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title('Test Reconstructed', fontsize=10)

    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(images_dir / f"epoch_{epoch}.png")
    plt.close()

def train():

    device = "mps"

    model = VAE()
    model.to(device)

    output_dir = Path("outputs")

    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    train_transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    train_dataset = MNIST(root="./mnist", train=True, download=True, transform=train_transform)
    test_dataset = MNIST(root="./mnist", train=False, transform=test_transform)

    batch_size = 32
    num_epochs = 50
    save_steps = 5


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    for e in tqdm(range(num_epochs), total=num_epochs):

        model.train()

        train_bce_losses = []
        train_kl_losses = []

        for i, (images, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs = images.view(-1, 784).to(device)

            reconstructed, mu, log_var = model(inputs)

            bce_loss, kl_loss = elbo_loss(inputs, reconstructed, mu, log_var)

            total_loss = bce_loss + kl_loss

            total_loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            train_bce_losses.append(bce_loss.item())
            train_kl_losses.append(kl_loss.item())

        train_mean_bce_loss = sum(train_bce_losses) / len(train_bce_losses)
        train_mean_kl_loss = sum(train_kl_losses) / len(train_kl_losses)

        print(f"Train Metrics: Epoch: {e}, mean bce_loss: {train_mean_bce_loss}, mean kl_loss: {train_mean_kl_loss}")

        model.eval()

        test_bce_losses = []
        test_kl_losses = []

        for i, (images, _) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            with torch.no_grad():
                inputs = images.view(-1, 784).to(device)

                reconstructed, mu, log_var = model(inputs)

                bce_loss, kl_loss = elbo_loss(inputs, reconstructed, mu, log_var)

                total_loss = bce_loss + kl_loss

                test_bce_losses.append(bce_loss.item())
                test_kl_losses.append(kl_loss.item())
        
        test_mean_bce_loss = sum(test_bce_losses) / len(test_bce_losses)
        test_mean_kl_loss = sum(test_kl_losses) / len(test_kl_losses)

        print(f"Test Metrics: Epoch: {e}, mean bce_loss: {test_mean_bce_loss}, mean kl_loss: {test_mean_kl_loss}")

        if (e + 1) % save_steps == 0:
            output_file = output_dir / "checkpoints" / f"step_{e}.pt"
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model, output_file)

            visualize_reconstructions(model, train_dataset, test_dataset, e, output_dir, device)

        
if __name__ == "__main__":

    train()

        

        


        

        


            

            







