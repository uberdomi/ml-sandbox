import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Tuple, List

def train_NN(model: nn.Module,
             train_loader: DataLoader,
             val_loader: DataLoader,
             optimizer: torch.optim.Optimizer,
             loss_fn: nn.modules.loss._Loss,
             num_epochs: int,
             log_dir: str | None ="runs/experiment"
    ) -> Tuple[nn.Module, List[float], List[float]]:
    """
    General function to train a neural network with tensorboard logging.
    
    Args:
        model: PyTorch model inheriting from nn.Module
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        num_epochs: Number of training epochs
        log_dir: Directory for tensorboard logs
    
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # ============ TRAINING PHASE ============
        model.train()  # Set model to training mode
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * data.size(0)
            train_samples += data.size(0)
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)
        
        # ============ VALIDATION PHASE ============
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():  # Disable gradient computation
            for data, targets in val_loader:
                # Move data to device
                data, targets = data.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(data)
                
                # Calculate loss
                loss = loss_fn(outputs, targets)
                
                # Accumulate loss
                val_loss += loss.item() * data.size(0)
                val_samples += data.size(0)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)
        
        # ============ LOGGING ============
        # Log to tensorboard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    # Close tensorboard writer
    writer.close()
    
    print("-" * 50)
    print("Training completed!")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    
    return model, train_losses, val_losses


# ============ EXAMPLE USAGE ============

# Example 1: Simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage demonstration
def example_usage():
    """
    Demonstrates how to use the train_NN function with MNIST data
    """
    from torchvision import datasets, transforms
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST data
    train_dataset = datasets.MNIST('data/', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data/', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    model = SimpleCNN(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train the model
    trained_model, train_losses, val_losses = train_NN(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=10,
        log_dir="runs/mnist_experiment"
    )
    
    print("\nTo view tensorboard logs, run:")
    print("tensorboard --logdir=runs/mnist_experiment")
    
    return trained_model, train_losses, val_losses

# Example 2: Using with a different model and optimizer
def example_with_different_setup():
    """
    Shows flexibility of the train_NN function with different configurations
    """
    # Different model (simple MLP)
    class MLP(nn.Module):
        def __init__(self, input_size=784, hidden_size=256, num_classes=10):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Different optimizer and learning rate
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    # The same train_NN function works with any configuration!
    # trained_model, train_losses, val_losses = train_NN(
    #     model=model,
    #     train_loader=train_loader,  # Assume these are defined
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     num_epochs=20,
    #     log_dir="runs/mlp_experiment"
    # )

if __name__ == "__main__":
    # Run example
    example_usage()