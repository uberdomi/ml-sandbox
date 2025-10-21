"""Early stopping mechanism to stop the training procedure when validation loss stops improving."""

class EarlyStopper:
    """Class to monitor validation loss and determine when to stop training early."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initialize the EarlyStopper.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        """
        Call method to check if training should be stopped.

        Args:
            val_loss (float): Current validation loss.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False

    # TODO : may include monitoring best model state as well
