"""Trainer"""

# Typing
from typing import Any, Optional, Literal, NamedTuple
from abc import abstractmethod

# Helper libraries
from copy import deepcopy
from logging import Logger, getLogger, FileHandler, StreamHandler, Formatter
import logging
import shutil

# Machine learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd

# --- Logging ---

def get_logger(name: str = 'ml-sandbox') -> Logger:
    """Create a logging object in the debugging format.

    Args:
        name (str, optional): Name of the object. Defaults to 'ml-sandbox'.

    Returns:
        Logger: An instance of a logger.
    
    Example:
        >>> # Use the logger
        >>> logger.debug('This is a debug message')
        >>> logger.info('This is an info message')
        >>> logger.error('This is an error message')
    """
    # Create logger
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)

    # # Create file handler
    # file_handler = FileHandler('app.log')
    # file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger

# --- TensorBoard Summary Writer ---

def get_summary_writer(run_dir: str = "logs", delete_logs=True) -> SummaryWriter:
    """Get a base TensorBoard summary writer.

    Args:
        run_dir (str): The directory where the TensorBoard logs will be saved.

    Returns:
        SummaryWriter: A TensorBoard summary writer.
    """
    if delete_logs:
        shutil.rmtree(run_dir, ignore_errors=True)
    
    return SummaryWriter(log_dir=run_dir)

# --- Utility functions ---

def to_float(x):
    """
    Convert a 1D float-like object (torch, numpy, pandas, or scalar) to a primitive float.

    Idempotent: repeated calls return the same float value.
    Raises ValueError for non-1D or multi-element inputs.
    """
    # Case 1: primitive float/int (idempotent base case)
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    
    # Case 2: PyTorch tensor
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"Tensor must contain exactly one element, got shape {tuple(x.shape)}")
        return float(x.item())
    
    # Case 3: NumPy array
    if isinstance(x, np.ndarray):
        if x.ndim != 1 or x.size != 1:
            raise ValueError(f"NumPy array must be 1D with one element, got shape {x.shape}")
        return float(x.item())
    
    # Case 4: Pandas Series
    if isinstance(x, pd.Series):
        if x.ndim != 1 or len(x) != 1:
            raise ValueError(f"Pandas Series must be 1D with one element, got length {len(x)}")
        return float(x.iloc[0])
    
    # Otherwise unsupported
    raise TypeError(f"Unsupported type: {type(x).__name__}")

# --- Base class ---

class TrainingResults(NamedTuple):
    model: nn.Module
    train_loss: float
    validation_loss: float
    gradient_norm: float

class Trainer:
    logger: Optional[Logger] = None
    # Early Stopping parameters
    patience = 5
    delta = 0
    best_score = None
    early_stopping = False
    early_stop = False
    counter = 0
    best_model_state = None
    # Regularization parameters
    regularization: bool = False
    regularization_weight: float
    regularization_weight2: float
    regularization_method: Literal["L1","L2","Elastic"]
    weight_random_noise: float = 0.0

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.modules.loss._Loss = nn.MSELoss(),
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        if_logging = True,
        scheduler_metric = False,
    ):
        """
        Helper class to unify the training procedures of the Pytorch models.
        Args:
            model: Pytorch Module.
            optimizer: Optimizer connected to the Pytorch module.
            loss_function: Optional Loss function. Default is MSELoss.
            scheduler: Optional Scheduler connected to the optimizer.
            if_logging: Enable logging throughout the training procedure.
            scheduler_metric: Use validation loss as a scheduler update step criterion (used only by the ReduceLROnPlateau scheduler).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        # Detect the current device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if if_logging:
            self.logger = get_logger()

        self.scheduler_metric = scheduler_metric

    def log_info(self, message: str):
        if self.logger is not None:
            self.logger.info(message)

    # --- Regularization ---

    def set_regularization(
        self,
        regularization: bool=True,
        method: Literal["L1","L2","Elastic"] = 'L1',
        weight: float=0.1,
        weight2: float=0.1,
    ) -> None:
        """
        Sets the parameters for regularization.
        Args:
            regularization: Bool to activate / deactivate regularization.
            method: Method used for regularization. Supported are "L1", "L2" and "Elastic".
            weight: Weight for the penalty term.
            weight2: Weight for the l2 regression in case of the elastic regularization.

        Returns: None
        """
        self.regularization = regularization
        self.regularization_method = method
        self.regularization_weight = weight
        self.regularization_weight2 = weight2

        self.log_info(f"Regularization set to {regularization} with the method {method} and weights beta1 = {weight}, beta2 = {weight2}")

    def calculate_sum_weights(self, p: int):
        """
        Calculates the sum of Lp norms to the power p of all weight matrices.
        Args:
            p: Power factor.

        Returns:
            Sum of all Lp norms to the power of p.
        """
        # Don't include NormLayers and biases in the regularization
        return sum(param.abs().pow(p).sum() for name, param in self.model.named_parameters()
                    if "bias" not in name and "norm" not in name)

    def apply_regularization(self) -> float:
        """
        Calculates the proper regularization terms for the weight matrices.

        Returns:
            The regularization term to be added to the loss function.
        """
        if self.regularization_method == "L1":
            sum_params = self.calculate_sum_weights(1)
            return self.regularization_weight * sum_params
        elif self.regularization_method == "L2":
            sum_params = self.calculate_sum_weights(2)
            return self.regularization_weight * sum_params
        elif self.regularization_method == "Elastic":
            sum_params = self.calculate_sum_weights(1)
            sum_params2 = self.calculate_sum_weights(2)
            return self.regularization_weight * sum_params + self.regularization_weight2 * sum_params2
        else:
            return 0

    # --- Early stopping ---

    def set_early_stopping(
        self,
        early_stopping = True,
        patience = 5,
        delta = 0,
    ) -> None:
        """
        Used to initialize early stopping.
        
        Args:
            early_stopping: Bool to activate / deactivate early stopping.
            patience: Number of epochs to stop earlier if there is no improvement.
            delta: Minimal change of loss to qualify as improvement.

        Returns:
            None
        """
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta

        self.log_info(f"Early stopping set to {early_stopping} with parameters patience = {patience}, delta = {delta}")

    def early_stopping_check(self, val_loss) -> None:
        """
        Checks early stopping conditions and sets corresponding flags.
        
        Args:
            val_loss: The current epoch validation loss.

        Returns:
            None
        """
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = deepcopy(self.model.state_dict())
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = deepcopy(self.model.state_dict())
            self.counter = 0
            self.early_stop = False

    def load_best_model_early_stopping(self) -> None:
        """
        Loads the model with the lowest validation loss.
        
        Returns:
            None

        """
        if self.early_stopping:
            try:
                self.model.load_state_dict(self.best_model_state)
            except TypeError:
                print("Best Model cannot be accessed - Probably no Validation Set was provided during training")

    # --- Dataset specific methods ---

    @abstractmethod
    def move_batch_to_device(
        self,
        batch: Any
    ) -> Any:
        """Move the batch object to the supported device and return it in the same format - to be used by further functions."""
        pass

    @abstractmethod
    def calculate_train_batch_loss(
        self,
        batch: Any
    ) -> Any:
        """Calculate the loss function value for the current batch in the training phase - apply regularization and noise as opposed to the validation loop. The output value is used for the backward pass, so backward() should be callable on it (standard behavior for return values of loss function calls)."""
        pass

    @abstractmethod
    def calculate_validation_batch_loss(
        self,
        batch: Any
    ) -> Any:
        """Calculate the loss function value for the current batch in the validation phase - pure loss function, no regularization or input noise."""
        pass

    # --- Training loop ---

    def training_loop(
        self,
        train_progress_bar: tqdm,
    ) -> float:
        """
        Perform the train loop based on the input progress bar (with the train dataloader) and return the corresponding loss.
        Args:
            train_progress_bar: Progress bar based on the dataloader of the train dataset.

        Returns:
            Loss of the epoch.
        """
        self.model.train() # Set the model to training mode
        self.optimizer.zero_grad() # Reset the gradients
        epoch_loss = 0.0

        for batch in train_progress_bar:
            batch = self.move_batch_to_device(batch)
            loss = self.calculate_train_batch_loss(batch)

            loss.backward() # Backward pass (compute gradients)
            self.optimizer.step() # Update model parameters
            batch_loss = loss.item()

            epoch_loss += batch_loss

            # Update progress bar with current batch loss
            train_progress_bar.set_postfix({'Batch Train Loss': f'{batch_loss:.4f}'})

        return epoch_loss

    def validation_loop(
        self,
        validation_progress_bar: tqdm
    ) -> float:
        """
        Perform the validation loop based on the input progress bar (with the validation dataloader) and return the corresponding loss.

        Args:
            validation_progress_bar: Progress bar based on the dataloader of the validation / test dataset.

        Returns:
            Validation/Test loss of the epoch.
        """
        self.model.eval() # Set the model to validation mode
        validation_loss = 0.0

        with torch.no_grad(): # Don't compute the gradients
            for batch in validation_progress_bar:
                batch = self.move_batch_to_device(batch)
                batch_loss = self.calculate_validation_batch_loss(batch)

                validation_loss += batch_loss

                validation_progress_bar.set_postfix({'Batch Validation Loss': f'{batch_loss:.4f}'})

        return validation_loss

    def train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader = None,
        epochs = 100,
        weight_random_noise: float = 0,
        val_frequency = 5,
        run_dir = 'logs'
    ) -> TrainingResults:
        """
        Trains the Pytorch Module according with the Optimizer, Loss and Scheduler specified in the initialization.
        
        Args:
            train_dataloader: Dataloader of the Training Dataset.
            validation_dataloader: Optional Dataloader of the Validation Dataset.
            epochs: Number of epochs to train.
            weight_random_noise: Optional strength of random disturbance.
            val_frequency: Frequency of print output.
            run_dir: Directory to save TensorBoard logs.

        Return:
            TrainingResults: The trained model alongside with its training, validation errors and sum of gradient norms, which can be used to compare the peroformance of different settings.
        """
        self.log_info("="*20 + " Model training started " + "="*20)
        self.weight_random_noise = weight_random_noise

        # Move model to device
        self.model.to(self.device)

        # Display current device
        self.log_info(f" Training on device: {self.device} ")

        # Initialize TensorBoard writer
        writer = get_summary_writer(run_dir)

        # Store the values for final return statement
        return_values = dict(
            train_loss = None,
            validation_loss = None,
            gradient_norm = None,
        )

        # ----- Training loop with tqdm progress bar -----
        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
            # --- Training phase ---
            epoch_loss = 0.0
            
            # Progress bar for batches within each epoch
            train_pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{epochs} - Training",
                leave=False,
                unit="batch"
            )
            epoch_loss = self.training_loop(train_pbar)

            train_pbar.close()

            # Calculate average training loss
            avg_train_loss = epoch_loss / len(train_dataloader)

            # --- Validation phase ----
            avg_val_loss = None
            if validation_dataloader is not None:
                val_pbar = tqdm(
                    validation_dataloader,
                    desc=f"Epoch {epoch+1}/{epochs} - Validation",
                    leave=False,
                    unit="batch"
                )
            
                validation_loss = self.validation_loop(val_pbar)
            
                val_pbar.close()
                avg_val_loss = validation_loss / len(validation_dataloader)
            
                if self.early_stopping:
                    self.early_stopping_check(validation_loss)
                    if self.early_stop:
                        self.log_info(">"*10 + " Early stopping condition fulfilled, exiting training loop")
                        break

            # - Scheduler step -
            if self.scheduler is not None:
                if self.scheduler_metric:
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # --- End of epoch logging ---
            # TensorBoard logging
            return_values["train_loss"] = avg_train_loss
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            if avg_val_loss is not None:
                return_values["validation_loss"] = avg_val_loss
                writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            # Log learning rate if scheduler is used
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Log weight distributions and gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and "bias" not in name and "norm" not in name:
                    # Log weight distributions
                    writer.add_histogram(f'Weights/{name}', param.data, epoch)

                    # Log gradient distributions (if gradients exist)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)

                        # Log gradient norms to detect vanishing/exploding gradients
                        grad_norm = param.grad.data.norm(2)
                        writer.add_scalar(f'Gradient_Norms/{name}', grad_norm, epoch)

            # Calculate and log total gradient norm
            total_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            return_values["gradient_norm"] = total_norm
            writer.add_scalar('Total_Gradient_Norm', total_norm, epoch)

            # Print epoch summary
            if (epoch + 1) % val_frequency == 0 or epoch == 0:
                val_str = f"Val Loss: {avg_val_loss:,.2f}" if avg_val_loss is not None else "Val Loss: None"
                tqdm.write(f"Epoch [{epoch+1:,}/{epochs:,}], Train Loss: {avg_train_loss:,.2f}, {val_str}, Grad Norm: {total_norm:,.2f}")

        # Close TensorBoard writer
        writer.close()

        self.log_info("="*20 + f" Training completed! TensorBoard logs saved to '{run_dir}' " + "="*20)
        self.log_info(f"To view logs, run from the console: tensorboard --logdir={run_dir}")

        return TrainingResults(
            model = self.model,
            train_loss = to_float(return_values["train_loss"]),
            validation_loss = to_float(return_values["validation_loss"]),
            gradient_norm = to_float(return_values["gradient_norm"]),
        )



# --- Implementations ---
class NN_Trainer(Trainer):
    """Neural network model trainer. Implements the usage of the model data input, where the loss is calculated w.r.t. the target values."""

    def move_batch_to_device(
        self,
        batch: Any
    ) -> Any:
        *inputs, targets = batch
        inputs = [inp.to(self.device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        targets = targets.to(self.device) if isinstance(targets, torch.Tensor) else targets

        return (*inputs, targets)

    def calculate_train_batch_loss(
        self,
        batch: Any
    ) -> Any:
        *inputs, targets = batch

        outputs = self.model(*inputs)
        targets_noisy = targets + self.weight_random_noise * torch.randn_like(targets)

        # Compute the loss and apply regularization
        loss = self.loss_function(outputs, targets_noisy.squeeze(-1))
        if self.regularization:
            loss += self.apply_regularization()

        return loss

    def calculate_validation_batch_loss(
        self,
        batch: Any
    ) -> Any:        
        *inputs, targets = batch

        prediction = self.model(*inputs)
        loss = self.loss_function(prediction, targets)

        return loss



