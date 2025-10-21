"""Trainer"""

# Typing
from typing import Any, Optional, Literal, NamedTuple, override
from abc import abstractmethod

# Machine learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

# Logging
from logging import Logger

from .log_utils import get_logger, get_summary_writer

# Utility classes
from .early_stopping import EarlyStopper
from .regularization import Regularizer


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
            raise ValueError(
                f"Tensor must contain exactly one element, got shape {tuple(x.shape)}"
            )
        return float(x.item())

    # Case 3: NumPy array
    if isinstance(x, np.ndarray):
        if x.ndim != 1 or x.size != 1:
            raise ValueError(
                f"NumPy array must be 1D with one element, got shape {x.shape}"
            )
        return float(x.item())

    # Case 4: Pandas Series
    if isinstance(x, pd.Series):
        if x.ndim != 1 or len(x) != 1:
            raise ValueError(
                f"Pandas Series must be 1D with one element, got length {len(x)}"
            )
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
    early_stopper: Optional[EarlyStopper] = None
    regularizer: Optional[Regularizer] = None
    weight_random_noise: float = 0.0

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        if_logging=True,
        scheduler_metric=False,
    ):
        """
        Helper class to unify the training procedures of the Pytorch models.
        Args:
            model: Pytorch Module.
            optimizer: Optimizer connected to the Pytorch module.
            scheduler: Optional Scheduler connected to the optimizer.
            if_logging: Enable logging throughout the training procedure.
            scheduler_metric: Use validation loss as a scheduler update step criterion (used only by the ReduceLROnPlateau scheduler).
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Detect the current device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if if_logging:
            self.logger = get_logger()

        self.scheduler_metric = scheduler_metric

    def log_info(self, message: str):
        if self.logger is not None:
            self.logger.info(message)

    # --- Regularization ---

    def initialize_regularization(
        self,
        method: Literal["L1", "L2", "Elastic"] = "L1",
        weight: float = 0.1,
        weight2: float = 0.1,
    ) -> "Trainer":
        """
        Initializes a regularization strategy.

        Args:
            method: Method used for regularization. Supported are "L1", "L2" and "Elastic".
            weight: Weight for the penalty term.
            weight2: Weight for the l2 regression in case of the elastic regularization.

        Returns:
            self reference, to chain the initialization methods.
        """
        self.regularizer = Regularizer(method=method, weight=weight, weight2=weight2)

        self.log_info(
            f"Regularization initialized with the method {method} and weights beta1 = {weight}, beta2 = {weight2}"
        )

        return self

    def calculate_regularization(self) -> float:
        """Calculate the regularization term based on the selected method."""

        if self.regularizer is None:
            return 0.0

        return self.regularizer.apply_regularization(self.model.named_parameters())

    # --- Early stopping ---

    def initialize_early_stopping(
        self,
        patience: int = 5,
        delta: float = 0.0,
    ) -> "Trainer":
        """
        Initializes the early stopping mechanism.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.

        Returns:
            self reference, to chain the initialization methods.
        """
        self.early_stopper = EarlyStopper(patience=patience, min_delta=delta)

        self.log_info(
            f"Early stopping initialized with parameters patience = {patience}, delta = {delta}"
        )

        return self

    def check_early_stopping(self, validation_loss: float) -> bool:
        """
        Checks if early stopping condition is met.

        Args:
            validation_loss: Current validation loss.
        Returns:
            Bool indicating whether to stop training early.
        """
        if self.early_stopper is None:
            return False

        return self.early_stopper(validation_loss)

    # --- Dataset specific methods ---

    @abstractmethod
    def move_batch_to_device(self, batch: Any) -> Any:
        """Move the batch object to the supported device and return it in the same format - to be used by further functions."""
        pass

    @abstractmethod
    def calculate_train_batch_loss(self, batch: Any) -> Any:
        """Calculate the loss function value for the current batch in the training phase - apply regularization and noise as opposed to the validation loop. The output value is used for the backward pass, so backward() should be callable on it (standard behavior for return values of loss function calls)."""
        pass

    @abstractmethod
    def calculate_validation_batch_loss(self, batch: Any) -> Any:
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
        self.model.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Reset the gradients
        epoch_loss = 0.0

        for batch in train_progress_bar:
            batch = self.move_batch_to_device(batch)
            loss = self.calculate_train_batch_loss(batch)

            loss.backward()  # Backward pass (compute gradients)
            self.optimizer.step()  # Update model parameters
            batch_loss = loss.item()

            epoch_loss += batch_loss

            # Update progress bar with current batch loss
            train_progress_bar.set_postfix({"Batch Train Loss": f"{batch_loss:.4f}"})

        return epoch_loss

    def validation_loop(self, validation_progress_bar: tqdm) -> float:
        """
        Perform the validation loop based on the input progress bar (with the validation dataloader) and return the corresponding loss.

        Args:
            validation_progress_bar: Progress bar based on the dataloader of the validation / test dataset.

        Returns:
            Validation/Test loss of the epoch.
        """
        self.model.eval()  # Set the model to validation mode
        validation_loss = 0.0

        with torch.no_grad():  # Don't compute the gradients
            for batch in validation_progress_bar:
                batch = self.move_batch_to_device(batch)
                batch_loss = self.calculate_validation_batch_loss(batch)

                validation_loss += batch_loss

                validation_progress_bar.set_postfix(
                    {"Batch Validation Loss": f"{batch_loss:.4f}"}
                )

        return validation_loss

    def train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader = None,
        epochs=100,
        weight_random_noise: float = 0.0,
        val_frequency=5,
        run_dir="logs",
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
        self.log_info("=" * 20 + " Model training started " + "=" * 20)
        self.weight_random_noise = weight_random_noise

        # Move model to device
        self.model.to(self.device)

        # Display current device
        self.log_info(f" Training on device: {self.device} ")

        # Initialize TensorBoard writer
        writer = get_summary_writer(run_dir)

        # Store the values for final return statement
        return_values = dict(
            train_loss=None,
            validation_loss=None,
            gradient_norm=None,
        )

        # ----- Training loop with tqdm progress bar -----
        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
            # --- Training phase ---
            epoch_loss = 0.0

            # Progress bar for batches within each epoch
            train_pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs} - Training",
                leave=False,
                unit="batch",
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
                    desc=f"Epoch {epoch + 1}/{epochs} - Validation",
                    leave=False,
                    unit="batch",
                )

                validation_loss = self.validation_loop(val_pbar)

                val_pbar.close()
                avg_val_loss = validation_loss / len(validation_dataloader)

                if self.check_early_stopping(validation_loss):
                    self.log_info(
                        ">" * 10
                        + " Early stopping condition fulfilled, exiting training loop"
                    )
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
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            if avg_val_loss is not None:
                return_values["validation_loss"] = avg_val_loss
                writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

            # Log learning rate if scheduler is used
            if self.scheduler is not None:
                current_lr = (
                    self.scheduler.get_last_lr()[0]
                    if hasattr(self.scheduler, "get_last_lr")
                    else self.optimizer.param_groups[0]["lr"]
                )
                writer.add_scalar("Learning_Rate", current_lr, epoch)

            # Log weight distributions and gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and "bias" not in name and "norm" not in name:
                    # Log weight distributions
                    writer.add_histogram(f"Weights/{name}", param.data, epoch)

                    # Log gradient distributions (if gradients exist)
                    if param.grad is not None:
                        writer.add_histogram(
                            f"Gradients/{name}", param.grad.data, epoch
                        )

                        # Log gradient norms to detect vanishing/exploding gradients
                        grad_norm = param.grad.data.norm(2)
                        writer.add_scalar(f"Gradient_Norms/{name}", grad_norm, epoch)

            # Calculate and log total gradient norm
            total_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            return_values["gradient_norm"] = total_norm
            writer.add_scalar("Total_Gradient_Norm", total_norm, epoch)

            # Print epoch summary
            if (epoch + 1) % val_frequency == 0 or epoch == 0:
                val_str = (
                    f"Val Loss: {avg_val_loss:,.2f}"
                    if avg_val_loss is not None
                    else "Val Loss: None"
                )
                tqdm.write(
                    f"Epoch [{epoch + 1:,}/{epochs:,}], Train Loss: {avg_train_loss:,.2f}, {val_str}, Grad Norm: {total_norm:,.2f}"
                )

        # Close TensorBoard writer
        writer.close()

        self.log_info(
            "=" * 20
            + f" Training completed! TensorBoard logs saved to '{run_dir}' "
            + "=" * 20
        )
        self.log_info(
            f"To view logs, run from the console: tensorboard --logdir={run_dir}"
        )

        return TrainingResults(
            model=self.model,
            train_loss=to_float(return_values["train_loss"]),
            validation_loss=to_float(return_values["validation_loss"]),
            gradient_norm=to_float(return_values["gradient_norm"]),
        )


# --- Implementations ---


class Classifier(Trainer):
    """Classifier model trainer. Implements the usage of the model data input, where the loss is calculated w.r.t. the target class labels."""

    def __init__(self, binary: bool = False, *args, **kwargs):
        """
        Args:
            binary: Whether to use binary classification (Binary Cross Entropy Loss) or multi-class classification (Cross Entropy Loss).
            *args: Additional arguments for the base Trainer class.
            **kwargs: Additional keyword arguments for the base Trainer class.
        """
        super().__init__(*args, **kwargs)
        if binary:
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()

    @override
    def move_batch_to_device(self, batch: Any) -> Any:
        *inputs, targets = batch
        inputs = [
            inp.to(self.device) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]
        targets = (
            targets.to(self.device) if isinstance(targets, torch.Tensor) else targets
        )

        return (*inputs, targets)

    @override
    def calculate_train_batch_loss(self, batch: Any) -> Any:
        *inputs, targets = batch

        logits = self.model(*inputs)

        # Don't add noise for classification tasks; compute the loss and apply regularization
        loss = self.loss_function(logits, targets)

        loss += self.calculate_regularization()

        return loss

    @override
    def calculate_validation_batch_loss(self, batch: Any) -> Any:
        *inputs, targets = batch

        prediction = self.model(*inputs)
        loss = self.loss_function(prediction, targets)

        return loss


class Regressor(Trainer):
    """Regressor model trainer. Implements the usage of the model data input, where the loss is calculated w.r.t. the target continuous values."""

    def __init__(self, loss_function: Literal["MSE", "L1"] = "MSE", *args, **kwargs):
        """
        Args:
            loss_function: The loss function to use for training. Supported are "MSE" (Mean Squared Error) and "L1" (Mean Absolute Error).
            *args: Additional arguments for the base Trainer class.
            **kwargs: Additional keyword arguments for the base Trainer class.
        """
        super().__init__(*args, **kwargs)
        if loss_function == "MSE":
            self.loss_function = nn.MSELoss()
        elif loss_function == "L1":
            self.loss_function = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

    @override
    def move_batch_to_device(self, batch: Any) -> Any:
        # Discard the targets from the batch and move to device
        *inputs, targets = batch
        inputs = [
            inp.to(self.device) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]

        return (*inputs,)

    @override
    def calculate_train_batch_loss(self, batch: Any) -> Any:
        # A tuple of inputs
        inputs = batch

        # Should have the same dimensions as inputs
        outputs = self.model(*inputs)
        inputs_noisy = inputs + self.weight_random_noise * torch.randn_like(inputs)

        # Compute the loss and apply regularization
        loss = self.loss_function(outputs, inputs_noisy)

        loss += self.calculate_regularization()

        return loss

    @override
    def calculate_validation_batch_loss(self, batch: Any) -> Any:
        # A tuple of inputs
        inputs = batch

        # Should have the same dimensions as inputs
        outputs = self.model(*inputs)
        loss = self.loss_function(outputs, inputs)

        return loss
