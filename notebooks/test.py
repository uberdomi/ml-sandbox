
if __name__ == "__main__":
    from src import (
        MnistDataset,
        Regressor,
        Autoencoder,
    )

    # Machine learning
    import torch
    from torch import nn as nn
    
    hparams = dict(
        # Training hyperparameters
        batch_size=32,
        num_epochs=100,
        # Model hyperparameters
        learning_rate=1e-3,
        regularization_weight=1e-5,
    )
    
    mnist = MnistDataset()
    dataloaders = mnist.get_dataloaders(
        train_split=0.6,
        val_split=0.2,
        test_split=0.2,
        batch_size=hparams['batch_size']
    )

    model = Autoencoder(dims = [28*28, 128, 64, 32], sigma=nn.LeakyReLU)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    trainer = Regressor(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_metric=True,
        loss_function="MSE"
    ).initialize_regularization(
        method="L2",
        weight=hparams['regularization_weight']
    )
    
    results = trainer.train(
        train_dataloader=dataloaders['train'],
        validation_dataloader=dataloaders['val'],
        epochs=hparams['num_epochs']
    )