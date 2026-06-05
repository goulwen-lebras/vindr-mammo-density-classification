import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.

    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of images and labels.

    criterion : torch.nn.Module
        Loss function.

    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.

    device : torch.device
        Device used for computation: CPU or CUDA.

    Returns
    -------
    average_loss : float
        Mean training loss over the epoch.
    """

    model.train()

    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)

    return average_loss