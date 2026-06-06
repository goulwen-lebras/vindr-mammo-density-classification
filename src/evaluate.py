import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, dataloader, device):
    """
    Evaluate a classification model and return metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.

    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of images and labels.

    device : torch.device
        Device used for computation: CPU or CUDA.

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, macro-F1, balanced accuracy,
        and confusion matrix.
    """

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(
            all_labels,
            all_predictions,
        ),
        "confusion_matrix": confusion_matrix(
            all_labels,
            all_predictions,
        ),
    }

    return metrics