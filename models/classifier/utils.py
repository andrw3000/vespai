import torch
import matplotlib.pyplot as plt
from torchvision import utils
from sklearn.metrics import f1_score


def show_batch(sample_batch):
    """Show image for a batch of samples."""
    images = sample_batch[0]
    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')


def cv2mpl(image):
    """Converts from BGR in OpenCV to RGB."""
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict(probs, threshold=0.8):
    """Returns one-hot class predictions with probabilities as input.

    Args:
        probs: tensor of model outputs one-hot encoded, shape (b, num_classes)
        threshold: scalar threshold value
    Returns:
        preds: tensor of predictions; 0 vector indicates threshold failure
    """

    with torch.no_grad():
        preds = torch.zeros(probs.shape, dtype=torch.float32)
        batch_inds = torch.arange(preds.shape[0])
        vals, max_inds = torch.max(probs, dim=-1)
        preds[batch_inds, max_inds] = 1.
        preds[probs < threshold] = 0.  # Adjust for threshold failures
    return preds


def accuracy(model, activation, loader, thresholds=None, num_classes=2):
    """Computes the accuracy over a range of thresholds."""

    if thresholds is None:
        thresholds = list(torch.linspace(0.5, 0.95, 10))

    correct = torch.zeros((len(thresholds), num_classes + 1))
    # totals = torch.zeros((num_classes + 1,))
    for images, targets in loader:

        # Count the total number in each class
        # totals[:-1] += targets.sum(dim=0)
        # totals[-1] += targets.shape[0] - torch.sum(targets)

        with torch.no_grad():
            logits = model(images)
            for th, th_val in enumerate(thresholds):
                preds = predict(activation(logits), threshold=th_val)

                # Count correct values in each classes
                correct[th, :-1] += torch.logical_and(
                    torch.eq(targets, preds),
                    torch.eq(targets, 1),
                ).sum(dim=0) / targets.shape[0]

                # Count correctly identified unclassified images
                correct[th, -1] += torch.logical_and(
                    targets.sum(dim=-1) == 0,
                    preds.sum(dim=-1) == 0,
                ).sum() / targets.shape[0]

    acc = correct.sum(-1) / len(loader)

    return acc


def onehot2ids(onehotmat):
    """Decodes one-hot encoded predictions."""
    num_classes = onehotmat.shape[-1]
    vals, class_ids = onehotmat.max(dim=-1)
    class_ids[vals == 0] = num_classes  # Null class at the end
    return class_ids


def f1scores(model, activation, loader):
    """Computes the F1 Score over a range of thresholds."""

    thresholds = list(torch.linspace(0., 1., 21))
    y_true = torch.empty(0)
    y_pred = [torch.empty(0)] * len(thresholds)

    for images, targets in loader:
        y_true = torch.cat((y_true, onehot2ids(targets)), dim=0)
        with torch.no_grad():
            logits = model(images)
        for th, th_val in enumerate(thresholds):
            preds = predict(activation(logits), threshold=th_val)
            y_pred[th] = torch.cat((y_pred[th], onehot2ids(preds)), dim=0)

    f1 = []
    av = []
    for th, th_val in enumerate(thresholds):
        f1.append(f1_score(y_true, y_pred[th], average='micro'))
        av.append(torch.eq(y_true, y_pred[th]).sum() / len(loader))

        print('Threshold {:.02f} | F1 score {:.03f}'
              .format(th_val, f1[th])
              )

    return f1, thresholds
