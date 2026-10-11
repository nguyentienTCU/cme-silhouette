import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from .losses import combined_loss
from .metrics import iou_metric

def overlay_mask(image, mask, alpha=0.5):
    """Overlay predicted mask (binary) onto grayscale image."""
    # If image is 2D, convert to RGB for overlay
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image
    mask_rgb = np.zeros_like(image_rgb)
    # Mark the red channel in areas where mask==1
    mask_rgb[:, :, 0] = mask * 255
    overlay = image_rgb.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return overlay.astype(np.uint8)

def validate(model, dataloader, device, alpha_loss):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = combined_loss(outputs, masks, alpha=alpha_loss)

            running_loss += loss.item() * inputs.size(0)
            running_iou += iou_metric(outputs, masks) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)

    return epoch_loss, epoch_iou

def visualize_predictions(model, dataloader, device, model_name, output_dir, num_samples=30, threshold=0.5, show=True):
    model.eval()
    all_inputs, all_masks, all_outputs = [], [], []
    total_collected = 0

    # Loop over batches until we collect enough samples.
    for batch in dataloader:
        inputs, masks = batch
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        outputs = torch.sigmoid(outputs).cpu().numpy()
        inputs = inputs.cpu().numpy()
        masks = masks.cpu().numpy()

        all_inputs.append(inputs)
        all_masks.append(masks)
        all_outputs.append(outputs)

        total_collected += inputs.shape[0]
        if total_collected >= num_samples:
            break

    # Concatenate the batches along the first dimension.
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # Use the minimum of num_samples and collected samples.
    num_samples = min(num_samples, all_inputs.shape[0])

    # Create subplots: 4 columns per sample (original, ground truth, predicted, overlay)
    fig, axs = plt.subplots(num_samples, 4, figsize=(25, 5*num_samples), squeeze=False)
    for i in range(num_samples):
        # Original Input
        axs[i, 0].imshow(all_inputs[i, 0, :, :], cmap='gray')
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis("off")

        # Ground Truth Mask
        axs[i, 1].imshow(all_masks[i, 0, :, :], cmap='gray')
        axs[i, 1].set_title("Ground Truth Mask")
        axs[i, 1].axis("off")

        # Predicted Mask (binary)
        pred_mask = (all_outputs[i, 0, :, :] > threshold).astype(np.float32)
        axs[i, 2].imshow(pred_mask, cmap='gray')
        axs[i, 2].set_title("Predicted Mask")
        axs[i, 2].axis("off")

        # Overlay: Input + Predicted Mask (using imshow overlay technique)
        axs[i, 3].imshow(all_inputs[i, 0, :, :], cmap='gray')
        axs[i, 3].imshow(pred_mask, cmap='Reds', alpha=0.3)
        axs[i, 3].set_title("Overlay: Input + Prediction")
        axs[i, 3].axis("off")

    plt.tight_layout()

    # Save inside the experiment folder
    plt.savefig(
        os.path.join(output_dir, f"predicted_samples_{model_name}.png"),
        dpi=150,
        bbox_inches="tight"
    )

    if show:
        plt.show()
    plt.close(fig)
