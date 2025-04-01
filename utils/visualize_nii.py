import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

def visualize_mri_sample(sample, modality='FLAIR'):
    """
    Visualize a sample from MRIDataset (output of DataLoader).
    """
    img_tensor = sample[modality]
    roi_tensor = sample['ROI']

    # Convert tensors to NumPy
    img = img_tensor.squeeze().numpy()  # [H, W, D]
    roi = roi_tensor.numpy() if roi_tensor is not None else None

    # Create sliders for all three planes
    axial_slider = widgets.IntSlider(min=0, max=img.shape[2] - 1, step=1, value=img.shape[2] // 2,
                                     description='Axial')
    coronal_slider = widgets.IntSlider(min=0, max=img.shape[1] - 1, step=1, value=img.shape[1] // 2,
                                       description='Coronal')
    sagittal_slider = widgets.IntSlider(min=0, max=img.shape[0] - 1, step=1, value=img.shape[0] // 2,
                                        description='Sagittal')

    # Output widget to hold the plots
    output = widgets.Output()

    def update_view(change=None):
        with output:
            output.clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Axial view
            axes[0].imshow(img[:, :, axial_slider.value].T, cmap="gray", origin="lower")
            if roi is not None:
                axes[0].imshow(np.ma.masked_where(roi[:, :, axial_slider.value] == 0, roi[:, :, axial_slider.value]).T,
                               cmap='autumn', alpha=0.5, origin="lower")
            axes[0].set_title(f'Axial slice {axial_slider.value}')
            axes[0].axis("off")

            # Coronal view
            axes[1].imshow(img[:, coronal_slider.value, :].T, cmap="gray", origin="lower")
            if roi is not None:
                axes[1].imshow(np.ma.masked_where(roi[:, coronal_slider.value, :] == 0, roi[:, coronal_slider.value, :]).T,
                               cmap='autumn', alpha=0.5, origin="lower")
            axes[1].set_title(f'Coronal slice {coronal_slider.value}')
            axes[1].axis("off")

            # Sagittal view
            axes[2].imshow(img[sagittal_slider.value, :, :].T, cmap="gray", origin="lower")
            if roi is not None:
                axes[2].imshow(np.ma.masked_where(roi[sagittal_slider.value, :, :] == 0, roi[sagittal_slider.value, :, :]).T,
                               cmap='autumn', alpha=0.5, origin="lower")
            axes[2].set_title(f'Sagittal slice {sagittal_slider.value}')
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

    # Attach the handler to all sliders
    axial_slider.observe(update_view, names='value')
    coronal_slider.observe(update_view, names='value')
    sagittal_slider.observe(update_view, names='value')

    # Initial display
    update_view()

    # Show the sliders and output
    display(widgets.VBox([axial_slider, coronal_slider, sagittal_slider]), output)