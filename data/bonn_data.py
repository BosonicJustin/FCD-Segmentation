import os
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BonnMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_roi=True):
        self.root_dir = root_dir
        self.transform = transform
        self.return_roi = return_roi

        # List subject folders
        self.subjects = sorted([d for d in os.listdir(root_dir) if d.startswith("sub-")])

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        anat_path = os.path.join(self.root_dir, subject_id, "anat")

        # File paths
        t1_path = os.path.join(anat_path, f"{subject_id}_acq-iso08_T1w.nii.gz")
        flair_path = os.path.join(anat_path, f"{subject_id}_acq-T2sel_FLAIR.nii.gz")
        roi_path = os.path.join(anat_path, f"{subject_id}_acq-T2sel_FLAIR_roi.nii.gz")

        # Load images
        t1_img = nib.load(t1_path).get_fdata()
        flair_img = nib.load(flair_path).get_fdata()

        # Normalize and convert to tensor
        t1_tensor = torch.from_numpy(t1_img).float().unsqueeze(0)
        flair_tensor = torch.from_numpy(flair_img).float().unsqueeze(0)

        sample = {
            'T1': t1_tensor,
            'FLAIR': flair_tensor,
            'subject_id': subject_id
        }

        # Load ROI only if it exists and is expected
        if self.return_roi and os.path.exists(roi_path):
            roi_img = nib.load(roi_path).get_fdata()
            sample['ROI'] = torch.from_numpy(roi_img).long()
        else:
            sample['ROI'] = torch.zeros(flair_img.shape, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample
