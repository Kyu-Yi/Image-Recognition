import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class SkinLesionDataset(Dataset):
    """Dataset class for loading skin lesion images"""

    def __init__(self, df, transform=None):

        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label


class DataPreparation:
    """Class for data preparation tasks"""

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
        self.images_dir = data_dir

        # Define transformations for training and validation/testing
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Define lesion type mapping
        self.lesion_type_dict = {
            'nv': 0,  # Melanocytic nevi
            'mel': 1,  # Melanoma
            'bkl': 2,  # Benign keratosis-like lesions
            'bcc': 3,  # Basal cell carcinoma
            'akiec': 4,  # Actinic keratoses
            'vasc': 5,  # Vascular lesions
            'df': 6  # Dermatofibroma
        }

    def load_and_prepare_data(self, samples_per_class=None):
        """
        Load and prepare the dataset
        """
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)

        # Map image IDs to file paths
        image_path_dict = {}
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_id = file.split('.')[0]
                    image_path_dict[image_id] = os.path.join(root, file)

        # Add file paths to metadata
        metadata['image_path'] = metadata['image_id'].map(image_path_dict)

        # Filter out rows with missing image paths
        metadata = metadata.dropna(subset=['image_path'])

        # Map diagnosis to numerical labels
        metadata['label'] = metadata['dx'].map(self.lesion_type_dict)

        # If samples_per_class is provided, limit the dataset size for testing
        if samples_per_class is not None:
            # Keep only a small number of samples per class
            balanced_df = pd.DataFrame()
            for class_idx in range(len(self.lesion_type_dict)):
                class_samples = metadata[metadata['label'] == class_idx].sample(
                    n=min(samples_per_class, sum(metadata['label'] == class_idx)),
                    random_state=42
                )
                balanced_df = pd.concat([balanced_df, class_samples])

            metadata = balanced_df.reset_index(drop=True)
            print(f"Using subset of data: {len(metadata)} samples total")

        # Split data into train, validation, and test sets
        train_val_df, test_df = train_test_split(
            metadata, test_size=0.2, random_state=42, stratify=metadata['label']
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label']
        )

        # Create datasets
        train_dataset = SkinLesionDataset(train_df, transform=self.train_transform)
        val_dataset = SkinLesionDataset(val_df, transform=self.val_transform)
        test_dataset = SkinLesionDataset(test_df, transform=self.val_transform)

        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # Calculate class weights for handling class imbalance
        class_counts = metadata['label'].value_counts().sort_index().values
        class_weights = torch.FloatTensor(1.0 / class_counts)
        class_weights = class_weights / class_weights.sum()

        # Print dataset information
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
        print(f"Class distribution: {class_counts}")

        return train_loader, val_loader, test_loader, class_weights, self.lesion_type_dict


def get_class_names(lesion_type_dict):
    """
    Get class names from lesion type dictionary
    """
    idx_to_class = {v: k for k, v in lesion_type_dict.items()}
    class_names = [idx_to_class[i] for i in range(len(lesion_type_dict))]
    return class_names