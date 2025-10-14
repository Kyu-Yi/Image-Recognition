import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Visualizer:
    """Class for visualization utilities"""

    def __init__(self, model, device, lesion_type_dict, results_dir='./results'):
        """
        Initialize the visualizer
        """
        self.model = model
        self.device = device
        self.lesion_type_dict = lesion_type_dict
        self.results_dir = results_dir

        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Create mapping from index to class name
        self.idx_to_class = {v: k for k, v in lesion_type_dict.items()}
        self.class_names = [self.idx_to_class[i] for i in range(len(lesion_type_dict))]

        # Define normalization transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def visualize_class_distribution(self, dataset_df):
        """
        Visualize class distribution in the dataset
        """
        plt.figure(figsize=(12, 6))

        # Count samples per class
        class_counts = dataset_df['label'].value_counts().sort_index()

        # Map indices to class names
        class_names = [self.idx_to_class[i] for i in class_counts.index]

        # Create bar plot
        ax = sns.barplot(x=class_names, y=class_counts.values)
        plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xlabel('Lesion Class', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add count labels
        for i, p in enumerate(ax.patches):
            ax.annotate(
                f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points'
            )

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'class_distribution.png'), dpi=300)
        plt.close()
        print(f"Class distribution plot saved to {os.path.join(self.results_dir, 'class_distribution.png')}")

    def visualize_sample_images(self, dataset, num_per_class=2):
        """
        Visualize sample images from each class
        """
        # Create figure
        fig, axes = plt.subplots(len(self.lesion_type_dict), num_per_class,
                                 figsize=(num_per_class * 3, len(self.lesion_type_dict) * 3))
        fig.suptitle('Sample Images from Each Lesion Class', fontsize=16, fontweight='bold', y=0.98)

        # Initialize counters for each class
        class_counters = {i: 0 for i in range(len(self.lesion_type_dict))}

        # Track if we've found enough examples
        completed_classes = set()

        # Loop through dataset
        for i in range(len(dataset)):
            image, label = dataset[i]

            # Skip if we already have enough examples of this class
            if label in completed_classes:
                continue

            # Convert tensor to numpy for display
            if isinstance(image, torch.Tensor):
                img = image.numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
            else:
                img = image

            # Plot image
            row = label
            col = class_counters[label]
            if len(self.lesion_type_dict) == 1:
                curr_ax = axes[col]
            else:
                curr_ax = axes[row, col]

            curr_ax.imshow(img)
            class_name = self.idx_to_class[label]
            curr_ax.set_title(f"{class_name}", fontsize=10)
            curr_ax.axis('off')

            # Update counter
            class_counters[label] += 1

            # Check if we have enough examples of this class
            if class_counters[label] >= num_per_class:
                completed_classes.add(label)

            # Check if we're done
            if len(completed_classes) == len(self.lesion_type_dict):
                break

        # Add class labels on the left side
        for i, class_idx in enumerate(range(len(self.lesion_type_dict))):
            class_name = self.idx_to_class[class_idx]
            fig.text(0.01, 0.5 + (0.5 - (i + 0.5) / len(self.lesion_type_dict)) * 0.95,
                     f"{class_name}",
                     fontsize=12,
                     ha='left',
                     va='center')

        plt.tight_layout()
        plt.subplots_adjust(left=0.15, top=0.95)
        plt.savefig(os.path.join(self.results_dir, 'sample_images.png'), dpi=300)
        plt.close()
        print(f"Sample images saved to {os.path.join(self.results_dir, 'sample_images.png')}")

    def visualize_feature_space(self, dataloader, max_samples=300):
        """
        Visualize the feature space using t-SNE and PCA
        """
        # Extract features
        features = []
        labels = []

        self.model.eval()
        count = 0

        # Register hook to get features
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        # Register hook for the layer before final classification
        try:
            hook = self.model.avgpool.register_forward_hook(get_activation('avgpool'))
            feature_layer_name = 'avgpool'
        except AttributeError:
            # For models without avgpool (like some custom models)
            # Try to hook onto the last conv layer or adapt as needed
            hook = self.model.layer4[-1].register_forward_hook(get_activation('layer4'))
            feature_layer_name = 'layer4'

        with torch.no_grad():
            for inputs, targets in dataloader:
                if count >= max_samples:
                    break

                batch_size = inputs.size(0)
                if count + batch_size > max_samples:
                    # Only take what we need
                    inputs = inputs[:max_samples - count]
                    targets = targets[:max_samples - count]

                inputs = inputs.to(self.device)
                # Forward pass
                _ = self.model(inputs)

                # Get features
                batch_features = activation[feature_layer_name].cpu().numpy()
                if len(batch_features.shape) > 2:
                    batch_features = batch_features.reshape(batch_features.shape[0], -1)
                features.append(batch_features)

                # Get labels
                batch_labels = targets.cpu().numpy()
                labels.append(batch_labels)

                count += inputs.size(0)

        # Remove hook
        hook.remove()

        # Concatenate batches
        features = np.vstack(features)
        labels = np.concatenate(labels)

        # First apply PCA (always works regardless of sample size)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)

        # Plot PCA
        plt.figure(figsize=(10, 8))
        for i in range(len(self.lesion_type_dict)):
            indices = labels == i
            if sum(indices) > 0:  # Only plot if we have samples of this class
                plt.scatter(
                    features_pca[indices, 0],
                    features_pca[indices, 1],
                    label=self.idx_to_class[i],
                    alpha=0.7
                )

        plt.title('PCA Visualization of Feature Space', fontsize=14, fontweight='bold')
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(title='Lesion Classes', title_fontsize=12, fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'pca_features.png'), dpi=300)
        plt.close()
        print(f"PCA visualization saved to {os.path.join(self.results_dir, 'pca_features.png')}")

        # Calculate and display variance explained
        var_exp = pca.explained_variance_ratio_
        plt.figure(figsize=(8, 5))
        plt.bar(range(2), var_exp, alpha=0.7)
        plt.ylabel('Variance Explained', fontsize=12)
        plt.xlabel('Principal Components', fontsize=12)
        plt.title('Variance Explained by Principal Components', fontsize=14, fontweight='bold')
        plt.xticks(range(2), [f'PC{i + 1}' for i in range(2)])
        plt.grid(axis='y', alpha=0.3)
        for i, v in enumerate(var_exp):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'pca_variance_explained.png'), dpi=300)
        plt.close()
        print(f"PCA variance explained plot saved to {os.path.join(self.results_dir, 'pca_variance_explained.png')}")

        # Now try t-SNE if we have enough samples
        if len(features) > 5:  # Only do t-SNE if we have enough samples
            try:
                # Apply t-SNE with appropriate perplexity
                perplexity = min(30, len(features) - 1)  # Ensure perplexity < n_samples
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                features_tsne = tsne.fit_transform(features)

                # Plot t-SNE
                plt.figure(figsize=(10, 8))
                for i in range(len(self.lesion_type_dict)):
                    indices = labels == i
                    if sum(indices) > 0:  # Only plot if we have samples of this class
                        plt.scatter(
                            features_tsne[indices, 0],
                            features_tsne[indices, 1],
                            label=self.idx_to_class[i],
                            alpha=0.7
                        )

                plt.title('t-SNE Visualization of Feature Space', fontsize=14, fontweight='bold')
                plt.xlabel('t-SNE Dimension 1', fontsize=12)
                plt.ylabel('t-SNE Dimension 2', fontsize=12)
                plt.legend(title='Lesion Classes', title_fontsize=12, fontsize=10)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'tsne_features.png'), dpi=300)
                plt.close()
                print(f"t-SNE visualization saved to {os.path.join(self.results_dir, 'tsne_features.png')}")
            except Exception as e:
                print(f"Warning: t-SNE visualization failed: {e}")
                print("Skipping t-SNE visualization due to sample size or error")
        else:
            print("Skipping t-SNE visualization - not enough samples")

    def visualize_augmentations(self, image_path):
        """
        Visualize augmentations applied to a single image
        """
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Define augmentation transforms
        augmentations = [
            ('Original', transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])),
            ('Horizontal Flip', transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor()
            ])),
            ('Vertical Flip', transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor()
            ])),
            ('Rotation', transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(20),
                transforms.ToTensor()
            ])),
            ('Color Jitter', transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor()
            ])),
            ('All Augmentations', transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor()
            ]))
        ]

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Augmentation Techniques', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        # Apply and display each augmentation
        for i, (title, transform) in enumerate(augmentations):
            img_transformed = transform(img)
            img_display = img_transformed.numpy().transpose((1, 2, 0))

            axes[i].imshow(img_display)
            axes[i].set_title(title, fontsize=12)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'augmentations.png'), dpi=300)
        plt.close()
        print(f"Augmentation visualization saved to {os.path.join(self.results_dir, 'augmentations.png')}")

    def visualize_model_architecture(self):
        """
        Visualize model architecture as a diagram
        """
        # Get layers info
        layers = []
        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Sequential):
                for sub_name, sub_module in module.named_children():
                    layers.append(f"{name}.{sub_name}: {type(sub_module).__name__}")
            else:
                layers.append(f"{name}: {type(module).__name__}")

        # Plot model architecture
        plt.figure(figsize=(10, max(8, len(layers) * 0.3)))
        y_positions = range(len(layers))

        # Create horizontal bars for layers
        plt.barh(y_positions, [0.8] * len(layers), left=[0.1] * len(layers), color='lightblue', alpha=0.8)

        # Add layer names
        for i, layer in enumerate(layers):
            plt.text(0.5, i, layer, ha='center', va='center', fontsize=10)

        plt.ylim(-0.5, len(layers) - 0.5)
        plt.xlim(0, 1)
        plt.title('Model Architecture', fontsize=14, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_architecture.png'), dpi=300)
        plt.close()
        print(f"Model architecture visualization saved to {os.path.join(self.results_dir, 'model_architecture.png')}")