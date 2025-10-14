import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from tqdm import tqdm


class Evaluator:
    """Class for evaluating the trained model"""

    def __init__(self, model, device, lesion_type_dict, results_dir='./results'):
        """
        Initialize the evaluator
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

    def evaluate(self, test_loader):
        """
        Evaluate the model on test data
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics with zero_division=0 parameter
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Generate and print classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
        print("\nClassification Report:")
        print(report)

        # Plot confusion matrices (raw and normalized)
        self.plot_confusion_matrix(cm)

        # Save results to CSV
        report_df = pd.DataFrame({
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1]
        })
        report_df.to_csv(os.path.join(self.results_dir, 'metrics.csv'), index=False)

        # Save detailed report
        with open(os.path.join(self.results_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

        return metrics

    def plot_confusion_matrix(self, cm):
        """
        Plot and save raw and normalized confusion matrices
        """
        # 1. Raw counts confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix_raw.png'), dpi=300)
        plt.close()
        print(f"Raw confusion matrix saved to {os.path.join(self.results_dir, 'confusion_matrix_raw.png')}")

        # 2. Row-normalized confusion matrix (recall perspective)
        row_sums = cm.sum(axis=1)
        cm_row_normalized = np.zeros_like(cm, dtype=float)
        for i in range(cm.shape[0]):
            if row_sums[i] > 0:
                cm_row_normalized[i, :] = cm[i, :] / row_sums[i]

        # Convert to percentages (multiply by 100)
        cm_row_normalized_pct = cm_row_normalized * 100

        # Create a custom formatter to add % symbol
        def percentage_formatter(x, pos):
            return f'{x:.0f}%'

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            cm_row_normalized_pct, annot=True, fmt='.0f', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            vmin=0, vmax=100
        )

        # Add percentage symbol to each annotation
        # Get the annotations from the heatmap
        for text in ax.texts:
            value = int(float(text.get_text()))
            text.set_text(f"{value}%")

        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix (Normalized by Row - Recall Perspective)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix_norm.png'), dpi=300)
        plt.close()
        print(f"Normalized confusion matrix saved to {os.path.join(self.results_dir, 'confusion_matrix_norm.png')}")

    def visualize_predictions(self, test_loader, num_images=12):
        """
        Visualize model predictions on sample images
        """
        self.model.eval()

        # Get sample images and predictions
        all_images = []
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                # Store batch results
                for i in range(inputs.size(0)):
                    all_images.append(inputs[i].cpu())
                    all_preds.append(preds[i].item())
                    all_labels.append(labels[i].item())
                    all_probs.append(probs[i].cpu().numpy())

                if len(all_images) >= num_images:
                    break

        # Limit to specified number of images
        all_images = all_images[:num_images]
        all_preds = all_preds[:num_images]
        all_labels = all_labels[:num_images]
        all_probs = all_probs[:num_images]

        # Plot images with predictions
        rows = int(np.ceil(num_images / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
        fig.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold', y=0.98)
        axes = axes.flatten()

        for i, (img, pred, label, prob) in enumerate(zip(all_images, all_preds, all_labels, all_probs)):
            if i >= num_images:
                break

            # Convert image for display
            img = img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            # Plot image
            axes[i].imshow(img)

            # Set title color based on correctness
            title_color = 'green' if pred == label else 'red'

            # Create title with class names and probability
            pred_class = self.idx_to_class[pred]
            true_class = self.idx_to_class[label]
            prob_value = prob[pred] * 100

            title = f"Pred: {pred_class} ({prob_value:.1f}%)\nTrue: {true_class}"
            axes[i].set_title(title, color=title_color, fontsize=11)
            axes[i].axis('off')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(os.path.join(self.results_dir, 'sample_predictions.png'), dpi=300)
        plt.close()
        print(f"Sample predictions saved to {os.path.join(self.results_dir, 'sample_predictions.png')}")

    def plot_per_class_metrics(self, metrics):
        """
        Plot per-class performance metrics
        """
        cm = metrics['confusion_matrix']

        # Calculate per-class metrics
        class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        # Handle NaN values if any class has no samples
        class_accuracy = np.nan_to_num(class_accuracy)

        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Class': self.class_names,
            'Accuracy': class_accuracy
        })

        # Plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Class', y='Accuracy', data=df)
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.0)  # Set y-axis from 0 to 1

        # Add value labels
        for i, p in enumerate(ax.patches):
            ax.annotate(
                f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points'
            )

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'per_class_accuracy.png'), dpi=300)
        plt.close()
        print(f"Per-class accuracy plot saved to {os.path.join(self.results_dir, 'per_class_accuracy.png')}")