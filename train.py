import os
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    """Class for training and validating the model"""

    def __init__(self, model, criterion, optimizer, scheduler, device, save_dir='./models'):
        """
        Initialize the trainer
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Use tqdm for progress bar
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # Backward pass + optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """
        Validate the model
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs=10, early_stopping_patience=5):
        """
        Train the model

        Returns:
            torch.nn.Module: Best trained model
        """
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()

        no_improve_counter = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Record statistics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(epoch)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # Early stopping
            if no_improve_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        time_elapsed = time.time() - start_time
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        # Load best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        # Plot training history
        self.plot_training_history()

        return self.model

    def save_model(self, epoch):
        """
        Save the current model
        """
        # Save best model
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))

        # Also save with epoch number
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, f'model_epoch_{epoch + 1}.pth')
        )

        print(f"Model saved with validation accuracy: {self.best_val_acc:.4f}")

    def plot_training_history(self):
        """Plot and save training and validation history"""
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()
        print(f"Training history saved to {os.path.join(self.save_dir, 'training_history.png')}")