import torch
import torch.nn as nn
import torchvision.models as models


class SkinLesionModel:
    """Class for creating and managing the skin lesion classification model"""

    def __init__(self, num_classes=7, use_pretrained=True):
        """
        Initialize the model
        """
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_model(self):
        """
        Create and return the ResNet18 model for skin lesion classification
        """
        # Load pre-trained ResNet18 model
        if self.use_pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)

        # Freeze early layers
        for param in list(model.parameters())[:-8]:
            param.requires_grad = False

        # Modify the final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)

        # Move model to appropriate device
        model = model.to(self.device)

        print(f"Created ResNet18 model with {self.num_classes} output classes")
        print(f"Using device: {self.device}")

        return model

    def get_optimizer(self, model, lr=0.001, weight_decay=1e-4):
        """
        Create and return an Adam optimizer for the model
        """
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_criterion(self, class_weights=None):
        """
        Create and return the loss function
        """
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using weighted CrossEntropyLoss")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")

        return criterion

    def get_scheduler(self, optimizer, mode='min', factor=0.5, patience=3):
        """
        Create and return a learning rate scheduler
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )

    def save_model(self, model, path):
        """
        Save the model to disk
        """
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load a saved model from storage
        """
        model = self.create_model()
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        print(f"Model loaded from {path}")
        return model