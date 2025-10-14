import os
import argparse
import torch
import pandas as pd
from data_preparation import DataPreparation, get_class_names
from model import SkinLesionModel
from train import Trainer
from evaluate import Evaluator
from visualize import Visualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Skin Lesion Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HAM10000 dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'visualize', 'all'], default='all',
                        help='Mode to run the script in')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model (for eval/vis modes)')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weighted_loss', action='store_true', help='Use weighted loss for class imbalance')
    parser.add_argument('--test_run', action='store_true', help='Run with a small subset of data for testing')
    parser.add_argument('--samples_per_class', type=int, default=20, help='Number of samples per class for test run')

    return parser.parse_args()


def setup_directories(output_dir):
    """Setup output directories"""
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return models_dir, results_dir


def main():
    """Main function"""
    args = parse_args()

    # Setup directories
    models_dir, results_dir = setup_directories(args.output_dir)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU - training will be slower")
    print(f"Using device: {device}")

    # Data preparation
    print("Preparing data...")
    data_prep = DataPreparation(args.data_dir)

    # Use subset for testing if specified
    if args.test_run:
        print(f"Running in test mode with {args.samples_per_class} samples per class")
        train_loader, val_loader, test_loader, class_weights, lesion_type_dict = data_prep.load_and_prepare_data(
            samples_per_class=args.samples_per_class
        )
    else:
        train_loader, val_loader, test_loader, class_weights, lesion_type_dict = data_prep.load_and_prepare_data()

    # Get class names
    class_names = get_class_names(lesion_type_dict)

    # Create model
    model_handler = SkinLesionModel(num_classes=len(lesion_type_dict), use_pretrained=True)

    # Training mode
    if args.mode in ['train', 'all']:
        print("Creating model...")
        model = model_handler.create_model()

        # Define loss function
        if args.weighted_loss:
            criterion = model_handler.get_criterion(class_weights)
        else:
            criterion = model_handler.get_criterion()

        # Define optimizer and scheduler
        optimizer = model_handler.get_optimizer(model, lr=args.learning_rate)
        scheduler = model_handler.get_scheduler(optimizer)

        # Create trainer
        trainer = Trainer(model, criterion, optimizer, scheduler, device, save_dir=models_dir)

        # Train model
        print(f"Starting training for {args.num_epochs} epochs...")
        trainer.train(train_loader, val_loader, num_epochs=args.num_epochs)

        # Save the final model
        final_model_path = os.path.join(models_dir, 'final_model.pth')
        model_handler.save_model(model, final_model_path)

        # Use the trained model for evaluation
        best_model_path = os.path.join(models_dir, 'best_model.pth')
    else:
        # Load pre-trained model
        if args.model_path:
            best_model_path = args.model_path
        else:
            best_model_path = os.path.join(models_dir, 'best_model.pth')

        if not os.path.exists(best_model_path):
            print(f"Model not found at {best_model_path}. Please train the model first or provide a valid model path.")
            return

        print(f"Loading model from {best_model_path}...")
        model = model_handler.load_model(best_model_path)

    # Evaluation mode
    if args.mode in ['evaluate', 'all']:
        print("Evaluating model...")
        evaluator = Evaluator(model, device, lesion_type_dict, results_dir=results_dir)
        metrics = evaluator.evaluate(test_loader)

        # Visualize predictions
        print("Generating prediction visualizations...")
        evaluator.visualize_predictions(test_loader)

        # Plot per-class metrics
        evaluator.plot_per_class_metrics(metrics)

    # Visualization mode
    if args.mode in ['visualize', 'all']:
        print("Generating visualizations...")
        visualizer = Visualizer(model, device, lesion_type_dict, results_dir=results_dir)

        # Load metadata for visualization
        metadata = pd.read_csv(os.path.join(args.data_dir, 'HAM10000_metadata.csv'))
        metadata['label'] = metadata['dx'].map(lesion_type_dict)

        # Visualize class distribution
        visualizer.visualize_class_distribution(metadata)

        # Visualize sample images
        from data_preparation import SkinLesionDataset
        test_dataset = test_loader.dataset
        visualizer.visualize_sample_images(test_dataset)

        # Visualize feature space
        visualizer.visualize_feature_space(test_loader, max_samples=300)

        # Find a sample image for augmentation visualization
        for root, _, files in os.walk(os.path.join(args.data_dir, 'images')):
            for file in files:
                if file.endswith('.jpg'):
                    sample_image_path = os.path.join(root, file)
                    visualizer.visualize_augmentations(sample_image_path)
                    break
            if 'sample_image_path' in locals():
                break

    print("Done!")


if __name__ == '__main__':
    main()