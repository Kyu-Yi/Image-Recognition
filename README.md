# Skin Lesion Classification Project


This project implements a deep learning model to classify skin lesions using the HAM10000 dataset. The model uses transfer learning with a ResNet18 architecture to classify dermatoscopic images into seven different categories of skin lesions.

## Setup Instructions

### 1. Install Requirements

First, install all required libraries by running:

```
pip install -r requirements.txt
```

This will install the CPU versions of PyTorch, torchvision, and other necessary libraries like scikit-learn, pandas, matplotlib, etc.

**For GPU support** (recommended for faster processing), install with:
```
pip install numpy pandas matplotlib seaborn scikit-learn pillow tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
(Replace cu118 with your CUDA version if different)

### 2. Dataset Options

You have two options for running this project:

#### Option A: Using the Provided Sample Dataset

A small sample dataset is included in the `sample_dataset` folder for testing purposes. This contains exactly 20 images from each of the 7 classes (140 total images) and allows you to test the code without downloading the full HAM10000 dataset.

#### Option B: Download Full HAM10000 Dataset

To use the full dataset (recommended for training):

1. Download the HAM10000 dataset from Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Extract the downloaded ZIP file to a location on your computer
3. Take note of the path where you extracted the dataset

The extracted folder should contain:
- HAM10000_metadata.csv
- HAM10000_images_part_1 folder
- HAM10000_images_part_2 folder

## Running the Project

### Using the Sample Dataset

To run evaluation with the sample dataset (for quick testing):

```
python main.py --data_dir sample_dataset --mode evaluate --model_path ./output/models/best_model.pth --output_dir ./sample_output
```

This will load the pre-trained model and evaluate it on the sample dataset without retraining. Results will be saved in the `./sample_output/results/` directory.

To run visualizations on the sample dataset:

```
python main.py --data_dir sample_dataset --mode visualize --model_path ./output/models/best_model.pth --output_dir ./sample_output
```

### Creating Your Own Sample Dataset

If you want to recreate the sample dataset:

```
python create_sample_dataset.py
```

This script will:
1. Create the sample_dataset folder structure
2. Randomly select 20 images from each lesion class (using a fixed random seed for reproducibility)
3. Copy these images to the sample_dataset/images folder
4. Create a filtered metadata file with only the selected images

### Using the Full HAM10000 Dataset

#### For Evaluation Only:

To evaluate the pre-trained model on the full dataset:

```
python main.py --data_dir C:\Users\BIGNA\Desktop\ham10000\archive --mode evaluate --model_path ./output/models/best_model.pth
```

Replace the path with the actual path where you extracted the HAM10000 dataset. Results will be saved in the default `./output/results/` directory.

For visualizations on the full dataset:

```
python main.py --data_dir C:\Users\BIGNA\Desktop\ham10000\archive --mode visualize --model_path ./output/models/best_model.pth
```

#### For Training and Evaluation:

To train a new model on the full dataset (could take several hours, GPU recommended):

```
python main.py --data_dir C:\Users\BIGNA\Desktop\ham10000\archive --mode all --num_epochs 15
```

#### For Quick Testing Before Full Training:

To run a quick test with a small subset of the dataset:

```
python main.py --data_dir C:\Users\BIGNA\Desktop\ham10000\archive --mode all --test_run --num_epochs 2
```

This will use only 20 samples per class and train for 2 epochs to verify everything works.

## Project Output

After running with the full dataset, the project will create:

- `./output/models/` folder with trained model weights
- `./output/results/` folder with visualizations:
  - Confusion matrix
  - Sample predictions
  - Feature space visualizations (PCA, t-SNE)
  - Class distribution
  - Per-class accuracy

When running with the sample dataset and the `--output_dir ./sample_output` parameter, results will be saved to:
- `./sample_output/results/` folder with the same types of visualizations

## Troubleshooting

### Directory Structure Issues

If you get file not found errors:
- Check that the HAM10000_metadata.csv file is in the correct location
- Make sure the image folders are properly located
- Verify the path provided to `--data_dir` is correct

### GPU Usage

The code will automatically detect if a CUDA-compatible GPU is available and use it. To check if your GPU is being detected, look for the message "CUDA available: True" in the output when running the program.

If you see "CUDA available: False" but you have a compatible GPU:
1. Make sure you've installed the GPU version of PyTorch as described in the Setup Instructions section
2. Ensure your GPU drivers are up to date
3. Verify that your GPU supports CUDA

If you need to reinstall PyTorch with GPU support:
```
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
(Replace cu118 with your CUDA version if different)

## Contact

If you have any questions or need assistance with the project, please contact the author.