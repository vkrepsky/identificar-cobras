import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from pathlib import Path
import time


# Define the CNN models (need to match the architecture of saved models)
class SnakeCNN(nn.Module):
    """
    Model CNN based on the paper architecture
    'A CNN Based Model for Venomous and Non-venomous Snake Classification'
    """

    def __init__(self, num_classes):
        super(SnakeCNN, self).__init__()

        # Following the paper: 3 conv layers with 16, 32, and 64 filters
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate input size for the first fully connected layer
        # For 224x224 input image, after 3 max pooling operations
        # spatial dimensions become 28x28
        fc_input_size = 64 * 28 * 28

        # Classifier following the paper's approach
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Paper uses 0.2 dropout
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SnakeCNNDeeper(nn.Module):
    """
    Extended version of the paper architecture with more regularization
    to help prevent overfitting for multi-class species classification.
    """

    def __init__(self, num_classes):
        super(SnakeCNNDeeper, self).__init__()

        # Feature extractor - deeper than the original paper but with similar structure
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Added batch normalization for better training stability
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth block (additional)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # For 224x224 input image, after 4 max pooling operations
        # spatial dimensions become 14x14
        fc_input_size = 128 * 14 * 14

        # Classifier with stronger regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SnakeCNNBalanced(nn.Module):
    """
    Modelo CNN balanceado para classificação de serpentes
    - Mais expressivo que o modelo paper
    - Menos complexo que o deeper model
    - Regularização moderada para evitar overfitting
    """

    def __init__(self, num_classes):
        super(SnakeCNNBalanced, self).__init__()

        # Feature extractor - balanço entre paper (3 camadas) e deeper (4 camadas)
        self.features = nn.Sequential(
            # Primeiro bloco convolucional
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Adiciona normalização, mas mantém estrutura simples
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Segundo bloco convolucional
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Terceiro bloco convolucional
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Camada adicional moderada - mais leve que a do deeper
            nn.Conv2d(64, 96, kernel_size=3, padding=1),  # 96 filtros em vez de 128
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Para uma imagem de entrada 224x224, após 4 operações de max pooling
        # as dimensões espaciais se tornam 14x14
        fc_input_size = 96 * 14 * 14  # Menos filtros que o deeper

        # Classificador com regularização moderada
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 256),  # Menor que o deeper (512)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),  # Intermediário entre paper (0.2) e deeper (0.3)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path, model_class):
    """
    Load a trained model from a saved checkpoint file

    Args:
        model_path: Path to the saved model file
        model_class: Class of the model to be loaded (SnakeCNN or SnakeCNNDeeper)

    Returns:
        model: Loaded model
        class_names: List of class names
        class_to_idx: Dictionary mapping class names to indices
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get class information
    class_names = checkpoint.get('class_names', [])
    class_to_idx = checkpoint.get('class_to_idx', {})

    # Create a new model instance
    model = model_class(len(class_names))

    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    return model, class_names, class_to_idx


def get_image_transform():
    """
    Returns the transformation to apply to test images
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_test_images(test_dir):
    """
    Collects all valid image files from the test directory
    and their corresponding label if the directory structure
    matches the training structure (class_name/image.jpg)

    Args:
        test_dir: Directory containing test images

    Returns:
        image_paths: List of image paths
        true_labels: List of true labels (if available)
    """
    image_paths = []
    true_labels = []
    has_class_structure = False

    # Check if test_dir contains subdirectories (class structure)
    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    if subdirs:
        has_class_structure = True
        print(f"Detected class structure in test directory with {len(subdirs)} classes")

        for class_name in subdirs:
            class_dir = os.path.join(test_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    true_labels.append(class_name)
    else:
        # Flat structure - no class information
        print("Flat structure detected in test directory (no class information)")
        for img_name in os.listdir(test_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(test_dir, img_name))

    return image_paths, true_labels, has_class_structure


def predict_image(model, image_path, transform, device, top_k=3):
    """
    Make a prediction for a single image

    Args:
        model: Trained model
        image_path: Path to the image
        transform: Image transformation to apply
        device: Device to use for inference (CPU/CUDA)
        top_k: Number of top predictions to return

    Returns:
        top_probs: Top k probabilities
        top_classes: Corresponding class indices
    """
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    # Get top k predictions
    top_probs, top_classes = torch.topk(probabilities, top_k)

    return top_probs.cpu().numpy(), top_classes.cpu().numpy()


def evaluate_model(model, image_paths, true_labels, class_names, class_to_idx, transform, device, model_name):
    """
    Evaluate model performance on test images

    Args:
        model: Trained model
        image_paths: List of image paths
        true_labels: List of true labels (if available)
        class_names: List of class names
        class_to_idx: Dictionary mapping class names to indices
        transform: Image transformation to apply
        device: Device to use for inference
        model_name: Name of the model for reporting

    Returns:
        results: Dictionary with evaluation results
    """
    # Initialize results
    results = {
        'model_name': model_name,
        'predictions': [],
        'timings': []
    }

    # Create reverse mapping from index to class name
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # Track predictions for confusion matrix if we have true labels
    if true_labels:
        y_true = []
        y_pred = []

        # Create mapping from class name to model index
        # This is needed because test labels may not match training labels exactly
        class_name_to_idx = {name: i for i, name in enumerate(class_names)}

    # Process each image
    print(f"\nEvaluating {model_name} on {len(image_paths)} test images...")

    for i, image_path in enumerate(image_paths):
        # Time the prediction
        start_time = time.time()
        top_probs, top_classes = predict_image(model, image_path, transform, device)
        inference_time = time.time() - start_time

        # Map indices to class names
        top_class_names = [class_names[idx] for idx in top_classes]

        # Get the image filename
        image_name = os.path.basename(image_path)

        # Add to results
        results['predictions'].append({
            'image_path': image_path,
            'image_name': image_name,
            'top_class': top_class_names[0],
            'top_probability': float(top_probs[0]),
            'top_k_classes': top_class_names,
            'top_k_probabilities': [float(p) for p in top_probs]
        })

        results['timings'].append(inference_time)

        # Update confusion matrix data if we have true labels
        if true_labels:
            true_label = true_labels[i]
            if true_label in class_name_to_idx:
                y_true.append(class_name_to_idx[true_label])
                y_pred.append(top_classes[0])
            else:
                print(f"Warning: True label '{true_label}' not found in model's class names")

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)} images...")

    # Calculate metrics if we have true labels
    if true_labels and len(y_true) > 0:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)
        results['accuracy'] = float(accuracy)

        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=[class_names[i] for i in range(len(class_names))],
                                       output_dict=True)
        results['classification_report'] = report

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()

    # Calculate average inference time
    results['avg_inference_time'] = sum(results['timings']) / len(results['timings'])

    return results


def compare_models_performance(paper_results, deeper_results, output_dir):
    """
    Compare the performance of both models and visualize results

    Args:
        paper_results: Results from the paper model
        deeper_results: Results from the deeper model
        output_dir: Directory to save comparison results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compare accuracies if available
    if 'accuracy' in paper_results and 'accuracy' in deeper_results:
        paper_acc = paper_results['accuracy']
        deeper_acc = deeper_results['accuracy']

        print("\n=== ACCURACY COMPARISON ===")
        print(f"Paper model: {paper_acc * 100:.2f}%")
        print(f"Deeper model: {deeper_acc * 100:.2f}%")
        print(
            f"Difference: {(deeper_acc - paper_acc) * 100:.2f}% {'(Deeper is better)' if deeper_acc > paper_acc else '(Paper is better)'}")

        # Plot accuracy comparison
        plt.figure(figsize=(8, 6))
        models = ['Paper Model', 'Deeper Model']
        accuracies = [paper_acc * 100, deeper_acc * 100]
        plt.bar(models, accuracies, color=['blue', 'orange'])
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy Comparison')
        plt.ylim(0, 100)

        # Add accuracy values on top of the bars
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')

        # Save the plot
        acc_plot_path = os.path.join(output_dir, f'accuracy_comparison_{timestamp}.png')
        plt.savefig(acc_plot_path)
        plt.close()
        print(f"Accuracy comparison plot saved to: {acc_plot_path}")

    # Compare inference times
    paper_time = paper_results['avg_inference_time']
    deeper_time = deeper_results['avg_inference_time']

    print("\n=== INFERENCE TIME COMPARISON ===")
    print(f"Paper model: {paper_time * 1000:.2f} ms per image")
    print(f"Deeper model: {deeper_time * 1000:.2f} ms per image")
    print(
        f"Difference: {(deeper_time - paper_time) * 1000:.2f} ms {'(Paper is faster)' if paper_time < deeper_time else '(Deeper is faster)'}")

    # Plot inference time comparison
    plt.figure(figsize=(8, 6))
    models = ['Paper Model', 'Deeper Model']
    times = [paper_time * 1000, deeper_time * 1000]  # Convert to milliseconds
    plt.bar(models, times, color=['blue', 'orange'])
    plt.ylabel('Inference Time (ms)')
    plt.title('Model Inference Time Comparison')

    # Add time values on top of the bars
    for i, t in enumerate(times):
        plt.text(i, t + 0.5, f'{t:.2f} ms', ha='center')

    # Save the plot
    time_plot_path = os.path.join(output_dir, f'inference_time_comparison_{timestamp}.png')
    plt.savefig(time_plot_path)
    plt.close()
    print(f"Inference time comparison plot saved to: {time_plot_path}")

    # Compare per-class performance if available
    if 'classification_report' in paper_results and 'classification_report' in deeper_results:
        paper_report = paper_results['classification_report']
        deeper_report = deeper_results['classification_report']

        # Extract per-class F1 scores
        paper_f1 = {cls: paper_report[cls]['f1-score'] for cls in paper_report if
                    cls not in ['accuracy', 'macro avg', 'weighted avg']}
        deeper_f1 = {cls: deeper_report[cls]['f1-score'] for cls in deeper_report if
                     cls not in ['accuracy', 'macro avg', 'weighted avg']}

        # Combine data for plotting
        combined_data = []
        for cls in set(paper_f1.keys()) | set(deeper_f1.keys()):
            combined_data.append({
                'Class': cls,
                'Paper Model': paper_f1.get(cls, 0),
                'Deeper Model': deeper_f1.get(cls, 0)
            })

        df = pd.DataFrame(combined_data)

        # Sort by difference to highlight where models perform most differently
        df['Difference'] = df['Deeper Model'] - df['Paper Model']
        df = df.sort_values('Difference', ascending=False)

        # Plot top 10 classes with biggest differences
        plt.figure(figsize=(12, 8))
        top_n = min(10, len(df))
        df_top = df.head(top_n)

        # Plot
        x = np.arange(len(df_top))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width / 2, df_top['Paper Model'], width, label='Paper Model')
        rects2 = ax.bar(x + width / 2, df_top['Deeper Model'], width, label='Deeper Model')

        # Add labels and legend
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Comparison by Class (Top differences)')
        ax.set_xticks(x)
        ax.set_xticklabels(df_top['Class'], rotation=45, ha='right')
        ax.legend()

        fig.tight_layout()

        # Save the plot
        f1_plot_path = os.path.join(output_dir, f'f1_score_comparison_{timestamp}.png')
        plt.savefig(f1_plot_path)
        plt.close()
        print(f"F1 score comparison plot saved to: {f1_plot_path}")

        # Save detailed class performance to CSV
        csv_path = os.path.join(output_dir, f'class_performance_comparison_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Detailed class performance comparison saved to: {csv_path}")

    # Save full results to JSON
    results_data = {
        'timestamp': timestamp,
        'paper_model': paper_results,
        'deeper_model': deeper_results,
        'comparison': {
            'paper_accuracy': paper_results.get('accuracy', None),
            'deeper_accuracy': deeper_results.get('accuracy', None),
            'accuracy_difference': None if 'accuracy' not in paper_results or 'accuracy' not in deeper_results else
            deeper_results['accuracy'] - paper_results['accuracy'],
            'paper_inference_time': paper_time,
            'deeper_inference_time': deeper_time,
            'inference_time_difference': deeper_time - paper_time
        }
    }

    # Remove large arrays from results to keep JSON file manageable
    if 'confusion_matrix' in results_data['paper_model']:
        results_data['paper_model']['confusion_matrix'] = 'saved_separately'
    if 'confusion_matrix' in results_data['deeper_model']:
        results_data['deeper_model']['confusion_matrix'] = 'saved_separately'

    # Save to JSON
    results_path = os.path.join(output_dir, f'model_comparison_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"Full comparison results saved to: {results_path}")

    # Create a conclusion on which model might be overfitting
    print("\n=== OVERFITTING ANALYSIS ===")
    if 'accuracy' in paper_results and 'accuracy' in deeper_results:
        # Compare test accuracy with validation accuracy (if available)
        # Since we don't have validation accuracies here, we'll just compare the two models
        if deeper_acc > paper_acc:
            print(
                "The deeper model performs better on the test set, suggesting it may not be overfitting despite its increased complexity.")
            print(
                "However, to fully determine overfitting, compare these results with the training and validation accuracies:")
            print(
                "- If training acc >> validation acc >> test acc: Both models likely overfit, with deeper model overfitting less")
            print("- If training acc ≈ validation acc ≈ test acc: Neither model is overfitting significantly")
        else:
            print(
                "The simpler paper model performs better on the test set, which might indicate the deeper model is overfitting.")
            print("This is common when a more complex model fits training data too closely but fails to generalize.")
            print("Check the training/validation curves from your original training run:")
            print("- If the deeper model had training acc >> validation acc: It's likely overfitting")
    else:
        print("Without labeled test data, it's harder to assess overfitting. Compare the model's behavior:")
        print("- Does the deeper model make more 'confident but wrong' predictions?")
        print("- Is there a large gap between training and validation accuracy in your training history?")

    print("\nTo reduce overfitting, consider:")
    print("1. More aggressive data augmentation")
    print("2. Increasing dropout rates")
    print("3. Adding more regularization (weight decay)")
    print("4. Collecting more training data")

    return results_data


def main():
    # Configuration
    paper_model_path = r"C:\tcc\modelo\output\paper_20250331_134110\models\snake_cnn_paper_paper_20250331_134110_final.pth"
    deeper_model_path = r"C:\tcc\modelo\output\deeper_20250331_103827\models\snake_cnn_deeper_deeper_20250331_103827_final.pth"
    balanced_model_path = r"C:\tcc\modelo\output_balanced\balanced_20250331_183307\models\snake_cnn_balanced_balanced_20250331_183307_final.pth"
    test_dir = r"C:\tcc\modelo\teste_overfitting"
    output_dir = r"C:\tcc\modelo\overfitting_analysis_results"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get image transformation
    transform = get_image_transform()

    # Load models
    print("\n=== Loading Models ===")
    print(f"Paper model: {paper_model_path}")
    paper_model, paper_class_names, paper_class_to_idx = load_model(paper_model_path, SnakeCNN)
    print(f"Loaded paper model with {len(paper_class_names)} classes")

    print(f"Deeper model: {deeper_model_path}")
    deeper_model, deeper_class_names, deeper_class_to_idx = load_model(deeper_model_path, SnakeCNNDeeper)
    print(f"Loaded deeper model with {len(deeper_class_names)} classes")

    print(f"Balanced model: {balanced_model_path}")
    balanced_model, balanced_class_names, balanced_class_to_idx = load_model(balanced_model_path, SnakeCNNBalanced)
    print(f"Loaded deeper model with {len(deeper_class_names)} classes")

    # Get test images
    print(f"\n=== Loading Test Images from {test_dir} ===")
    image_paths, true_labels, has_class_structure = get_test_images(test_dir)
    print(f"Found {len(image_paths)} test images")

    if has_class_structure:
        print(f"Test set includes true labels for {len(true_labels)} images")
    else:
        print("Test set does not have class structure, can only evaluate predictions without accuracy")

    # Evaluate paper model
    paper_results = evaluate_model(
        model=paper_model,
        image_paths=image_paths,
        true_labels=true_labels if has_class_structure else [],
        class_names=paper_class_names,
        class_to_idx=paper_class_to_idx,
        transform=transform,
        device=device,
        model_name="Paper Model"
    )

    # Evaluate deeper model
    deeper_results = evaluate_model(
        model=deeper_model,
        image_paths=image_paths,
        true_labels=true_labels if has_class_structure else [],
        class_names=deeper_class_names,
        class_to_idx=deeper_class_to_idx,
        transform=transform,
        device=device,
        model_name="Deeper Model"
    )

    # Compare model performance
    comparison_results = compare_models_performance(paper_results, deeper_results, output_dir)

    print("\n=== Analysis Complete ===")
    print(f"Results have been saved to: {output_dir}")


if __name__ == "__main__":
    main()