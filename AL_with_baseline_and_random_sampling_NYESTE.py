import torch
import torchvision
import numpy as np
from copy import deepcopy
import argparse
from tqdm import tqdm  # For displaying progress bars during training
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE  # For visualizing decision boundaries

# Check if running in a Jupyter notebook environment
if 'ipykernel' in sys.modules:
    debug = False
else:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action='store_true', help="Debug mode")
    args = ap.parse_args()
    debug = args.debug

torch.manual_seed(0)

### Hyperparameters
val_split = 0.1             # Fraction of data to be used for validation
unlabelled_size = 0.99      # Fraction of the training data to be initially unlabelled
lr = 0.0005                 # Learning rate for the optimizer
batch_size = 64             # Batch size for training and validation
num_epochs = 100            # Number of epochs for each training phase
label_iterations = 2        # Number of iterations of active learning or random sampling
top_frac = 0.01             # Fraction of uncertain samples selected in each active learning iteration

### Setting up the MNIST dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

if debug:
    train_dataset.data = train_dataset.data[:1000]
    train_dataset.targets = train_dataset.targets[:1000]
    torch.set_num_threads(4)

val_dataset = deepcopy(train_dataset)
train_size = int((1 - val_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
indexes = torch.randperm(len(train_dataset)).tolist()

indexes_val = indexes[train_size:]
val_dataset.targets = val_dataset.targets[indexes_val]
val_dataset.data = val_dataset.data[indexes_val]
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False)

indexes_train = indexes[:train_size]
train_dataset.targets = train_dataset.targets[indexes_train]
train_dataset.data = train_dataset.data[indexes_train]

unlabelled_size = int(unlabelled_size * len(train_dataset))
indexes_train = torch.randperm(len(train_dataset)).tolist()
unlabelled_dataset = deepcopy(train_dataset)
unlabelled_dataset.targets = unlabelled_dataset.targets[indexes_train[:unlabelled_size]]
unlabelled_dataset.data = unlabelled_dataset.data[indexes_train[:unlabelled_size]]
train_dataset.targets = train_dataset.targets[indexes_train[unlabelled_size:]]
train_dataset.data = train_dataset.data[indexes_train[unlabelled_size:]]
start_train_dataset = deepcopy(train_dataset)
start_unlabelled_dataset = deepcopy(unlabelled_dataset)

print(f"Initial labeled images in training set: {len(train_dataset)}")
print(f"Initial unlabeled images in unlabeled set: {len(unlabelled_dataset)}")


# Function to transfer data from the unlabelled dataset to the labelled dataset
def transfer_unlabelled_to_labelled(unlabelled_dataset, train_dataset, indexes, uncertainties=None):
    # If uncertainties are provided, sort indexes by uncertainty
    if uncertainties is not None:
        # Sort indexes by uncertainty, selecting the top uncertain samples (up to 25)
        sorted_indexes = [idx for _, idx in sorted(zip(uncertainties, indexes), reverse=True)]
        selected_indexes = sorted_indexes[:25]  # Top uncertain samples, max 25
    else:
        # If no uncertainties are provided (random sampling), use indexes directly
        selected_indexes = indexes[:25] if len(indexes) > 25 else indexes
    # Convert list of selected indices to a boolean mask for efficient filtering
    mask = torch.tensor([i in selected_indexes for i in range(len(unlabelled_dataset.targets))])
    # Save selected images and labels before modifying unlabelled_dataset
    selected_images = unlabelled_dataset.data[mask]
    selected_labels = unlabelled_dataset.targets[mask]
    # Add selected unlabelled samples to the labelled training dataset
    train_dataset.targets = torch.cat([train_dataset.targets, selected_labels])
    train_dataset.data = torch.cat([train_dataset.data, selected_images])
    # Remove the added samples from the unlabelled dataset
    unlabelled_dataset.targets = unlabelled_dataset.targets[~mask]
    unlabelled_dataset.data = unlabelled_dataset.data[~mask]
    # Display added images for "eye test"
    display_added_images(selected_images, selected_labels)
    return train_dataset, unlabelled_dataset


# Function to display added images
def display_added_images(images, labels):
    # Convert images to float and normalize before displaying
    images = images.float() / 255.0  # Convert to Float and scale to [0, 1]
    num_images = min(25, len(images))  # Display up to 25 images
    grid_size = int(np.ceil(np.sqrt(num_images)))  # Define grid size based on number of images
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 10))
    fig.suptitle("Top Uncertain Newly Added Images in Active Learning")
    # Loop over grid and display each image if available
    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < num_images:
            image, label = images[i], labels[i]
            ax.imshow(image.numpy().squeeze(), cmap='gray')
            ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def validate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Function to visualize decision boundaries using t-SNE with a subset of data
def visualize_decision_boundaries(model, unlabelled_dataset, device, sample_size=500):
    model.eval()
    embeddings = []
    labels = []
    
    # Use a subset of unlabelled data if the dataset is large
    subset_indices = torch.randperm(len(unlabelled_dataset))[:sample_size]
    subset_data = unlabelled_dataset.data[subset_indices]
    subset_targets = unlabelled_dataset.targets[subset_indices]

    with torch.no_grad():
        for image, label in zip(subset_data, subset_targets):
            image = image.float().unsqueeze(0).unsqueeze(0).to(device) / 255.0  # Normalize and add batch dimension
            embedding = model(image).cpu().numpy()
            embeddings.append(embedding)
            labels.append(label)

    # Run t-SNE for dimensionality reduction
    embeddings = np.array(embeddings).reshape(len(subset_data), -1)
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(embeddings)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=range(10), title="Digits")
    plt.title("t-SNE of Decision Boundaries in Unlabelled Dataset")
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(weights=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model_parameters = deepcopy(model.state_dict())
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, val_interval=1, description=""):
    accuracies = []
    print(f"Starting training: {description}")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % val_interval == 0:
            val_accuracy = validate_model(model, val_loader, device)
            accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}, {description} Accuracy: {val_accuracy:.2f}%')
    return accuracies

def label_iteration(model, train_dataset, unlabelled_dataset, device, top_frac=top_frac):
    model.eval()
    predictions = []
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # Collect predictions for uncertainty
    with torch.no_grad():
        for images, _ in tqdm(unlabelled_loader):
            images = images.to(device)
            outputs = model(images).softmax(dim=1)
            predictions.extend(outputs.detach().cpu().numpy())
    predictions = torch.tensor(predictions)
    top_percent = int(top_frac * len(predictions))
    # Calculate uncertainties as the top confidence of each prediction
    uncertainties, top_indices = predictions.max(-1)[0].topk(top_percent, largest=False)
    print(f"Adding {len(top_indices)} images to training set for Active Learning.")
    # Pass both top_indices and uncertainties to transfer_unlabelled_to_labelled
    train_dataset, unlabelled_dataset = transfer_unlabelled_to_labelled(unlabelled_dataset, train_dataset, top_indices, uncertainties)
    visualize_decision_boundaries(model, unlabelled_dataset, device)
    return train_dataset, unlabelled_dataset

def random_sampling_iteration(unlabelled_dataset, sample_size):
    random_indices = torch.randperm(len(unlabelled_dataset))[:sample_size]
    print(f"Adding {len(random_indices)} images to training set for Random Sampling.")
    return random_indices


# Active Learning Loop
datapoint_list = []  # To store the number of labeled datapoints for active learning
accuracy_list = []  # To store accuracy after each iteration of active learning
for i in range(label_iterations):
    description = f"Active Learning Iteration {i + 1}"
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(model_parameters)  # Reset model parameters before training
    # Train model and save accuracies
    accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, val_interval=10, description=description)
    datapoint_list.append(len(train_dataset))
    accuracy_list.append(accuracies)
    if i < label_iterations - 1:
        train_dataset, unlabelled_dataset = label_iteration(model, train_dataset, unlabelled_dataset, device, top_frac=top_frac)
        print(f"After AL iteration {i + 1}:")
        print(f" - Labeled images in training set: {len(train_dataset)}")
        print(f" - Remaining unlabeled images in unlabeled set: {len(unlabelled_dataset)}")
        visualize_decision_boundaries(model, unlabelled_dataset, device)  # Visualize decision boundaries after adding new samples

# Baseline Model
print("\nRunning Baseline Model")
n_datapoints = len(train_dataset) - len(start_train_dataset)
model.load_state_dict(model_parameters)  # Reset model parameters for baseline
train_dataset.data = torch.cat([start_train_dataset.data, start_unlabelled_dataset.data[:n_datapoints]])
train_dataset.targets = torch.cat([start_train_dataset.targets, start_unlabelled_dataset.targets[:n_datapoints]])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print("Baseline model training:")
print(f" - Labeled images in training set: {len(train_dataset)}")
print(f" - Unlabeled images in unlabeled set: {len(unlabelled_dataset)}")
baseline_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, val_interval=10, description="Baseline Model")

# Random Sampling Loop for Comparison
random_datapoint_list = []  # Stores number of labeled data points over iterations for random sampling
random_accuracy_list = []  # Stores accuracy at each iteration for random sampling
random_train_dataset = deepcopy(start_train_dataset)
random_unlabelled_dataset = deepcopy(start_unlabelled_dataset)
for i in range(label_iterations):
    description = f"Random Sampling Iteration {i + 1}"
    train_loader = torch.utils.data.DataLoader(random_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(model_parameters)  # Reset model to initial state for fair comparison
    # Train model and store accuracies for random sampling method
    accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, val_interval=10, description=description)
    random_datapoint_list.append(len(random_train_dataset))
    random_accuracy_list.append(accuracies)
    
    # Perform random sampling to add new samples to the labeled dataset
    if i < label_iterations - 1:
        random_indices = random_sampling_iteration(random_unlabelled_dataset, int(top_frac * len(random_unlabelled_dataset)))
        random_train_dataset, random_unlabelled_dataset = transfer_unlabelled_to_labelled(random_unlabelled_dataset, random_train_dataset, random_indices)
        print(f"After Random Sampling iteration {i + 1}:")
        print(f" - Labeled images in training set: {len(random_train_dataset)}")
        print(f" - Remaining unlabeled images in unlabeled set: {len(random_unlabelled_dataset)}")
        visualize_decision_boundaries(model, random_unlabelled_dataset, device)  # Visualize decision boundaries after adding random samples

# Plotting the accuracy results for all three methods
datapoints = np.array(datapoint_list)  # Data points for active learning
accuracies = np.array(accuracy_list).max(-1)  # Max accuracy per iteration for active learning
random_datapoints = np.array(random_datapoint_list)  # Data points for random sampling
random_accuracies = np.array(random_accuracy_list).max(-1)  # Max accuracy per iteration for random sampling

# Create a plot showing Active Learning, Random Sampling, and Baseline accuracies
plt.figure(figsize=(10, 5))
plt.plot(datapoints, accuracies, label='Active Learning Accuracy')
plt.plot(random_datapoints, random_accuracies, label='Random Sampling Accuracy', linestyle='--')
plt.hlines(max(baseline_accuracy), min(datapoints), max(datapoints), label='Baseline Accuracy', color='red')
plt.xlabel('Datapoints')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()