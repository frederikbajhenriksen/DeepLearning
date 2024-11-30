import torch
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm  # For displaying progress bars during training
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # For visualizing decision boundaries
import pandas as pd
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from PIL import Image

def set_name(name):
    def decorator(func):
        func.__name__ = name
        return func
    return decorator

def kmeans_torch(x, k, num_iters=100, tol=1e-4):
    """
    Torch implementation of k-means clustering
    
    Args:
        x: Input tensor of shape (n_samples, n_features)
        k: Number of clusters
        num_iters: Maximum iterations
        tol: Convergence tolerance
    """
    n_samples = x.size(0)

    # Validate k
    k = min(k, n_samples)
    
    # Get target dtype and device
    target_dtype = x.dtype
    device = x.device

    # Randomly initialize centroids
    rand_indices = torch.randperm(n_samples, device=device)[:k]
    centroids = x[rand_indices].clone()
    
    for _ in range(num_iters):
        # Calculate distances
        distances = torch.cdist(x, centroids)
        
        # Assign points to nearest centroid
        labels = torch.argmin(distances, dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids[i] = x[mask].mean(dim=0)
            else:
                new_centroids[i] = centroids[i].clone()
        
        # Check convergence
        if torch.norm(new_centroids - centroids) < tol:
            break
            
        centroids = new_centroids.clone()
    labels = labels.to(dtype=target_dtype)
    return labels, centroids

def get_nn(features, num_neighbors):
    """Calculate nearest neighbors using scikit-learn on CPU."""
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)
    return distances[:, 1:], indices[:, 1:]

def calculate_typicality(features, num_neighbors):
    """Calculate typicality based on the mean distance to nearest neighbors."""
    distances, _ = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    typicality = 1 / (mean_distance + 1e-5)  # Higher density = higher typicality
    return typicality

class ActiveLearning:
    """ Active Learning class for MNIST dataset using PyTorch
    Args:
        dataObj: PyTorch dataset object (MNIST)
        unlabelled_size: Percentage of initial unlabelled data for active learning
        label_iterations: Number of iterations for active learning
        num_epochs: Number of epochs for training
        criterion: Loss function for training
        debug: Debug mode for faster training
        lr: Learning rate for optimizer
        seed: Seed for reproducibility
        val_split: Validation split percentage
        b: Budget for each iteration of active learning
        delta: Delta value for ProbCover
        alpha: Purity value for ProbCover
    """
    MAX_NUM_CLUSTERS = 500  # Example value, adjust as needed
    MIN_CLUSTER_SIZE = 5    # Example value, adjust as needed
    K_NN = 20               # Number of neighbors for density calculation
    def __init__(self, dataObj, 
                 unlabelled_size, 
                 label_iterations, 
                 num_epochs,criterion=torch.nn.CrossEntropyLoss(), 
                 debug=False, 
                 lr=0.0005, 
                 seed=0, 
                 val_split=0.1, 
                 b=25,
                 delta=None,
                 alpha=0.75,
                 quiet=False):
        self.dataObj = deepcopy(dataObj)            # This is the dataset that is passed to the class (MNIST in this case and with val removed)
        self.unlabelled_size = unlabelled_size      # The size of the unlabelled dataset in percentage of the total dataset (minus validation)
        self.label_iterations = label_iterations    # Number of iterations for active learning
        self.debug = debug                          # Debug mode for faster training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
        self.seed = seed                            # Seed for reproducibility
        torch.manual_seed(seed)     # Set seed for reproducibility
        self.val_data = None        # Validation data (NOT INDEX)
        self.val_split = val_split  # Validation split in percentage
        self.uSet = None            # Unlabelled data Indices
        self.lSet = None            # Labelled data Indices

        self.first_uSet = None      # Initial unlabelled data Indices
        self.first_lSet = None      # Initial labelled data Indices

        self.num_epochs = num_epochs # Number of epochs for training

        # We need to check 
        sample_data, _ = self.dataObj[0]
        if isinstance(sample_data, np.ndarray):
            self.input_channels = sample_data.shape[2] if sample_data.ndim == 3 else 1
        elif isinstance(sample_data, torch.Tensor):
            self.input_channels = sample_data.shape[0] if sample_data.ndim == 3 else 1
        elif isinstance(sample_data, Image.Image):
            # Convert PIL Image to tensor to get shape
            sample_data = torchvision.transforms.ToTensor()(sample_data)
            self.input_channels = sample_data.shape[0]
        else:
            raise ValueError("Unsupported data type for dataset")

        if self.debug:
            self.dataObj.data = self.dataObj.data[:1000]
            self.dataObj.targets = self.dataObj.targets[:1000]
            torch.set_num_threads(4)
        
        self.init_data(val_split)
        self.b = min(b, len(self.uSet))
        if isinstance(self.dataObj, torchvision.datasets.MNIST):
            self.k = len(self.dataObj.classes)
        elif isinstance(self.dataObj, torchvision.datasets.CIFAR10):
            self.k = len(self.dataObj.classes)
        else:
            self.k = len(self.dataObj.targets.unique())

        # self.model = torchvision.models.resnet18(weights=False) # den brokker sig over False i stedet for None argument
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.k)
        if self.input_channels != 3:
            # Adjust the first convolutional layer
            self.model.conv1 = torch.nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        self.model_parameters = deepcopy(self.model.state_dict())
        self.model = self.model.to(self.device)

        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.quiet = quiet

        ########### 
        # ProbCover
        ###########
        if delta is None:
            self.approx_delta(alpha)
        else:
            self.delta = delta # TODO: Delta needs to be optimized as described in the paper
            if self.delta <= 0:
                raise ValueError('Delta must be positive')
            if self.delta >= 1:
                print('Warning: Delta is very large, this may lead to a fully connected graph')
        try:
            self.graph_df = self.construct_graph()
        except:
            raise ValueError('Graph construction failed')

        ###########
        # TypiClust
        ###########
        # Perform clustering and select initial samples
        self.centroids, self.labels = self.perform_clustering()

        if self.debug:
            print("Model initializing")
        
    #########
    # General functions for active learning
    #########
    def init_data(self, val_split=0.1):
        """Initialize the data for active learning by splitting into training and validation sets."""
        self.val_data = deepcopy(self.dataObj)
        train_size = int((1 - val_split) * len(self.dataObj))
        indexes = torch.randperm(len(self.dataObj)).tolist()

        val_index = indexes[train_size:]
        train_index = indexes[:train_size]

        if isinstance(self.dataObj.data, np.ndarray):
            # For CIFAR10, keep data as NumPy arrays and targets as lists
            self.val_data.data = self.val_data.data[val_index]
            self.val_data.targets = [self.val_data.targets[i] for i in val_index]

            self.dataObj.data = self.dataObj.data[train_index]
            self.dataObj.targets = [self.dataObj.targets[i] for i in train_index]
        elif isinstance(self.dataObj.data, torch.Tensor):
            # For MNIST, data and targets are tensors
            self.val_data.data = self.val_data.data[val_index]
            self.val_data.targets = self.val_data.targets[val_index]

            self.dataObj.data = self.dataObj.data[train_index]
            self.dataObj.targets = self.dataObj.targets[train_index]
        else:
            raise ValueError("Unsupported data type for dataset")
        # Create labeled/unlabeled splits
        self.unlabelled_size = int(self.unlabelled_size * len(self.dataObj))
        indexes_train = torch.randperm(len(self.dataObj)).tolist()
        self.uSet = indexes_train[:self.unlabelled_size]
        self.lSet = indexes_train[self.unlabelled_size:]

        self.first_uSet = deepcopy(self.uSet)
        self.first_lSet = deepcopy(self.lSet)
 
    def reset_data(self):
        """ Reset the data to the initial state for the next active learning method"""
        self.uSet = deepcopy(self.first_uSet)
        self.lSet = deepcopy(self.first_lSet)

        torch.manual_seed(self.seed)
        # Reset model parameters
        # self.model = torchvision.models.resnet18(weights=False) # den brokker sig over False i stedet for None argument
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.k)
        if self.input_channels != 3:
            # Adjust the first convolutional layer
            self.model.conv1 = torch.nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
        )
        self.model_parameters = deepcopy(self.model.state_dict())
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005) # Reset optimizer

    def val_loader(self):
        """ Return validation data loader for model evaluation
        Returns:
            DataLoader: Validation data loader
        """
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.val_data.transform = transform
        return torch.utils.data.DataLoader(self.val_data, batch_size=1024, shuffle=False)
    
    def train_loader(self):
        """ Return training data loader for active learning iterations
        Returns:
            DataLoader: Training data loader
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # Include other transforms if needed
        ])
        train_data = torch.utils.data.Subset(self.dataObj, self.lSet)
        train_data.dataset.transform = transform

        return torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

    def unlabelled_loader(self, batch_size=64):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # Include other transforms if needed
        ])

        unlabelled_dataset = torch.utils.data.Subset(self.dataObj, self.uSet)
        unlabelled_dataset.dataset.transform = transform

        return torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False)

    def validate_model(self):
        """ Validate the model on the validation set and return accuracy """
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader():
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def train_model(self,val_interval=1, description=""):
        """ Train the model for a specified number of epochs and return validation accuracies
        Args:
            val_interval: Interval for validation checks
            description: Description for the training run
        """
        accuracies = []
        if not self.quiet:
            print(f"Starting training: {description}")
        for epoch in tqdm(range(self.num_epochs), disable=self.quiet):
            self.model.train()
            for images, labels in self.train_loader():
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % val_interval == 0:
                val_accuracy = self.validate_model()
                accuracies.append(val_accuracy)
                if not self.quiet:
                    print(f'Epoch {epoch + 1}, {description} Accuracy: {val_accuracy:.2f}%')
        return accuracies
    
    def visualize_decision_boundaries(self, sample_size=500):
        """ Visualize decision boundaries using t-SNE for dimensionality reduction 
        Args:
            sample_size: Number of samples to use for visualization
        """
        self.model.eval()
        embeddings = []
        labels = []
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # Add any other necessary transforms (e.g., normalization)
        ])

        # Use a subset of unlabelled data if the dataset is large
        subset_indices = torch.randperm(len(self.uSet))[:sample_size]
        subset = torch.utils.data.Subset(self.dataObj, subset_indices)
        subset.dataset.transform = transform
        subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for image, label in subset_loader:
                # Get the embedding
                image = image.to(self.device)
                
                embedding = self.model(image).cpu()
                embeddings.append(embedding)
                labels.append(label)
        
        # Process on CPU
        embeddings = torch.cat(embeddings, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        # Prepare embeddings for t-SNE
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(embeddings)

        # Plot the results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
        plt.legend(handles=scatter.legend_elements()[0], labels=range(self.k), title="Classes")
        plt.title("t-SNE of Decision Boundaries in Unlabelled Dataset")
        plt.savefig('tsne_decision_boundaries.png', dpi=300)
        plt.close()

    def display_added_images(self, images, labels):
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
                #ax.imshow(image.numpy().squeeze(), cmap='gray')
                ax.imshow(image.cpu().numpy().squeeze(), cmap='gray') # Changed!
                ax.set_title(f"Label: {label}")
            ax.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig('added_images_grid.png', dpi=300)
        plt.close()
    
    def transfer_unlabelled_to_labelled(self, indexes, method_name):
        """ Transfer unlabelled samples to labelled dataset
        Args:
            indexes: List of indices to transfer from unlabelled to labelled dataset
            method_name: Name of the active learning method being used
        """
        if not self.quiet:
            print(f" - Labeled images in training set: {len(self.lSet)}")
            print(f" - Remaining unlabeled images in unlabeled set: {len(self.uSet)}")
            print(f"Adding {len(indexes)} images to the training set using {method_name}.")
        self.lSet = np.append(self.lSet, indexes)
        mask = np.isin(self.uSet, indexes, invert=True)
        self.uSet = np.array(self.uSet)[mask].tolist()
        # Display added images for "eye test"
        #self.display_added_images(selected_images, selected_labels) # Commented out for now, not working..

        # Fetch newly added images and their labels
        if isinstance(self.dataObj.data, torch.Tensor):
            added_images = torch.stack([self.dataObj.data[i] for i in indexes])
            added_labels = torch.tensor([self.dataObj.targets[i] for i in indexes])
        elif isinstance(self.dataObj.data, np.ndarray):
            added_images = torch.tensor(self.dataObj.data[indexes])
            added_labels = torch.tensor([self.dataObj.targets[i] for i in indexes])
        elif isinstance(self.dataObj.data, list):
            added_images = torch.stack([self.dataObj.data[i] for i in indexes])
            added_labels = torch.tensor([self.dataObj.targets[i] for i in indexes])
        else:
            raise ValueError("Unsupported data type for dataset")

        # Save grid of added images
        self.save_added_images(added_images, added_labels,  method_name)

    def save_added_images(self, images, labels, method_name):
        """
        Save a grid of images with labels as a PNG file.

        Args:
            images: Tensor of images to be displayed in a grid.
            labels: Tensor of corresponding labels for the images.
            method_name: Name of the active learning method being used.
        """
        # Normalize images to [0, 1] and convert to float
        images = images.float() / 255.0
        num_images = min(25, len(images))
        grid_size = int(np.ceil(np.sqrt(num_images)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 10))

        # Set dynamic title based on iteration and method
        iteration = len(self.lSet) // self.b
        title = f"Added Images - {method_name} Iteration {iteration}"
        fig.suptitle(title)

        # Populate grid
        for i in range(grid_size * grid_size):
            ax = axes[i // grid_size, i % grid_size]
            if i < num_images:
                #ax.imshow(images[i].numpy().squeeze(), cmap='gray')
                ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray') # Changed!
                ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')

        plt.tight_layout()
        filename = f"added_images_{method_name}_iteration_{iteration}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

        if not self.quiet:
            print(f"Saved added images grid to {filename}")

    #########
    # Methods for ProbCover
    #########
    def construct_graph(self, batch_size=500):
        """ Construct a graph based on the distance between unlabelled and labelled samples
        Args:
            batch_size: Batch size for processing samples
        Returns:
            DataFrame: Graph representation with columns x, y, d
        """
        xs, ys, ds = [], [], []
        # distance computations are done in GPU
        if isinstance(self.dataObj.data, torch.Tensor):
            combined_indices = torch.tensor(np.append(self.lSet, self.uSet))
            features = self.dataObj.data[combined_indices].float()
        elif isinstance(self.dataObj.data, np.ndarray):
            combined_indices = np.concatenate((self.lSet, self.uSet))
            features = self.dataObj.data[combined_indices]
            features = torch.from_numpy(features).float()
        else:
            raise ValueError("Unsupported data type for dataset")
        
        # Reshape images to 2D format (N, 784) for distance computation
        cuda_features = features.reshape(features.shape[0], -1)
        # Normalize features
        cuda_features = F.normalize(cuda_features, p=2, dim=1).to(self.device)

        for i in range(len(cuda_features) // batch_size):
            cur_features = cuda_features[i * batch_size:(i + 1) * batch_size]
            dists = torch.cdist(cur_features.float(), cuda_features.float())
            mask = dists < self.delta
            x,y = mask.nonzero().T
            
            ds.append(dists[mask].cpu())
            xs.append(x.cpu() + i * batch_size)
            ys.append(y.cpu())
        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        # Create a sparse DataFrame to represent the graph
        return pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
    def approx_delta(self, alpha=0.95, max_delta=1, min_delta=0.01, random_shuffle=False):
        """
        Approximate optimal delta using k-means clustering as pseudo-labels.
        delta* = max{delta : purity(delta) >= alpha}
        
        Args:
            alpha: Purity threshold (default 0.95)
            max_delta: Maximum delta to test (default 1)
            min_delta: Minimum delta to test (default 0.001)
            random_shuffle: Whether to shuffle deltas for efficient search (default False)
        """
        print(f"Approximating delta for purity > {alpha}")
        
        # Prepare features and labels
        if isinstance(self.dataObj.data, torch.Tensor):
            features = self.dataObj.data[torch.tensor(self.uSet)].reshape(len(self.uSet), -1).float()
        elif isinstance(self.dataObj.data, np.ndarray):
            features = self.dataObj.data[self.uSet].reshape(len(self.uSet), -1)
            features = torch.from_numpy(features).float()
        else:
            raise ValueError("Unsupported data type for dataset")
        features = F.normalize(features, p=2, dim=1).to(self.device) # Normalize features 
        
        
        labels, _ = kmeans_torch(features, k=self.k)
        labels = labels.to(self.device)
        
        # Process in smaller chunks
        chunk_size = 1000  # Reduce memory usage
        num_points = len(features)
        purities = []
        
        deltas = np.linspace(min_delta, max_delta, 50)
        if random_shuffle:
            np.random.shuffle(deltas)
            cur_peak = 0
            cur_valley = 1
        
        for delta in tqdm(deltas, desc=f'Testing delta ', disable=self.quiet):
            if random_shuffle and delta < cur_peak:
                purities.append(1.5)
                #tqdm.write(f'Delta: {delta:.3f}, Purity: skipped')
                continue
            elif random_shuffle and delta > cur_valley:
                purities.append(0.0)
                #tqdm.write(f'Delta: {delta:.3f}, Purity: skipped')
                continue
            chunk_purities = []
            
            # Process chunks of points
            for i in range(0, num_points, chunk_size):
                end_idx = min(i + chunk_size, num_points)
                chunk_features = features[i:end_idx]
                
                # Calculate distances for current chunk
                dists = torch.cdist(chunk_features, features)
                mask = dists < delta
                
                # Get labels for neighbors
                chunk_labels = labels[i:end_idx].unsqueeze(1)
                neighbor_labels = labels.unsqueeze(0).expand(end_idx - i, -1)
                
                # Calculate purity for chunk
                same_label = (chunk_labels == neighbor_labels) & mask
                chunk_purity = (same_label.sum(1).float() / mask.sum(1).clamp(min=1)).mean()
                chunk_purities.append(chunk_purity.item())
                
            # Average purity across chunks
            purity = np.mean(chunk_purities)
            purities.append(purity)

            if random_shuffle and purity >= alpha:
                cur_peak = delta
            elif random_shuffle and purity < alpha - 0.2:
                cur_valley = delta 
            
            if not self.quiet:
                tqdm.write(f'Delta: {delta:.3f}, Purity: {purity:.3f}')
        
        # Select optimal delta
        purities = np.array(purities)
        valid_deltas = deltas[purities >= alpha]
        self.delta = np.max(valid_deltas) if len(valid_deltas) > 0 else min_delta
        
        print(f'Selected delta: {self.delta:.3f}')
        #self.visualize_deltas(deltas, purities, random_shuffle)
        return self.delta
    def visualize_deltas(self, deltas, purities, random_shuffle=False):
        """ Visualize purities for different deltas
        Args:
            deltas: List of deltas
            purities: List of purities
            random_shuffle: Whether deltas were shuffled
        """
        if random_shuffle:
            # Sort deltas and set all purities that are above 1 to the closest value below 1
            deltas, purities = zip(*sorted(zip(deltas, purities)))
            purities = np.minimum.accumulate(purities[::-1])[::-1]

        plt.figure(figsize=(10, 5))
        plt.plot(deltas, purities, label='Purity')
        plt.axvline(self.delta, color='r', linestyle='--', label='Selected Delta')
        plt.xlabel('Delta')
        plt.ylabel('Purity')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('delta_purity_visualization.png', dpi=300)
        plt.close()

    #########
    # Methods for TypiClust
    #########
    def extract_features(self, cluster=False):
        if cluster:
            model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
            # Adjust the first convolutional layer to accept one-channel input
            if self.input_channels != 3:
                # Adjust the first convolutional layer
                model.conv1 = torch.nn.Conv2d(
                    in_channels=self.input_channels,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False
                )
            model.to(self.device)
        else:
            model = deepcopy(self.model)
        
        # Remove the classification layer
        model.fc = torch.nn.Identity()
        model.eval()
        
        features = []
        with torch.no_grad():
            for images, _ in self.unlabelled_loader():
                images = images.to(self.device)
                output = model(images)
                features.append(output)
                
        features = torch.cat(features)
        features = F.normalize(features, p=2, dim=1)
        return features
    
    def perform_clustering(self):
        """ Cluster the data using K-means and return centroids and labels """
        num_clusters = min(len(self.lSet) + self.b, self.MAX_NUM_CLUSTERS)
        if not self.quiet:
            print(f"Performing clustering with {num_clusters} clusters")
        # Features are the penultimate layer of the model
        features = self.extract_features(cluster=True)

        if not self.quiet:
            print("features extracted")
        labels, centroids = kmeans_torch(features, num_clusters)

        return centroids, labels.cpu()
    def select_typical_samples(self):
        """ Select typical samples from each cluster based on density """
        # Extract features
        #features = self.dataObj.data[torch.tensor(self.uSet)].reshape(len(self.uSet), -1).float()
        #features = F.normalize(features, p=2, dim=1).to(self.device) # Normalize features 
        features = self.extract_features()

        # Handle minimum cluster size and sorting
        cluster_ids, cluster_sizes = np.unique(self.labels, return_counts=True)
        clusters_df = pd.DataFrame({
            'cluster_id': cluster_ids,
            'cluster_size': cluster_sizes,
            'neg_cluster_size': -1 * cluster_sizes
        })
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        clusters_df = clusters_df.sort_values(['neg_cluster_size'])

        typical_samples = []
        for i in range(self.b):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = np.where(self.labels == cluster)[0]
            rel_feats = features[indices]
            typicality = calculate_typicality(rel_feats.cpu(), min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            typical_samples.append(idx)
        return typical_samples
    
    #########
    # Methods for DCoM
    #########


    #########
    # Sampling Methods
    #########
    @set_name("Least Confidence")
    def least_confidence(self,top_frac=0.15):
        self.model.eval()
        predictions = []

        # Collect predictions for uncertainty
        with torch.no_grad():
            for images, _ in tqdm(self.unlabelled_loader(), disable=self.quiet):
                images = images.to(self.device)
                outputs = self.model(images).softmax(dim=1)
                predictions.extend(outputs.detach().cpu().numpy())
        # predictions = torch.tensor(predictions) # udkommenteret, prøver at effektivisere kode med np.array()
        predictions = torch.tensor(np.array(predictions))
        top_percent = int(top_frac * len(predictions))
        # Calculate uncertainties as the top confidence of each prediction
            # Least confidence sampling: Select samples with the highest uncertainty (lowest max probability)
        uncertainties, top_indices = predictions.max(dim=1)  # Get max probability and its indices
        top_indices = uncertainties.topk(top_percent, largest=False).indices  # Select least confident samples
        selected_indexes = top_indices[:self.b]
        self.transfer_unlabelled_to_labelled(indexes=selected_indexes, method_name = "Least Confidence")

    @set_name("Margin Sampling")
    def margin_sampling(self,top_frac=0.15):
        self.model.eval()
        predictions = []

        # Collect predictions for uncertainty
        with torch.no_grad():
            for images, _ in tqdm(self.unlabelled_loader(), disable=self.quiet):
                images = images.to(self.device)
                outputs = self.model(images).softmax(dim=1)
                predictions.extend(outputs.detach().cpu().numpy())
        top_percent = int(top_frac * len(predictions))
        # predictions = torch.tensor(predictions) # udkommenteret, prøver at effektivisere kode med np.array()
        predictions = torch.tensor(np.array(predictions))
        sorted_preds, _ = predictions.topk(2, dim=1)  # Get top two predictions for each sample
        margin = sorted_preds[:, 0] - sorted_preds[:, 1]  # Calculate margin (difference)
        top_indices = margin.topk(top_percent, largest=False).indices  # Select smallest margins
        selected_indexes = top_indices[:self.b]
        self.transfer_unlabelled_to_labelled(indexes=selected_indexes, method_name = "Margin Sampling")
    
    @set_name("Entropy Sampling")
    def entropy_sampling(self,top_frac=0.15):
        self.model.eval()
        predictions = []

        # Collect predictions for uncertainty
        with torch.no_grad():
            for images, _ in tqdm(self.unlabelled_loader(), disable=self.quiet):
                images = images.to(self.device)
                outputs = self.model(images).softmax(dim=1)
                predictions.extend(outputs.detach().cpu().numpy())
        top_percent = int(top_frac * len(predictions))
        # predictions = torch.tensor(predictions) # udkommenteret, prøver at effektivisere kode med np.array()
        predictions = torch.tensor(np.array(predictions))
        entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=1)
        top_indices = entropy.topk(top_percent, largest=True).indices
        selected_indexes = top_indices[:self.b]
        self.transfer_unlabelled_to_labelled(indexes=selected_indexes, method_name = "Entropy Sampling")

    @set_name("Random Sampling")
    def random_sampling(self, sample_size=None):
        """ Randomly sample from the unlabelled dataset for random sampling
        Args:
            sample_size: Number of samples to randomly sample
        Returns:
            List: Randomly sampled indices from unlabelled dataset"""
        
        if sample_size is None:
            sample_size = self.b
        random_indices = torch.randperm(len(self.uSet))[:sample_size]
        if not self.quiet:
            print(f"Adding {len(random_indices)} images to training set for Random Sampling.")
        self.transfer_unlabelled_to_labelled(random_indices, method_name = "Random Sampling")
        return random_indices

    @set_name("ProbCover")
    def prob_cover_labeling(self):
        """ Label unlabelled samples based on the ProbCover algorithm """
        combined_indices = torch.tensor(np.append(self.lSet, self.uSet))
        selected = []
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in range(self.b):
            coverage = len(covered_samples) / len(combined_indices)
            # Select samples with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(combined_indices))
            if not self.quiet:
                if i % 10 == 0:
                    print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            cur = degrees.argmax() # Here the paper uses random selection and their code uses this.

            # Remove incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)
        
        assert len(selected) == self.b, 'added a wrong number of samples'
        
        # Transfer selected samples from unlabelled to labelled
        if not self.quiet:
            print(min(selected))
        self.transfer_unlabelled_to_labelled(selected, method_name = "ProbCover")
    
    @set_name("TypiClust")
    def typiclust_labeling(self):
        """ Label unlabelled samples based on the TypiClust algorithm """
        # Ensure we have typical samples selected using TypiClust
        selected = self.select_typical_samples()
        if not self.quiet:
            print(f"Number of typical samples: {len(selected)}")

        # Iterate through and select samples to be labeled
        if len(selected) > self.b:
            selected = selected[:self.b]  # Select only the required number of samples
        if not self.quiet:
            print(f"Selected {len(selected)} samples using TypiClust for labeling.")

        # Transfer selected samples from unlabelled to labelled
        self.transfer_unlabelled_to_labelled(selected, method_name = "TypiClust")
    
    #########
    # Final functions
    #########
    def Al_Loop(self, function, title="Random Sampling", plot=True):
    # Active Learning Loop
        datapoint_list = []  # To store the number of labeled datapoints for active learning
        accuracy_list = []  # To store accuracy after each iteration of active learning
        for i in range(self.label_iterations):
            description = f"{title} {i + 1}"
            self.model.load_state_dict(self.model_parameters)  # Reset model parameters before training
            # Train model and save accuracies
            accuracies = self.train_model(val_interval=30, description=description)
            datapoint_list.append(len(self.lSet))
            accuracy_list.append(accuracies)
            if i < self.label_iterations - 1:
                function(self)
                if not self.quiet:
                    print(f"After AL iteration {i + 1}:")
                    print(f" - Labeled images in training set: {len(self.lSet)}")
                    print(f" - Remaining unlabeled images in unlabeled set: {len(self.uSet)}")
            if plot:
                self.visualize_decision_boundaries()
        self.reset_data()
        return datapoint_list, accuracy_list
    
    def compare_methods(self, methods=[random_sampling, least_confidence, margin_sampling, entropy_sampling, prob_cover_labeling, typiclust_labeling], no_plot=False):
        # Run Active Learning Loop
        datapoint_lists, accuracy_lists = [], []
        for method in methods:
            datapoint_list, accuracy_list = self.Al_Loop(method, title=method.__name__)
            datapoint_lists.append(np.array(datapoint_list))
            accuracy_lists.append(np.array(accuracy_list).max(-1))

        # Plotting the accuracy results for all three methods
        if no_plot:
            return datapoint_lists, accuracy_lists
    
        plt.figure(figsize=(10, 5))
        for i, method in enumerate(methods):
            plt.plot(datapoint_lists[i], accuracy_lists[i], label=method.__name__)
        plt.xlabel('Datapoints')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('method_comparison.png', dpi=300)
        plt.close()
        return datapoint_lists, accuracy_lists

    def test_methods(self, n_tests = 2, 
                     methods=[random_sampling, least_confidence, 
                              margin_sampling, entropy_sampling, 
                              prob_cover_labeling, typiclust_labeling], 
                     plot=True, quiet = False):
        self.quiet = quiet
        # Initialize result dictionaries for each method
        method_results = {method.__name__: {
            'datapoints': [],
            'accuracies': []
        } for method in methods}
        
        for i in range(n_tests):
            # Set seeds
            self.seed = np.random.randint(0, 100000)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            print(f"Starting Test {i}")  # Debug: Starting the test
            
            for method in tqdm(methods, desc=f"Test {i}"):
                print(f"Starting method: {method.__name__}")  # Debug: Starting the method
                
                try:
                    # Run AL Loop
                    datapoint_list, accuracy_list = self.Al_Loop(method, title=method.__name__, plot=plot)
                    print(f"Method {method.__name__} completed successfully.")  # Debug: Method completed

                    # Store results by method
                    method_results[method.__name__]['datapoints'].append(np.array(datapoint_list))
                    method_results[method.__name__]['accuracies'].append(np.array(accuracy_list))
                
                except Exception as e:
                    print(f"Error in method {method.__name__}: {e}")  # Debug: Capture any errors
                
                if not self.quiet:
                    print(f"Test {i} {method.__name__} done.")
            
            # Debug: Method results summary after all iterations
            print(f"Results after Test {i}: {method_results}")

        # Map method names to consistent keys
        method_key_map = {
            'Random Sampling': 'random_sampling',
            'Least Confidence': 'least_confidence',
            'Margin Sampling': 'margin_sampling',
            'Entropy Sampling': 'entropy_sampling',
            'ProbCover': 'prob_cover_labeling',
            'TypiClust': 'typiclust_labeling',
            'DCoM': 'dcom_labeling'
        }

        # Calculate statistics for each method
        aggregated_results = {}
        for method_name, results in method_results.items():
            print(f"Processing method: {method_name}")  # Debug: Method name
            try:
                # Get the standardized key
                standardized_key = method_key_map.get(method_name)
                if not standardized_key:
                    print(f"Method {method_name} does not have a predefined key. Skipping...")
                    continue

                # Convert lists to arrays for calculations
                print(f"Raw datapoints: {results['datapoints']}")  # Debug: Raw datapoints
                print(f"Raw accuracies: {results['accuracies']}")  # Debug: Raw accuracies

                datapoints = np.array(results['datapoints'])
                accuracies = np.array(results['accuracies'])

                # Debug: Shape and content of arrays
                print(f"{method_name} datapoints (array): {datapoints.shape}, content: {datapoints}")
                print(f"{method_name} accuracies (array): {accuracies.shape}, content: {accuracies}")

                # Calculate mean and error
                mean_accuracy = accuracies.mean(axis=0)
                error_accuracy = accuracies.std(axis=0)
                error_accuracy = 1.96 * error_accuracy / np.sqrt(n_tests)  # 95% CI

                # Store results using the standardized key
                aggregated_results[standardized_key] = {
                    'datapoints': datapoints,
                    'mean_accuracy': mean_accuracy,
                    'error_accuracy': error_accuracy
                }
                print(f"Aggregated results for {standardized_key}: {aggregated_results[standardized_key]}")  # Debug: Aggregated results
            except Exception as e:
                print(f"Error processing method {method_name}: {e}")  # Debug: Error message



        plt.figure(figsize=(10, 5))
        for method_name, results in aggregated_results.items():
            # Get correct shapes for plotting
            x = results['datapoints'].mean(axis=0)
            y = results['mean_accuracy'].reshape(-1)  # Flatten to 1D
            yerr = results['error_accuracy'].reshape(-1)  # Flatten to 1D
            
            plt.errorbar(x, y, 
                        yerr=yerr,
                        label=method_name,
                        capsize=3)
        
        plt.xlabel('Datapoints')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('test_methods_accuracy.png', dpi=300)
        plt.close()


        random_sampling = aggregated_results['random_sampling']['mean_accuracy'].reshape(-1)
        random_sampling_err = aggregated_results['random_sampling']['error_accuracy'].reshape(-1)

        plt.figure(figsize=(10, 5))
        for method_name, results in aggregated_results.items():
            # TAke the difference from random sampling
            try:
                x = results['datapoints'].mean(axis=0)
                y = results['mean_accuracy'].reshape(-1)
                yerr = results['error_accuracy'].reshape(-1)

                # Debugging: Print shapes and content
                print(f"{method_name} x shape: {x.shape}, content: {x}")
                print(f"{method_name} y shape: {y.shape}, content: {y}")
                print(f"{method_name} yerr shape: {yerr.shape}, content: {yerr}")

                y = y - random_sampling
                yerr = np.sqrt(yerr**2 + random_sampling_err**2)
                plt.errorbar(x, y, 
                            yerr=yerr,
                            label=method_name,
                            capsize=3)
            except Exception as e:
                print(f"Error plotting method {method_name}: {e}")

        plt.xlabel('Datapoints')
        plt.ylabel('Accuracy Difference from Random Sampling')
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig('test_methods_difference.png', dpi=300)
        plt.close()

        return aggregated_results