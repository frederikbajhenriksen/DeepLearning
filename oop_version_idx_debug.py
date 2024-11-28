import torch
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm  # For displaying progress bars during training
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # For visualizing decision boundaries
import pandas as pd
import torch.nn.functional as F

# TODO: Nice visualization of results from tests
# TODO: ADD VISUALIZATION OF THE CHOSEN IMAGES
# TODO: hyperparameter tuning method
# TODO: SAVE results to a file

def set_name(name):
    def decorator(func):
        func.__name__ = name
        return func
    return decorator

class ActiveLearning:
    def __init__(self, dataObj, unlabelled_size, label_iterations, num_epochs,criterion=torch.nn.CrossEntropyLoss(), debug=False, lr=0.0005, seed=0, val_split=0.1, b=25):
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
        self.b = b                  # Budget for each iteration of active learning

        self.model = torchvision.models.resnet18(weights=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_parameters = deepcopy(self.model.state_dict())
        self.model = self.model.to(self.device)

        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.quiet = False

        if self.debug:
            print("Model initializing")
            self.dataObj.data = self.dataObj.data[:1000]
            self.dataObj.targets = self.dataObj.targets[:1000]
            torch.set_num_threads(4)
        
        self.init_data(val_split)
    
    #########
    # General functions for active learning
    #########
    def init_data(self, val_split=0.1):
        """ Initialize the data for active learning by splitting into training and validation sets 
        Args:
            val_split: Validation split percentage
        """
        self.val_data = deepcopy(self.dataObj)
        train_size = int((1 - val_split) * len(self.dataObj))
        val_size = len(self.dataObj) - train_size
        indexes = torch.randperm(len(self.dataObj)).tolist()

        val_index = indexes[train_size:]
        self.val_data.targets = self.val_data.targets[val_index]
        self.val_data.data = self.val_data.data[val_index]
        
        indexes_train = indexes[:train_size]
        self.dataObj.targets = self.dataObj.targets[indexes_train]
        self.dataObj.data = self.dataObj.data[indexes_train]

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
        self.model = torchvision.models.resnet18(weights=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_parameters = deepcopy(self.model.state_dict())
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005) # Reset optimizer

    def val_loader(self):
        """ Return validation data loader for model evaluation
        Returns:
            DataLoader: Validation data loader
        """
        return torch.utils.data.DataLoader(self.val_data, batch_size=1024, shuffle=False)
    
    def train_loader(self):
        """ Return training data loader for active learning iterations
        Returns:
            DataLoader: Training data loader
        """
        return torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataObj, self.lSet), batch_size=64, shuffle=True, drop_last=True)
    
    def unlabelled_loader(self):
        """ Return unlabelled data loader for active learning iterations 
        Returns:
            DataLoader: Unlabelled data loader
        """
        return torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataObj, self.uSet), batch_size=64, shuffle=False, drop_last=False)

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
        
        # Use a subset of unlabelled data if the dataset is large
        
        subset_indices = torch.randperm(len(self.uSet))[:sample_size]
        selected_indices = torch.tensor(self.uSet)[subset_indices]
        subset_data = self.dataObj.data[selected_indices]
        subset_targets = self.dataObj.targets[selected_indices]

        with torch.no_grad():
            for image, label in zip(subset_data, subset_targets):
                image = image.float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0  # Normalize and add batch dimension
                embedding = self.model(image).cpu().numpy()
                embeddings.append(embedding)
                labels.append(label)
        # Run t-SNE for dimensionality reduction
        embeddings = np.array(embeddings).reshape(len(subset_data), -1)
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
        plt.legend(handles=scatter.legend_elements()[0], labels=range(10), title="Digits")
        plt.title("t-SNE of Decision Boundaries in Unlabelled Dataset")
        plt.show()

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
                ax.imshow(image.numpy().squeeze(), cmap='gray')
                ax.set_title(f"Label: {label}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def transfer_unlabelled_to_labelled(self, indexes):
        """ Transfer unlabelled samples to labelled dataset
        Args:
            indexes: List of indices to transfer from unlabelled to labelled dataset
            uncertainties: List of uncertainties for each sample
        """
        if not self.quiet:
            print(f" - Labeled images in training set: {len(self.lSet)}")
            print(f" - Remaining unlabeled images in unlabeled set: {len(self.uSet)}")
            print(f"Adding {len(indexes)} images to the training set.")
        self.lSet = np.append(self.lSet, indexes)
        mask = np.isin(self.uSet, indexes, invert=True)
        self.uSet = np.array(self.uSet)[mask].tolist()
        # Display added images for "eye test"
        #self.display_added_images(selected_images, selected_labels) # Commented out for now, not working..

    #########
    # Baseline Sampling Methods
    #########
    @set_name("Least Confidence")
    def least_confidence(self,top_frac=0.1):
        self.model.eval()
        predictions = []

        # Collect predictions for uncertainty
        with torch.no_grad():
            for images, _ in tqdm(self.unlabelled_loader(), disable=self.quiet):
                images = images.to(self.device)
                outputs = self.model(images).softmax(dim=1)
                predictions.extend(outputs.detach().cpu().numpy())
        predictions = torch.tensor(predictions)
        top_percent = int(top_frac * len(predictions))
        # Calculate uncertainties as the top confidence of each prediction
            # Least confidence sampling: Select samples with the highest uncertainty (lowest max probability)
        uncertainties, top_indices = predictions.max(dim=1)  # Get max probability and its indices
        top_indices = uncertainties.topk(top_percent, largest=False).indices  # Select least confident samples
        selected_indexes = top_indices[:self.b]
        self.transfer_unlabelled_to_labelled(indexes=selected_indexes)

    @set_name("Margin Sampling")
    def margin_sampling(self,top_frac=0.1):
        self.model.eval()
        predictions = []

        # Collect predictions for uncertainty
        with torch.no_grad():
            for images, _ in tqdm(self.unlabelled_loader(), disable=self.quiet):
                images = images.to(self.device)
                outputs = self.model(images).softmax(dim=1)
                predictions.extend(outputs.detach().cpu().numpy())
        top_percent = int(top_frac * len(predictions))
        predictions = torch.tensor(predictions)
        sorted_preds, _ = predictions.topk(2, dim=1)  # Get top two predictions for each sample
        margin = sorted_preds[:, 0] - sorted_preds[:, 1]  # Calculate margin (difference)
        top_indices = margin.topk(top_percent, largest=False).indices  # Select smallest margins
        selected_indexes = top_indices[:self.b]
        self.transfer_unlabelled_to_labelled(indexes=selected_indexes)
    
    @set_name("Entropy Sampling")
    def entropy_sampling(self,top_frac=0.1):
        self.model.eval()
        predictions = []

        # Collect predictions for uncertainty
        with torch.no_grad():
            for images, _ in tqdm(self.unlabelled_loader(), disable=self.quiet):
                images = images.to(self.device)
                outputs = self.model(images).softmax(dim=1)
                predictions.extend(outputs.detach().cpu().numpy())
        top_percent = int(top_frac * len(predictions))
        predictions = torch.tensor(predictions)
        entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=1)
        top_indices = entropy.topk(top_percent, largest=True).indices
        selected_indexes = top_indices[:self.b]
        self.transfer_unlabelled_to_labelled(indexes=selected_indexes)

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
        self.transfer_unlabelled_to_labelled(random_indices)
        return random_indices

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
    
    def compare_methods(self, methods=[random_sampling, least_confidence, margin_sampling, entropy_sampling], no_plot=False):
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
        plt.show()
        return datapoint_lists, accuracy_lists

    def test_methods(self, n_tests = 2, methods=[random_sampling, least_confidence, margin_sampling, entropy_sampling], plot=True, quiet = False):
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
            'Entropy Sampling': 'entropy_sampling'
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
        plt.show()

        random_sampling = aggregated_results['random_sampling']['mean_accuracy']
        random_sampling_err = aggregated_results['random_sampling']['error_accuracy']

        plt.figure(figsize=(10, 5))
        for method_name, results in aggregated_results.items():
            # TAke the difference from random sampling
            x = results['datapoints'].mean(axis=0)
            y = results['mean_accuracy'].reshape(-1)
            yerr = results['error_accuracy'].reshape(-1)

            y = y - random_sampling
            yerr = np.sqrt(yerr**2 + random_sampling_err**2)
            plt.errorbar(x, y, 
                        yerr=yerr,
                        label=method_name,
                        capsize=3)

        plt.xlabel('Datapoints')
        plt.ylabel('Accuracy Difference from Random Sampling')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return aggregated_results
    



##############
# This is an example of an expansion of the Active Learning class to implement the ProbCover algorithm (Use the same for other expansions, e.g. DCoM)
# You can also expand on the ActiveLearning class to include more general functions that can be used by all algorithms (such as baseline methods)
# Or inherit from probcover to include everything
##############
# TODO: Make faster or import from a library faiss
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
    
    # Get target dtype and device
    target_dtype = x.dtype

    # Randomly initialize centroids
    rand_indices = torch.randperm(n_samples)[:k]
    centroids = x[rand_indices]
    
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
                new_centroids[i] = centroids[i]
        
        # Check convergence
        if torch.norm(new_centroids - centroids) < tol:
            break
            
        centroids = new_centroids
    labels = labels.to(dtype=target_dtype)
    return labels, centroids

class ProbCover(ActiveLearning):
    #########
    # Create a new constructor to add any new parameters needed for the new algorithm and call the parent constructor
    #########
    def __init__(self, dataObj, unlabelled_size, label_iterations, num_epochs,criterion=torch.nn.CrossEntropyLoss(), debug=False, lr=0.0005, seed=0, val_split=0.1, b=25, delta=None, alpha=0.7):
        super().__init__(dataObj, unlabelled_size, label_iterations, num_epochs, criterion, debug, lr, seed, val_split, b)
        
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
            
    #########
    # Custom functions for the new algorithm 
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
        combined_indices = torch.tensor(np.append(self.lSet, self.uSet))
        subset_dataset = torch.utils.data.Subset(self.dataObj, combined_indices)

        #cuda_features = torch.stack([img for img, _ in subset_dataset]).to_device(self.device)
        cuda_features = torch.stack([img for img, _ in subset_dataset]).to(self.device)
        # Reshape images to 2D format (N, 784) for distance computation
        cuda_features = cuda_features.reshape(cuda_features.shape[0], -1)
        # Normalize features
        cuda_features = F.normalize(cuda_features, p=2, dim=1)

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
    
    def approx_delta(self, alpha=0.95, max_delta=1, min_delta=0.01, random_shuffle=True):
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
        features = self.dataObj.data[torch.tensor(self.uSet)].reshape(len(self.uSet), -1).float()
        features = F.normalize(features, p=2, dim=1).to(self.device) # Normalize features 
        if self.debug:
            labels = self.dataObj.targets[torch.tensor(self.uSet)].to(self.device)
        else:
            # Use k-means clustering as pseudo-labels
            unique_labels = len(self.dataObj.targets.unique())
            labels, _ = kmeans_torch(features, k=unique_labels)
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
        
        for delta in tqdm(deltas, desc=f'Testing delta '):
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
        plt.show()

    #########
    # Create a label iteration function for the new algorithm
    #########
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
        self.transfer_unlabelled_to_labelled(selected)

    ###########
    # Include the new method in the compare_methods function and add the old methods to the list
    ###########
    @set_name("Least Confidence")
    def least_confidence(self,top_frac=0.1):
        return super().least_confidence(top_frac)
    @set_name("Margin Sampling")
    def margin_sampling(self,top_frac=0.1):
        return super().margin_sampling(top_frac)
    @set_name("Entropy Sampling")
    def entropy_sampling(self,top_frac=0.1):
        return super().entropy_sampling(top_frac)
    @set_name("Random Sampling")
    def random_sampling(self, sample_size=None):
        return super().random_sampling(sample_size)
    def compare_methods(self, methods=[random_sampling, prob_cover_labeling, least_confidence, margin_sampling, entropy_sampling], no_plot=False):
        return super().compare_methods(methods, no_plot)



######################################################################
# TYPICLUST
######################################################################

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import NearestNeighbors

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

class TypiClust(ActiveLearning):

    MAX_NUM_CLUSTERS = 500  # Example value, adjust as needed
    MIN_CLUSTER_SIZE = 5    # Example value, adjust as needed
    K_NN = 20               # Number of neighbors for density calculation

    def __init__(self, dataObj, unlabelled_size, label_iterations, num_epochs, criterion=torch.nn.CrossEntropyLoss(), debug=False, lr=0.0005, seed=0, val_split=0.1, b=25, k=10):
        # Call the parent constructor
        super().__init__(dataObj, unlabelled_size, label_iterations, num_epochs, criterion, debug, lr, seed, val_split, b)
        self.k = k  # Number of clusters for TypiClust

        # Perform clustering and select initial samples
        self.centroids, self.labels = self.perform_clustering()
        self.typical_samples = self.select_typical_samples()

    def perform_clustering(self):
        """ Cluster the data using K-means and return centroids and labels """
        num_clusters = min(len(self.lSet) + self.b, self.MAX_NUM_CLUSTERS)
        if num_clusters <= 50:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        else:
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, random_state=0)

        # features = self.extract_features(self.dataObj.data[self.uSet])
        features = self.dataObj.data[torch.tensor(self.uSet)].reshape(len(self.uSet), -1).float()
        features = F.normalize(features, p=2, dim=1).to(self.device) # Normalize features
        if not self.quiet:
            print("features extracted")
        labels = kmeans.fit_predict(features)
        return kmeans.cluster_centers_, labels

    def select_typical_samples(self):
        """ Select typical samples from each cluster based on density """

        # Extract features
        features = self.dataObj.data[torch.tensor(self.uSet)].reshape(len(self.uSet), -1).float()
        features = F.normalize(features, p=2, dim=1).to(self.device) # Normalize features 
        
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
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            typical_samples.append(idx)

        return typical_samples
    

    #########
    # Create a label iteration function for TypiClust
    #########
    @set_name("TypiClust")
    def typiclust_labeling(self):
        """ Label unlabelled samples based on the TypiClust algorithm """
        # Ensure we have typical samples selected using TypiClust
        selected = self.typical_samples
        if not self.quiet:
            print(f"Number of typical samples: {len(self.typical_samples)}")

        # Iterate through and select samples to be labeled
        if len(selected) > self.b:
            selected = selected[:self.b]  # Select only the required number of samples
        if not self.quiet:
            print(f"Selected {len(selected)} samples using TypiClust for labeling.")

        # Transfer selected samples from unlabelled to labelled
        self.transfer_unlabelled_to_labelled(selected)

    ###########
    # Include the new method in the compare_methods function and add the old methods to the list
    ###########

    @set_name("Random Sampling")
    def random_sampling(self, sample_size=None):
        return super().random_sampling(sample_size)

    def compare_methods(self, methods=[typiclust_labeling, random_sampling], no_plot=False):
        return super().compare_methods(methods, no_plot, title="TypiClust")


# ######################################################################
#  DCoM
# ######################################################################

class DCoM(ProbCover):
    def __init__(self, dataObj, unlabelled_size, label_iterations, num_epochs, criterion=torch.nn.CrossEntropyLoss(), 
                 debug=False, lr=0.0005, seed=0, val_split=0.1, b=25, delta=None, alpha=0.7, competence_threshold=0.5):
        """Constructor for DCoM class, inheriting from ProbCover."""

        super().__init__(dataObj, unlabelled_size, label_iterations, num_epochs, criterion, debug, lr, seed, val_split, b, delta, alpha)
        self.competence_threshold = competence_threshold

    def calculate_competence(self):
        """Calculates the competence score of the learner based on coverage."""

        coverage_probability = len(self.graph_df.y.unique()) / len(self.dataObj)
        return coverage_probability

    def dcom_scoring(self):
        """Implements the DCoM dynamic scoring combining coverage and margin-based uncertainty."""

        self.model.eval()
        with torch.no_grad():
            predictions = []
            for images, _ in self.unlabelled_loader():
                images = images.to(self.device)
                outputs = self.model(images).softmax(dim=1)
                predictions.extend(outputs.cpu().numpy())
            predictions = torch.tensor(predictions)
        
        # Margin scores
        margins = predictions.topk(2, dim=1).values
        margin_uncertainty = margins[:, 0] - margins[:, 1]
        
        # Competence score combining coverage and margin uncertainty
        competence_score = self.calculate_competence()
        adjusted_scores = (1 - competence_score) * self.out_degree_rank() + competence_score * margin_uncertainty
        return adjusted_scores

    def out_degree_rank(self):
        """Calculates the out-degree rank for all unlabeled samples based on the current graph."""
        
        degrees = torch.bincount(torch.tensor(self.graph_df.x), minlength=len(self.dataObj))
        return degrees[self.uSet]

    #########
    # Create a label iteration function for DCoM
    #########

    @set_name("DCoM")
    def dcom_sampling(self):
        """Selects samples based on DCoM scoring and transfers them to the labeled set."""

        scores = self.dcom_scoring()
        top_indices = torch.argsort(scores, descending=True)[:self.b]
        selected = torch.tensor(self.uSet)[top_indices].tolist()
        self.transfer_unlabelled_to_labelled(selected)

    ###########
    # Include the new method in the compare_methods function and add the old methods to the list
    ###########

    #def compare_methods(self, methods=[random_sampling, prob_cover_labeling, dcom_sampling, least_confidence, margin_sampling, entropy_sampling], no_plot=False):
    def compare_methods(self, methods=[dcom_sampling], no_plot=False):
        """Adds DCoM to the list of methods for comparison."""

        return super().compare_methods(methods, no_plot)
