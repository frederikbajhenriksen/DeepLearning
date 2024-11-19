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
            
            for method in methods:
                # Run AL Loop
                datapoint_list, accuracy_list = self.Al_Loop(method, title=method.__name__, plot=plot)
                
                # Store results by method
                method_results[method.__name__]['datapoints'].append(np.array(datapoint_list))
                method_results[method.__name__]['accuracies'].append(np.array(accuracy_list))
                
                if not self.quiet:
                    print(f"Test {i} {method.__name__} done.")
            print(f"Tests {100 * (i+1) / n_tests}% done.\n")

        # Calculate statistics for each method
        aggregated_results = {}
        for method_name, results in method_results.items():
            # Convert lists to arrays for calculations
            datapoints = np.array(results['datapoints'])
            accuracies = np.array(results['accuracies'])
            
            # Calculate mean and error
            mean_accuracy = accuracies.mean(axis=0)
            error_accuracy = accuracies.std(axis=0)
            error_accuracy = 1.96 * error_accuracy / np.sqrt(n_tests)  # 95% CI
            
            aggregated_results[method_name] = {
                'datapoints': datapoints,
                'mean_accuracy': mean_accuracy,
                'error_accuracy': error_accuracy
            }

        # TODO: REMOVE THIS
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

        cuda_features = torch.stack([img for img, _ in subset_dataset]).to_device(self.device)
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

class DCoM(ActiveLearning):
    def __init__(self, dataObj, unlabelled_size, label_iterations, num_epochs, criterion=torch.nn.CrossEntropyLoss(), debug=False, lr=0.0005, seed=0, val_split=0.1, b=25, alpha=0.7):
        super().__init__(dataObj, unlabelled_size, label_iterations, num_epochs, criterion, debug, lr, seed, val_split, b)
        self.alpha = alpha
        self.delta = 0.1  # Initialize delta with a default value
        self.graph_df = None
        self.embedding_space = None  # Will be computed during initialization

    def compute_margin(self, model, uSet):
        """ Compute the normalized margin between the two highest softmax outputs for each sample in unlabelled set """
        model.eval()
        margins = []
        with torch.no_grad():
            for images, _ in tqdm(torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataObj, uSet), batch_size=64, shuffle=False)):
                images = images.to(self.device)
                outputs = model(images).softmax(dim=1)
                top_vals, _ = outputs.topk(2, dim=1)
                margin = 1 - (top_vals[:, 0] - top_vals[:, 1])  # Margin = 1 - normalized margin
                margins.extend(margin.cpu().numpy())
        
        # Clear cache to manage memory after processing
        torch.cuda.empty_cache()

        return np.array(margins)

    def compute_coverage(self, L, delta):
        """ Compute the coverage probability of the current labeled set """
        if delta is None:
            raise ValueError("Delta value cannot be None in compute_coverage.")

        xs, ys, ds = [], [], []
        combined_indices = torch.tensor(np.append(L, self.uSet))
        subset_dataset = torch.utils.data.Subset(self.dataObj, combined_indices)
        cuda_features = torch.stack([img for img, _ in subset_dataset]).cuda().reshape(len(combined_indices), -1)
        cuda_features = F.normalize(cuda_features, p=2, dim=1)

        for i in range(len(cuda_features) // 500):
            cur_features = cuda_features[i * 500:(i + 1) * 500]
            dists = torch.cdist(cur_features.float(), cuda_features.float())
            mask = dists < delta
            x, y = mask.nonzero().T
            ds.append(dists[mask].cpu())
            xs.append(x.cpu() + i * 500)
            ys.append(y.cpu())
        
        # Clear cache after processing to avoid memory overflow
        torch.cuda.empty_cache()

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        return pd.DataFrame({'x': xs, 'y': ys, 'd': ds})

    def update_competence_score(self, coverage, SL):
        """ Compute the competence score based on the coverage probability """
        return (1 + np.exp(-30 * (1 - SL))) / (1 + np.exp(-30 * (coverage - 0.9)))

    def select_query(self, uSet, L, delta_avg, SL):
        """ Select q points using Dynamic Coverage & Margin Mix """
        margin = self.compute_margin(self.model, uSet)
        coverage = self.compute_coverage(L, delta_avg)

        query_set = []
        ranking_scores = []
        
        for idx, u in enumerate(uSet):
            out_degree_rank = np.sum(coverage['x'] == u)  # Out-Degree Rank: Number of neighbors within the coverage
            ranking_score = SL * (1 - margin[idx]) + (1 - SL) * out_degree_rank
            ranking_scores.append((u, ranking_score))

        ranking_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by the ranking score
        query_set = [x[0] for x in ranking_scores[:self.b]]
        return query_set

    def expand_delta(self, query_set, L, delta_max=1.0):
        """ Expand the delta for the new labeled points """
        new_deltas = []
        coverage = self.compute_coverage(L, delta_max)
        for v in query_set:
            delta_opt = self.binary_search_delta(v, L, coverage)
            new_deltas.append(delta_opt)
        return new_deltas

    def binary_search_delta(self, v, L, coverage, delta_max=1.0):
        """ Perform binary search to find optimal delta for the new labeled point """
        low, high = 0, delta_max
        while high - low > 0.01:
            mid = (low + high) / 2
            if self.compute_coverage(L + [v], mid) > coverage:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def transfer_unlabelled_to_labelled(self, indexes):
        """ Transfer unlabelled samples to labelled dataset """
        print(f"Before labeling: uSet size: {len(self.uSet)}, lSet size: {len(self.lSet)}")
      
        self.lSet.extend(indexes)
        self.uSet = [item for item in self.uSet if item not in indexes]
        
        print(f"After labeling: uSet size: {len(self.uSet)}, lSet size: {len(self.lSet)}")

        # Optionally visualize the decision boundaries for added clarity
        self.visualize_decision_boundaries()


    #########
    # Create a label iteration function for the DCoM algorithm
    #########

    @set_name("DCoM")
    def dcom_labeling(self):
        """ Label unlabelled samples based on the DCoM algorithm """
        
        # Ensure that delta is valid
        if self.delta is None:
            self.delta = 0.1  # Default delta value if not updated yet

        coverage = self.compute_coverage(self.lSet, self.delta)
        
        SL = self.update_competence_score(coverage, 0.9)  # Example value for SL

        query_set = self.select_query(self.uSet, self.lSet, delta_avg=0.75, SL=SL)

        if len(query_set) == 0:
            print("No more samples to label! Stopping iteration.")
            return  # Stop if no more samples to label
        
        print(f"Selected query set: {query_set[:5]}...")  # Debug: print first 5 queries
        
        self.transfer_unlabelled_to_labelled(query_set)

        if len(query_set) > 0:
            print(f"After query set selection: Remaining unlabeled: {len(self.uSet)} samples")

        self.visualize_decision_boundaries()

    @set_name("Random Sampling")
    def random_sampling(self, sample_size=None):
        """ Randomly sample from the unlabelled dataset for random sampling """
        return super().random_sampling(sample_size)

    #def compare_methods(self, methods=[uncertainty_labeling, random_sampling, dcom_labeling], no_plot=False):

    def compare_methods(self, methods=[dcom_labeling], no_plot=False):
        return super().compare_methods(methods, no_plot) # TODO: FIX THIS!!
        
        
        """datapoint_lists, accuracy_lists = [], []
        for method in methods:
            datapoint_list, accuracy_list = self.Al_Loop(method, title=method.__name__)
            datapoint_lists.append(np.array(datapoint_list))
            accuracy_lists.append(np.array(accuracy_list).max(-1))

        # Plotting the accuracy results for all three methods
        if not no_plot:
            plt.figure(figsize=(10, 5))
            for i, method in enumerate(methods):
                plt.plot(datapoint_lists[i], accuracy_lists[i], label=method.__name__)
            plt.xlabel('Datapoints')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            plt.show()

        return datapoint_lists, accuracy_lists"""
