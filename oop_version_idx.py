import torch
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm  # For displaying progress bars during training
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # For visualizing decision boundaries
import pandas as pd
from sklearn.cluster import KMeans
import torch.nn.functional as F

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

class ActiveLearning:
    def __init__(self, dataObj, unlabelled_size, label_iterations, num_epochs,criterion=torch.nn.CrossEntropyLoss(), debug=False, lr=0.0005, seed=0, val_split=0.1, b=25):
        self.dataObj = deepcopy(dataObj) # This is the dataset that is passed to the class (MNIST in this case and with val removed)
        self.unlabelled_size = unlabelled_size # The size of the unlabelled dataset in percentage of the total dataset (minus validation)
        self.label_iterations = label_iterations # Number of iterations for active learning
        self.debug = debug # Debug mode for faster training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
        torch.manual_seed(seed)     # Set seed for reproducibility
        self.val_data = None        # Validation data (NOT INDEX)
        self.val_split = val_split  # Validation split in percentage
        self.uSet = None            # Unlabelled data Indices
        self.lSet = None            # Labelled data Indices

        self.first_uSet = None # Initial unlabelled data Indices
        self.first_lSet = None # Initial labelled data Indices

        self.num_epochs = num_epochs # Number of epochs for training
        self.b = b # Budget for each iteration of active learning

        self.model = torchvision.models.resnet18(weights=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_parameters = deepcopy(self.model.state_dict())
        self.model = self.model.to(self.device)

        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if self.debug:
            print("Model initializing")
            self.dataObj.data = self.dataObj.data[:1000]
            self.dataObj.targets = self.dataObj.targets[:1000]
            torch.set_num_threads(4)
        
        self.init_data(val_split)
        
    def init_data(self, val_split=0.1):
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
        self.uSet = deepcopy(self.first_uSet)
        self.lSet = deepcopy(self.first_lSet)

    def val_loader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=1024, shuffle=False)
    
    def train_loader(self):
        return torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataObj, self.lSet), batch_size=64, shuffle=True, drop_last=True)
    
    def unlabelled_loader(self):
        return torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataObj, self.uSet), batch_size=64, shuffle=False, drop_last=False)

    def validate_model(self):
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
        accuracies = []
        print(f"Starting training: {description}")
        for epoch in tqdm(range(self.num_epochs)):
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
                print(f'Epoch {epoch + 1}, {description} Accuracy: {val_accuracy:.2f}%')
        return accuracies
    
    def visualize_decision_boundaries(self, sample_size=500):
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
    
    def transfer_unlabelled_to_labelled(self, indexes, uncertainties=None):
        # If uncertainties are provided, sort indexes by uncertainty
        if uncertainties is not None:
            # Sort indexes by uncertainty, selecting the top uncertain samples (up to b)
            sorted_indexes = [idx for _, idx in sorted(zip(uncertainties, indexes), reverse=True)]
            selected_indexes = sorted_indexes[:self.b]  # Top uncertain samples, max b
        else:
            # If no uncertainties are provided (random sampling), use indexes directly
            selected_indexes = indexes[:self.b] if len(indexes) > 25 else indexes
        # Convert list of selected indices to a boolean mask for efficient filtering
        #mask = torch.tensor([i in selected_indexes for i in range(len(self.uSet))])
        # Save selected images and labels before modifying unlabelled_dataset
        #selected_images = self.uSet[mask]
        # Add selected unlabelled samples to the labelled training dataset
        print(f" - Labeled images in training set: {len(self.lSet)}")
        self.lSet = np.append(self.lSet, selected_indexes)
        print(f"Adding {len(selected_indexes)} images to the training set.")
        # Remove the added samples from the unlabelled dataset
        mask = np.isin(self.uSet, selected_indexes, invert=True)
        self.uSet = np.array(self.uSet)[mask].tolist()
        # Display added images for "eye test"
        #self.display_added_images(selected_images, selected_labels) # Commented out for now, not working..
        return self.lSet, self.uSet
    
    def uncertainty_labeling(self, top_frac=0.1, batch_size=64):
        self.model.eval()
        predictions = []

        # Collect predictions for uncertainty
        with torch.no_grad():
            for images, _ in tqdm(self.unlabelled_loader()):
                images = images.to(self.device)
                outputs = self.model(images).softmax(dim=1)
                predictions.extend(outputs.detach().cpu().numpy())
        predictions = torch.tensor(predictions)
        top_percent = int(top_frac * len(predictions))
        # Calculate uncertainties as the top confidence of each prediction
        uncertainties, top_indices = predictions.max(-1)[0].topk(top_percent, largest=False)
        # print(f"Adding {len(top_indices)} images to training set for Active Learning.")
        # Pass both top_indices and uncertainties to transfer_unlabelled_to_labelled
        self.transfer_unlabelled_to_labelled(indexes=top_indices, uncertainties=uncertainties)
        self.visualize_decision_boundaries()

    def random_sampling_iteration(self, sample_size):
        random_indices = torch.randperm(len(self.uSet))[:sample_size]
        print(f"Adding {len(random_indices)} images to training set for Random Sampling.")
        return random_indices

    def Al_Loop(self, function, title="Uncertainty Sampling"):
    # Active Learning Loop
        datapoint_list = []  # To store the number of labeled datapoints for active learning
        accuracy_list = []  # To store accuracy after each iteration of active learning
        for i in range(self.label_iterations):
            description = f"{title} {i + 1}"
            self.model.load_state_dict(self.model_parameters)  # Reset model parameters before training
            # Train model and save accuracies
            accuracies = self.train_model(val_interval=10, description=description)
            datapoint_list.append(len(self.lSet))
            accuracy_list.append(accuracies)
            if i < self.label_iterations - 1:
                function()
                print(f"After AL iteration {i + 1}:")
                print(f" - Labeled images in training set: {len(self.lSet)}")
                print(f" - Remaining unlabeled images in unlabeled set: {len(self.uSet)}")
        self.reset_data()
        return datapoint_list, accuracy_list
    
    def random_sampling(self,top_frac=0.1):
        datapoint_list = []  # Stores number of labeled data points over iterations for random sampling
        accuracy_list = []  # Stores accuracy at each iteration for random sampling
        for i in range(self.label_iterations):
            description = f"Random Sampling Iteration {i + 1}"
            self.model.load_state_dict(self.model_parameters)  # Reset model to initial state for fair comparison
            # Train model and store accuracies for random sampling method
            accuracies = self.train_model(val_interval=10, description=description)
            datapoint_list.append(len(self.lSet))
            accuracy_list.append(accuracies)
            
            # Perform random sampling to add new samples to the labeled dataset
            if i < self.label_iterations - 1:
                random_indices = self.random_sampling_iteration(int(top_frac * len(self.uSet)))
                self.transfer_unlabelled_to_labelled(random_indices)
                print(f"After Random Sampling iteration {i + 1}:")
                print(f" - Labeled images in training set: {len(self.lSet)}")
                print(f" - Remaining unlabeled images in unlabeled set: {len(self.uSet)}")
                self.visualize_decision_boundaries()  # Visualize decision boundaries after adding random samples
        self.reset_data()
        return datapoint_list, accuracy_list
    
    def compare_methods(self,no_plot=False):
        # Run Active Learning Loop
        datapoint_list, accuracy_list = self.Al_Loop(self.uncertainty_labeling)
        # Run Random Sampling Loop
        random_datapoint_list, random_accuracy_list = self.random_sampling()
        
        # Plotting the accuracy results for all three methods
        datapoints = np.array(datapoint_list)  # Data points for active learning
        accuracies = np.array(accuracy_list).max(-1)  # Max accuracy per iteration for active learning
        random_datapoints = np.array(random_datapoint_list)  # Data points for random sampling
        random_accuracies = np.array(random_accuracy_list).max(-1)  # Max accuracy per iteration for random sampling

        if no_plot:
            return datapoint_list, accuracy_list, random_datapoint_list, random_accuracy_list
        # Create a plot showing Active Learning, Random Sampling
        plt.figure(figsize=(10, 5))
        plt.plot(datapoints, accuracies, label='Active Learning Accuracy')
        plt.plot(random_datapoints, random_accuracies, label='Random Sampling Accuracy', linestyle='--')
        plt.xlabel('Datapoints')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # TODO: Add baseline method for comparison
        return datapoint_list, accuracy_list, random_datapoint_list, random_accuracy_list

##############
# This is an example of an expansion of the Active Learning class to implement the ProbCover algorithm (Use the same for other expansions, e.g. DCoM)
# You can also expand on the ActiveLearning class to include more general functions that can be used by all algorithms (such as baseline methods)
# Or inherit from probcover to include everything
##############
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
            if self.delta >= 20:
                print('Warning: Delta is very large, this may lead to a fully connected graph')
        try:
            self.graph_df = self.construct_graph()
        except:
            raise ValueError('Graph construction failed')
            
    #########
    # Custom functions for the new algorithm 
    #########
    def construct_graph(self, batch_size=500):
        xs, ys, ds = [], [], []
        # distance computations are done in GPU
        combined_indices = torch.tensor(np.append(self.lSet, self.uSet))
        subset_dataset = torch.utils.data.Subset(self.dataObj, combined_indices)

        cuda_features = torch.stack([img for img, _ in subset_dataset]).cuda()
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
    def prob_cover_labeling(self):
        combined_indices = torch.tensor(np.append(self.lSet, self.uSet))
        selected = []
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in range(self.b):
            coverage = len(covered_samples) / len(combined_indices)
            # Select samples with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(combined_indices))
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
        print(min(selected))
        self.transfer_unlabelled_to_labelled(selected)
        self.visualize_decision_boundaries()

    #########
    # Extend the compare_methods function to include the new algorithm and plot the results
    #########
    def compare_methods(self, no_plot=False):
    
        prob_data_lsit, prob_accuracy_list = self.Al_Loop(self.prob_cover_labeling, title="ProbCover")

        datapoint_list, accuracy_list, random_datapoint_list, random_accuracy_list = super().compare_methods(no_plot=True)

        # Plotting the accuracy results for all three methods
        datapoints = np.array(datapoint_list)
        accuracies = np.array(accuracy_list).max(-1)
        random_datapoints = np.array(random_datapoint_list)
        random_accuracies = np.array(random_accuracy_list).max(-1)
        prob_datapoints = np.array(prob_data_lsit)
        prob_accuracies = np.array(prob_accuracy_list).max(-1)

        if no_plot:
            return datapoint_list, accuracy_list, random_datapoint_list, random_accuracy_list, prob_data_lsit, prob_accuracy_list

        # Create a plot showing Active Learning, Random Sampling, and ProbCover
        plt.figure(figsize=(10, 5))
        plt.plot(datapoints, accuracies, label='Active Learning Accuracy')
        plt.plot(random_datapoints, random_accuracies, label='Random Sampling Accuracy', linestyle='--')
        plt.plot(prob_datapoints, prob_accuracies, label='ProbCover Accuracy', linestyle='-.')
        plt.xlabel('Datapoints')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return datapoint_list, accuracy_list, random_datapoint_list, random_accuracy_list, prob_data_lsit, prob_accuracy_list
