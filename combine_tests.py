import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def combine_experiment_runs(mean1, se1, n1, mean2, se2, n2):
    """
    Combine two sets of experimental runs with their means, standard errors, and sample sizes.
    
    Parameters:
    mean1 (float): Mean of the first set of runs
    se1 (float): Standard error of the first set of runs
    n1 (int): Number of runs in the first set
    mean2 (float): Mean of the second set of runs
    se2 (float): Standard error of the second set of runs
    n2 (int): Number of runs in the second set
    
    Returns:
    tuple: (combined_mean, combined_standard_error)
    """
    # Calculate combined sample size
    n_combined = n1 + n2
    
    # Calculate pooled mean (weighted average)
    combined_mean = (n1 * mean1 + n2 * mean2) / n_combined
    
    # Calculate pooled variance
    # First, convert standard errors to variances
    # Variance = (Standard Error)^2
    var1 = se1**2
    var2 = se2**2
    
    # Calculate pooled variance
    # Weighted average of variances, adjusted for sample sizes
    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n_combined - 2)
    
    # Calculate combined standard error
    # Uses the pooled variance and combined sample size
    combined_standard_error = np.sqrt(pooled_variance / n_combined)
    
    return combined_mean, combined_standard_error

# load experimental results
df2 = pd.read_csv('test_methods_results_MNIST_3.csv')
df3 = pd.read_csv('test_methods_results_MNIST_2.csv')

# extract datapoints from the dataframes
datapoints = []
for i in df2['Random Sampling'][1].replace('[', '').replace(']', '').split('\n ')[0].split(' '):
    if i != '':
        datapoints.append(int(i))

datapoints2 = []
for i in df3['Random Sampling'][1].replace('[', '').replace(']', '').split('\n ')[0].split(' '):
    if i != '':
        datapoints2.append(int(i))
n = 20

data = np.concatenate((datapoints2,datapoints))


# for method in df1.columns[1:]:
#     try:
#         mean1 = np.array(df1[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         se1 = np.array(df1[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         n1 = n
#         mean2 = np.array(df2[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         se2 = np.array(df2[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         n2 = n
#         combined_mean, combined_se = combine_experiment_runs(mean1, se1, n1, mean2, se2, n2)
#         mean = np.array(df3[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         se = np.array(df3[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)

#         combined_mean = np.append(mean,combined_mean)
#         combined_se = np.append(se,combined_se)
#         # Adjust indices for the combined data
#         total_indices = np.arange(len(combined_mean))
#         plt.plot(total_indices, combined_mean, label=method)
#         plt.fill_between(total_indices, combined_mean - combined_se, combined_mean + combined_se, alpha=0.2)
#     except:
#         combined_mean = np.array(df2[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         combined_se = np.array(df2[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         mean = np.array(df3[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         se = np.array(df3[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)

#         combined_mean = np.append(mean,combined_mean)
#         combined_se = np.append(se,combined_se)
#         # Adjust indices for the combined data
#         total_indices = np.arange(len(combined_mean))
#         combined_mean
#         plt.plot(total_indices, combined_mean, label=method)
#         plt.fill_between(total_indices, combined_mean - combined_se, combined_mean + combined_se, alpha=0.2)

# plt.xticks(total_indices, data)
# plt.xlabel('Number of datapoints')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('MNIST_total_accuracies.pdf')
# plt.show()


# mean = np.array(df3['Random Sampling'][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
# se = np.array(df3['Random Sampling'][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
# mean2 = np.array(df2['Random Sampling'][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
# se2 = np.array(df2['Random Sampling'][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
# mean1 = np.array(df1['Random Sampling'][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
# se1 = np.array(df1['Random Sampling'][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
# combined_mean, combined_se = combine_experiment_runs(mean1, se1, n, mean2, se2, n)
# combined_mean_r = np.append(mean,combined_mean)
# combined_se_r = np.append(se,combined_se)
# total_indices = np.arange(len(combined_mean))

# for method in df1.columns[2:]:
#     try:
#         mean1 = np.array(df1[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         se1 = np.array(df1[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         n1 = n
#         mean2 = np.array(df2[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         se2 = np.array(df2[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         n2 = n
#         combined_mean, combined_se = combine_experiment_runs(mean1, se1, n1, mean2, se2, n2)
#         mean = np.array(df3[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         se = np.array(df3[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)

#         combined_mean = np.append(mean,combined_mean) - combined_mean_r
#         combined_se = np.append(se,combined_se) - combined_se_r
#         # Adjust indices for the combined data
#         total_indices = np.arange(len(combined_mean))
#         combined_mean
#         plt.plot(total_indices, combined_mean, label=method)
#         plt.fill_between(total_indices, combined_mean - combined_se, combined_mean + combined_se, alpha=0.2)
#     except:
#         combined_mean = np.array(df2[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         combined_se = np.array(df2[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         mean = np.array(df3[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
#         se = np.array(df3[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)

#         combined_mean = np.append(mean,combined_mean) - combined_mean_r
#         combined_se = np.append(se,combined_se) - combined_se_r
#         # Adjust indices for the combined data
#         total_indices = np.arange(len(combined_mean))
#         combined_mean
#         plt.plot(total_indices, combined_mean, label=method)
#         plt.fill_between(total_indices, combined_mean - combined_se, combined_mean + combined_se, alpha=0.2)
        

# plt.xticks(total_indices, data)
# plt.xlabel('Number of datapoints')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('MNIST_difference_accuracies.pdf')
# plt.show()
















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cycler import cycler

# Set consistent styles
plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab10.colors)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def combine_experiment_runs(mean1, se1, n1, mean2, se2, n2):
    n_combined = n1 + n2
    combined_mean = (n1 * mean1 + n2 * mean2) / n_combined
    var1 = se1**2
    var2 = se2**2
    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n_combined - 2)
    combined_standard_error = np.sqrt(pooled_variance / n_combined)
    return combined_mean, combined_standard_error

# Load experimental results
df1 = pd.read_csv('test_methods_results_MNIST_4.csv')
df2 = pd.read_csv('test_methods_results_MNIST_3.csv')
df3 = pd.read_csv('test_methods_results_MNIST_2.csv')
df4 = pd.read_csv('test_methods_results_MNIST_5.csv')

# Extract datapoints
datapoints = [
    int(i) for i in df2['Random Sampling'][1].replace('[', '').replace(']', '').split('\n ')[0].split(' ') if i != ''
]
datapoints2 = [
    int(i) for i in df3['Random Sampling'][1].replace('[', '').replace(']', '').split('\n ')[0].split(' ') if i != ''
]
data = np.concatenate((datapoints2, datapoints))

n = 20
linestyles = ['-', '--', ':', '-.']

# Assign unique colors to methods
methods = df1.columns[1:]
colors = plt.cm.tab10.colors[:len(methods)]
method_colors = {method: colors[i] for i, method in enumerate(methods)}

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

mean1_r = np.array(df3['Random Sampling'][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
se1_r = np.array(df3['Random Sampling'][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
mean2_r = np.array(df4['Random Sampling'][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
se2_r = np.array(df4['Random Sampling'][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
mean_r, se_r = combine_experiment_runs(mean1_r, se1_r, n, mean2_r, se2_r, n)
# # Plot Total Accuracy
for i, method in enumerate(methods):
    mean1 = np.array(df3[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
    se1 = np.array(df3[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
    mean2 = np.array(df4[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
    se2 = np.array(df4[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
    combined_mean, combined_se = combine_experiment_runs(mean1, se1, n, mean2, se2, n)
    print(f"Method: {method}, Mean: {np.round(combined_mean,2)}")
    print(f"Method: {method}, SE: {combined_se}")
    combined_mean = combined_mean - mean_r
    if method == 'Random Sampling':
        combined_se = combined_se - se_r
    else:
        combined_se = combined_se + se_r

    total_indices = np.arange(len(combined_mean))
    axes[0].plot(total_indices, combined_mean, color=method_colors[method], linestyle=linestyles[i % len(linestyles)], label=method)
    axes[0].fill_between(total_indices, combined_mean - combined_se, combined_mean + combined_se, color=method_colors[method], alpha=0.2)
print(datapoints2)
axes[0].set_xticks(total_indices)
axes[0].set_xticklabels(datapoints2, rotation=45)
axes[0].set_xlabel('Cummulative Budget')
axes[0].set_ylabel('Difference in Accuracy')
axes[0].set_title('Difference in Accuracy of All Methods on MNIST')
axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.8)
axes[0].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)


# Plot Accuracy Differences
mean2_r = np.array(df2['Random Sampling'][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
se2_r = np.array(df2['Random Sampling'][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
mean1_r = np.array(df1['Random Sampling'][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
se1_r = np.array(df1['Random Sampling'][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
combined_mean_r, combined_se_r = combine_experiment_runs(mean1_r, se1_r, 10, mean2_r, se2_r, 20)

for i, method in enumerate(methods):
    try:
        mean1 = np.array(df1[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
        se1 = np.array(df1[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
        mean2 = np.array(df2[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
        se2 = np.array(df2[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
        combined_mean, combined_se = combine_experiment_runs(mean1, se1, 10, mean2, se2, n)
        print(f"Method: {method}, Mean: {np.round(combined_mean,2)}")
        combined_mean = combined_mean - combined_mean_r
        if method == 'Random Sampling':
            combined_se = combined_se - combined_se_r
        else:
            combined_se = combined_se + combined_se_r
        
    except:
        combined_mean = np.array(df2[method][2].replace('[', '').replace(']', '').split('\n ')).astype(float)
        combined_se = np.array(df2[method][3].replace('[', '').replace(']', '').split('\n ')).astype(float)
        print(f"Method: {method}, Mean: {np.round(combined_mean,2)}")
        if method == 'Random Sampling':
            combined_se = combined_se - combined_se_r
        combined_mean = combined_mean - combined_mean_r
        
    total_indices = np.arange(len(combined_mean))
    axes[1].plot(total_indices, combined_mean, color=method_colors[method], linestyle=linestyles[i % len(linestyles)], label=method)
    axes[1].fill_between(total_indices, combined_mean - combined_se, combined_mean + combined_se, color=method_colors[method], alpha=0.2)
print(datapoints)
axes[1].set_xticks(total_indices)
axes[1].set_xticklabels(datapoints, rotation=45)
axes[1].set_xlabel('Cummulative Budget')
axes[1].set_ylabel('Difference in Accuracy')
axes[1].set_title('Difference in Accuracy of All Methods on MNIST')
#axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.8)
axes[1].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the combined figure
plt.savefig('cifar_accuracies_combined.pdf', dpi=300)
plt.show()





        