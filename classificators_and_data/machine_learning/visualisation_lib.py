import matplotlib.pyplot as plt
import numpy as np
import time
import os
import seaborn as sns
import datetime


def plot_heatmap_mean_test_scores(cv_results_df, param_x, param_y):
    """
    Plots a heatmap of mean test scores for combinations of two hyperparameters.
    """
    pivot_table = cv_results_df.pivot_table(
        values='mean_test_score',
        index='param_' + param_y,
        columns='param_' + param_x
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='viridis')
    plt.title(f"Mean Test Score Heatmap ({param_x} vs. {param_y})")
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.show()


def plot_fit_and_score_times(cv_results_df, param_name):
    """
    Plots fit times and score times vs. a specified hyperparameter.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=cv_results_df['param_' + param_name],
        y=cv_results_df['mean_fit_time'],
        marker='o',
        label='Mean Fit Time'
    )
    sns.lineplot(
        x=cv_results_df['param_' + param_name],
        y=cv_results_df['mean_score_time'],
        marker='o',
        label='Mean Score Time'
    )
    plt.title(f"Fit and Score Times vs. {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mean_test_scores(cv_results_df, param_name):
    """
    Plots mean test scores vs. a specified hyperparameter.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=cv_results_df['param_' + param_name],
        y=cv_results_df['mean_test_score'],
        marker='o'
    )
    plt.title(f"Mean Test Score vs. {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Mean Test Score")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm, model_name, result_folder):
    """
    Plot and save the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    plot_path = os.path.join(result_folder, f"confusion_matrix_{model_name}_{int(datetime.now().timestamp())}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")
    
    
def plot_test_scores(cv_results, model_name):
    """Plot the test scores for each hyperparameter combination."""
    mean_test_score = cv_results['mean_test_score']
    params = cv_results['params']
    
    param_str = [f"{p}" for p in params]
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_test_score, marker='o')
    plt.xticks(np.arange(len(param_str)), param_str, rotation=90, fontsize=10)
    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('Mean Test Score')
    plt.title(f'Mean Test Scores for {model_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cv_results(cv_results, model_name):
    """Plot mean and std of test scores for each hyperparameter combination across CV splits."""
    mean_test_score = cv_results['mean_test_score']
    std_test_score = cv_results['std_test_score']
    params = cv_results['params']
    
    param_str = [f"{p}" for p in params]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(np.arange(len(param_str)), mean_test_score, yerr=std_test_score, fmt='o', capsize=5)
    plt.xticks(np.arange(len(param_str)), param_str, rotation=90, fontsize=10)
    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('Mean Test Score with Std Dev')
    plt.title(f'CV Results for {model_name} (Mean + Std)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fit_and_score_times(cv_results, model_name):
    """Plot the mean fit time and score time for each hyperparameter combination."""
    mean_fit_time = cv_results['mean_fit_time']
    mean_score_time = cv_results['mean_score_time']
    params = cv_results['params']
    
    param_str = [f"{p}" for p in params]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(mean_fit_time, marker='o', color='blue', label='Fit Time')
    ax[0].set_xticks(np.arange(len(param_str)))
    ax[0].set_xticklabels(param_str, rotation=90, fontsize=10)
    ax[0].set_ylabel('Mean Fit Time (s)')
    ax[0].set_title(f'Mean Fit Time for {model_name}')
    ax[0].grid(True)

    ax[1].plot(mean_score_time, marker='o', color='green', label='Score Time')
    ax[1].set_xticks(np.arange(len(param_str)))
    ax[1].set_xticklabels(param_str, rotation=90, fontsize=10)
    ax[1].set_ylabel('Mean Score Time (s)')
    ax[1].set_title(f'Mean Score Time for {model_name}')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    
    
def plot_cv_results(cv_results, model_name, result_folder):
    """
    Plot mean test scores for each hyperparameter combination.
    Supports up to 2 hyperparameters for plotting.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    params = cv_results['params']
    mean_test_scores = cv_results['mean_test_score']
    
    hyperparameters = list(params[0].keys())
    n_hyperparameters = len(hyperparameters)
    
    if n_hyperparameters == 1:
        param_name = hyperparameters[0]
        param_values = [p[param_name] for p in params]
        plt.figure()
        plt.plot(param_values, mean_test_scores, marker='o')
        plt.xlabel(param_name)
        plt.ylabel('Mean Test Score')
        plt.title(f'Mean Test Score vs {param_name} for {model_name}')
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = os.path.join(result_folder, f'cv_results_{model_name}_{param_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"CV results plot saved to {plot_path}")
        
    elif n_hyperparameters == 2:
        param_name1 = hyperparameters[0]
        param_name2 = hyperparameters[1]
        param_values1 = sorted(set([p[param_name1] for p in params]))
        param_values2 = sorted(set([p[param_name2] for p in params]))
        score_matrix = np.zeros((len(param_values2), len(param_values1)))
        for idx, p in enumerate(params):
            i = param_values1.index(p[param_name1])
            j = param_values2.index(p[param_name2])
            score_matrix[j, i] = mean_test_scores[idx]
        plt.figure()
        sns.heatmap(score_matrix, annot=True, xticklabels=param_values1, yticklabels=param_values2, cmap='viridis')
        plt.xlabel(param_name1)
        plt.ylabel(param_name2)
        plt.title(f'Hyperparameter Heatmap for {model_name}')
        plt.tight_layout()
        
        plot_path = os.path.join(result_folder, f'cv_results_heatmap_{model_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"CV results heatmap saved to {plot_path}")
    else:
        print("Cannot plot CV results: more than two hyperparameters varied.")
        
        
def plot_feature_importances(importances, n_channels, model_name, result_folder):
    """
    Plot and save the feature importances as a scalp map.
    """
    import mne
    import matplotlib.pyplot as plt
    
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:n_channels]
    
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
    info.set_montage(montage)
    
    data = importances.reshape(n_channels, 1)
    evoked = mne.EvokedArray(data, info)
    
    fig = evoked.plot_topomap(times=0, size=3, show=False)
    
    plot_path = os.path.join(result_folder, f'feature_importances_scalp_{model_name}_{int(datetime.now().timestamp())}.png')
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Feature importances scalp map saved to {plot_path}")
    
    
    
def plot_erps(X, y, event_labels, result_folder):
    """
    Plot ERPs for different conditions.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_channels, n_times)
    - y: numpy array of labels
    - event_labels: list of unique labels
    """
    import matplotlib.pyplot as plt
    
    n_channels = X.shape[1]
    n_times = X.shape[2]
    
    for ch in range(n_channels):
        plt.figure(figsize=(10, 6))
        for label in event_labels:
            idx = np.where(y == label)[0]
            erp = np.mean(X[idx, ch, :], axis=0)
            plt.plot(time, erp, label=f'Label {label}')
        plt.title(f'ERP for Channel {ch+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (μV)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(result_folder, f'ERP_Channel_{ch+1}.png')
        plt.savefig(plot_path)
        plt.close()
