# PCA Analysis for RanPAC Features_h

This directory contains tools for performing Principal Component Analysis (PCA) on the Features_h extracted during RanPAC experiments.

## Overview

The RanPAC implementation has been enhanced to automatically save `Features_h` (the transformed features after random projection) for each task. These saved features can then be analyzed using PCA to understand:

- Feature variance distribution across tasks
- Dimensionality patterns per task
- Cross-task similarity in feature representations
- 2D visualizations using PCA and t-SNE

## Files Added/Modified

### Modified Files
- `RanPAC.py`: Added `_save_features_h()` method to automatically save Features_h and labels per task

### New Files
- `pca_analysis.py`: Main PCA analysis script
- `run_pca_analysis.py`: Convenience script for running analysis on multiple experiments
- `PCA_Analysis_README.md`: This documentation file

## How Features_h are Saved

During RanPAC training, when the `replace_fc()` method is called for each task, the following files are automatically saved:

```
logs/{model_name}/{dataset}/{init_cls}/{increment}/{exp_name}/features_h/
├── task_0_features_h.npy  # Features_h for task 0
├── task_0_labels.npy      # Corresponding labels for task 0
├── task_1_features_h.npy  # Features_h for task 1
├── task_1_labels.npy      # Corresponding labels for task 1
└── ...
```

- **Features_h**: For RP methods (M > 0), this is `ReLU(Features_f @ W_rand)`. For non-RP methods, this is just `Features_f`.
- **Labels**: The corresponding class labels for each sample.

## Usage

### Prerequisites

Install required dependencies:
```bash
pip install matplotlib seaborn scikit-learn pandas
```

### Basic Usage

1. **Run a RanPAC experiment** (Features_h will be saved automatically):
```bash
python main.py --config args/your_config.json
```

2. **Analyze the saved features**:
```bash
# Analyze a specific experiment (with automatic caching)
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment

# With custom number of PCA components
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment --n_components 100

# Force recompute PCA (ignore cache)
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment --force_recompute

# Only generate plots from existing cache (fast)
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment --plot_only

# Disable caching completely
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment --no_cache

# Save results to custom directory
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment --output_dir /path/to/results
```

3. **Analyze multiple experiments**:
```bash
# Using the convenience script (with caching)
python run_pca_analysis.py logs/ncm/cifar224/0/10/experiment1 logs/adapter/cifar224/0/10/experiment2

# Recursively find and analyze all experiments in a directory
python run_pca_analysis.py --recursive logs/ncm/cifar224/

# Use shared cache for multiple experiments (memory efficient)
python run_pca_analysis.py --shared_cache --output_base ./results logs/exp1 logs/exp2

# Force recompute all PCA results
python run_pca_analysis.py --force_recompute --recursive logs/ncm/cifar224/

# Only regenerate plots from existing caches (very fast)
python run_pca_analysis.py --plot_only --recursive logs/ncm/cifar224/
```

## 🚀 PCA Result Caching

**New Feature**: The PCA analysis now supports intelligent caching to avoid time-consuming recomputation!

### How Caching Works

1. **Automatic Caching**: PCA results are automatically saved to `pca_cache.pkl` in the output directory
2. **Cache Validation**: The system checks if cached results are still valid:
   - Same number of PCA components
   - Same task structure (task IDs)
   - Same feature dimensions per task
3. **Smart Loading**: Valid cache is automatically loaded, invalid cache triggers recomputation

### Cache Benefits

- **Time Savings**: Skip PCA computation (can save minutes/hours for large datasets)
- **Plot Iteration**: Quickly adjust plot parameters without recomputing PCA
- **Experimentation**: Try different visualization approaches efficiently

### Cache Control Options

- `--force_recompute`: Ignore cache and recompute PCA
- `--no_cache`: Disable caching completely
- `--plot_only`: Only generate plots from existing cache
- `--cache_file`: Specify custom cache file location
- `--shared_cache`: Use shared cache for multiple experiments

### Example Workflow

```bash
# First run: Compute PCA and cache results (takes time)
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment --n_components 50

# Subsequent runs: Load from cache and regenerate plots (fast!)
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment --plot_only

# Adjust number of components: Will recompute and update cache
python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/my_experiment --n_components 100
```

### Output

The PCA analysis generates the following outputs:

#### Plots
1. **`pca_explained_variance.png`**: Four-panel plot showing:
   - Explained variance ratio per component
   - Cumulative explained variance
   - Singular values (log scale)
   - Variance captured comparison between tasks

2. **`pca_tsne_2d_embeddings.png`**: 2D visualizations showing:
   - PCA projections (first 2 components) for each task
   - t-SNE embeddings for each task
   - Color-coded by class labels

3. **`cross_task_similarity.png`**: Heatmap showing similarity between tasks based on their PCA subspaces

#### Analysis Files
- **`pca_analysis_summary.txt`**: Text summary with key statistics for each task
- **`pca_cache.pkl`**: Cached PCA results for fast reloading (binary file)
- **Individual `.npy` files**: Can be loaded for further custom analysis

### Example Analysis Summary

```
============================================================
PCA ANALYSIS SUMMARY
============================================================

TASK 0:
  Original features shape: (500, 768)
  Number of classes: 10
  Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  PCA components: 50
  First PC explains: 0.1234 of variance
  Top 5 PCs explain: 0.4567 of variance
  Top 10 PCs explain: 0.6789 of variance

TASK 1:
  Original features shape: (500, 768)
  Number of classes: 10
  Classes: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  PCA components: 50
  First PC explains: 0.1123 of variance
  Top 5 PCs explain: 0.4321 of variance
  Top 10 PCs explain: 0.6543 of variance

CROSS-TASK SIMILARITY:
  Task 0 vs Task 1: 0.7234
  Task 0 vs Task 2: 0.6891
  Task 1 vs Task 2: 0.7456
```

## Implementation Details

### Features_h Computation
- **RP methods** (M > 0): `Features_h = ReLU(Features_f @ W_rand)`
- **Non-RP methods** (M = 0): `Features_h = Features_f`

### Cross-Task Similarity
Computed using the cosine similarity between PCA subspaces:
1. Extract top-k PCA components for each task
2. Compute cross-correlation matrix between component sets
3. Use mean of maximum correlations as similarity measure

### Error Handling
- Graceful handling of missing files
- Automatic adjustment of PCA components based on data shape
- Warning messages for incomplete experiments

## Advanced Usage

### Custom Analysis
You can load the saved numpy arrays directly for custom analysis:

```python
import numpy as np

# Load features and labels for task 0
features = np.load('logs/experiment/features_h/task_0_features_h.npy')
labels = np.load('logs/experiment/features_h/task_0_labels.npy')

# Perform custom analysis
# ...
```

### Integration with Other Tools
The saved `.npy` files can be easily integrated with other analysis tools like:
- TensorBoard projector
- Custom matplotlib/seaborn visualizations
- Scikit-learn clustering algorithms
- Custom dimensionality reduction techniques

## Troubleshooting

### Common Issues
1. **No features_h directory found**: Make sure you've run a RanPAC experiment that reached the `replace_fc()` stage
2. **Permission errors**: Ensure write permissions for the logs directory
3. **Memory issues**: For large datasets, consider reducing `n_components` or sampling fewer points for t-SNE

### Debugging
Check the RanPAC training logs for messages like:
```
Saved Features_h for task 0: shape (500, 768)
Saved to: logs/ncm/cifar224/0/10/my_experiment/features_h/task_0_features_h.npy
```

## Future Enhancements

Potential future additions:
- Interactive visualizations with plotly
- Clustering analysis
- Feature importance analysis
- Comparison across different methods/datasets
- Integration with wandb or tensorboard 

# Combined Features Analysis

In addition to per-task analysis, the enhanced PCA analysis now includes **combined features analysis** that evaluates the overall quality of features across all tasks together. This provides insights into:

## Combined Linear Separability
- **Global Classification**: How well linear classifiers perform on the entire feature space
- **Task Separability**: Whether features can distinguish which task samples came from (important for catastrophic forgetting)
- **Inter-task Class Analysis**: For classes appearing in multiple tasks, how task-specific their representations are

## Combined Gaussian Assessment
- **Global Distribution Quality**: Whether the overall feature space follows Gaussian assumptions
- **Cross-task Normality**: Distribution quality when all task features are combined

## Key Insights

### Task Separability Analysis
- **High Task Separability** (>random baseline + 0.2): Potential catastrophic forgetting
- **Moderate Task Separability**: Monitor for forgetting effects  
- **Low Task Separability**: Good feature sharing across tasks

### Combined vs Individual Performance
- Compare individual task performance with combined performance
- Identify whether combining tasks improves or degrades separability
- Understand global feature space quality

## Additional Visualizations

The analysis generates `combined_features_analysis.png` with 6 panels:
1. **Individual vs Combined Separability**: Bar chart comparing SVM accuracy
2. **Task Prediction Analysis**: How well features can predict task origin
3. **Combined Classifier Performance**: SVM, Logistic Regression, and LDA accuracies
4. **Gaussian Assumption Comparison**: Individual vs combined normality tests
5. **Per-Class Task Separability**: Scatter plot of classes across tasks
6. **Overall Quality Summary**: Normalized comparison of all metrics

## Usage

The combined analysis runs automatically with existing commands:

```bash
python pca_analysis.py --exp_path logs/model_name/dataset/init_cls/increment/exp_name
```

Results are included in the comprehensive analysis summary with specific recommendations for:
- Global classifier selection
- Catastrophic forgetting risk assessment
- Cross-task feature quality evaluation 