# 6G OAM THz Dataset - Setup and Usage Guide

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv oam_6g_env
source oam_6g_env/bin/activate  # On Windows: oam_6g_env\Scripts\activate

# Install dependencies
pip install -r competition_requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn; print('Setup successful!')"
```

### 2. Load and Explore Dataset
```python
from dataset_loader import OAMDatasetLoader

# Initialize loader
loader = OAMDatasetLoader()

# Load all datasets
datasets = loader.load_all_datasets()

# Quick overview
for name, df in datasets.items():
    print(f"{name}: {df.shape[0]} samples, {df.shape[1]} features")

# Generate visualizations
loader.visualize_dataset_overview(datasets, 'dataset_overview.png')
```

### 3. Train Baseline Models (Track 1)
```python
from ml_baselines import BaselineModels

# Preprocess data
preprocessed = loader.preprocess_dataset(datasets['comprehensive'])

# Train baseline models
baseline = BaselineModels()
results = baseline.train_all_models(
    preprocessed['X_train'], preprocessed['y_train'],
    preprocessed['X_test'], preprocessed['y_test']
)

print(results)
```

### 4. Multi-objective Optimization (Track 2)
```python
from ml_baselines import MultiObjectiveOptimizer

# Define parameter space
parameter_space = {
    'frequency': (300, 600),      # GHz
    'tx_power': (10, 30),         # dBm
    'distance': (50, 200),        # meters
    'oam_mode': (1, 8)            # OAM mode
}

# Initialize optimizer
optimizer = MultiObjectiveOptimizer()

# Find Pareto optimal solutions
pareto_solutions = optimizer.find_pareto_front(parameter_space, datasets['comprehensive'])
print(f"Found {len(pareto_solutions)} Pareto optimal solutions")

# Visualize results
optimizer.plot_pareto_front('pareto_front.png')
```

### 5. Deep Reinforcement Learning (Track 3)
```python
from drl_environment import OAMTHzEnv

# Create environment
env = OAMTHzEnv(scenario='comprehensive')

# Test environment
state = env.reset()
for step in range(100):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    if done:
        break

# Get performance summary
summary = env.get_performance_summary()
print(f"Average reward: {summary['avg_reward']:.2f}")

# Train with stable-baselines3 (if installed)
from stable_baselines3 import PPO
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
```

### 6. Evaluate Results
```python
from evaluation_tools import PerformanceEvaluator, CompetitionScorer

# Initialize evaluator
evaluator = PerformanceEvaluator()

# Evaluate your model
results = evaluator.evaluate_regression_model(y_true, y_pred, "My_Model")

# Compare with baselines
comparison = evaluator.benchmark_comparison(all_results)

# Score for competition
scorer = CompetitionScorer()
track1_score = scorer.score_track1_submission(y_true, y_pred)
print(f"Competition Score: {track1_score['final_score']:.3f}")
```

## File Descriptions

- **dataset_loader.py**: Comprehensive dataset loading and preprocessing utilities
- **ml_baselines.py**: Baseline machine learning models for all competition tracks
- **drl_environment.py**: OpenAI Gym environment for reinforcement learning
- **evaluation_tools.py**: Performance evaluation and competition scoring tools
- **competition_requirements.txt**: Complete list of Python dependencies

## Competition Tracks

### Track 1: Throughput Optimization
- **Goal**: Maximize data throughput prediction accuracy
- **Metric**: R² score with penalties for large errors
- **Baseline**: XGBoost, Random Forest, Neural Networks

### Track 2: Multi-objective Optimization
- **Goal**: Optimize throughput, energy efficiency, and latency simultaneously
- **Metric**: Pareto front quality and solution diversity
- **Baseline**: Genetic algorithms, PSO, NSGA-II

### Track 3: Deep Reinforcement Learning
- **Goal**: Learn optimal parameter policies through interaction
- **Metric**: Episode rewards and final performance
- **Environment**: Custom OpenAI Gym environment

## Advanced Usage

### Custom Model Integration
```python
# Example: Add your custom model
class MyCustomModel:
    def fit(self, X, y):
        # Your training code
        pass
    
    def predict(self, X):
        # Your prediction code
        pass

# Integrate with evaluation framework
baseline.models['my_model'] = MyCustomModel()
```

### Hyperparameter Optimization
```python
# Example parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Optimize hyperparameters
best_model = baseline.hyperparameter_optimization('xgboost', X_train, y_train, param_grid)
```

### Custom Environment Scenarios
```python
# Create environment for specific scenario
env_urban = OAMTHzEnv(scenario='outdoor_urban')
env_indoor = OAMTHzEnv(scenario='indoor_realistic')
env_mobility = OAMTHzEnv(scenario='high_mobility')
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Use data chunking for large datasets
```python
# Process data in chunks
for chunk in pd.read_csv('large_dataset.csv', chunksize=10000):
    process_chunk(chunk)
```

2. **Package Conflicts**: Use virtual environment
```bash
pip install --upgrade pip
pip install --force-reinstall -r competition_requirements.txt
```

3. **CUDA Issues**: Install GPU-specific packages
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Performance Tips

1. **Data Loading**: Use pickle for faster loading
```python
# Save preprocessed data
import pickle
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(preprocessed, f)

# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    preprocessed = pickle.load(f)
```

2. **Model Training**: Use parallel processing
```python
from sklearn.ensemble import RandomForestRegressor

# Enable parallel processing
rf = RandomForestRegressor(n_jobs=-1)  # Use all CPU cores
```

3. **Memory Optimization**: Use appropriate data types
```python
# Optimize memory usage
df['OAM_Mode'] = df['OAM_Mode'].astype('int8')
df['Frequency_GHz'] = df['Frequency_GHz'].astype('float32')
```

## Competition Guidelines

### Submission Format

1. **Track 1**: Submit predictions as CSV with columns ['sample_id', 'predicted_throughput']
2. **Track 2**: Submit Pareto solutions as JSON with objective values
3. **Track 3**: Submit trained model and performance logs

### Evaluation Criteria

- **Reproducibility**: Include random seeds and environment details
- **Documentation**: Provide clear method description
- **Code Quality**: Follow PEP 8 style guidelines
- **Innovation**: Novel approaches receive bonus points

### Timeline

- **Phase 1 (Weeks 1-2)**: Dataset exploration and baseline implementation
- **Phase 2 (Weeks 3-5)**: Model development and optimization
- **Phase 3 (Weeks 6-7)**: Final evaluation and documentation
- **Phase 4 (Week 8)**: Submission and peer review

## Support and Resources

- **Documentation**: See README_DATASET.md for detailed dataset description
- **Issues**: Report bugs and ask questions on GitHub
- **Papers**: Cite relevant 6G and OAM research in your submissions
- **Community**: Join discussions in competition forum

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{6g_oam_thz_2025,
  title={6G OAM THz Communication Dataset for Machine Learning},
  author={6G Research Team},
  year={2025},
  publisher={IEEE DataPort},
  url={https://github.com/your-repo/6G-OAM-THz-Dataset}
}
```