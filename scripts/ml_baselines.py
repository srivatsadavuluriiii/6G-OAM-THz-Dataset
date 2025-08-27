#!/usr/bin/env python3
"""
Machine Learning Baseline Models for 6G OAM THz Dataset
======================================================

Comprehensive baseline implementations for the 6G OAM THz dataset competition.
Includes models for all three competition tracks:
- Track 1: Throughput Optimization
- Track 2: Multi-objective Optimization  
- Track 3: Deep Reinforcement Learning

Author: 6G Research Team
Version: 1.0
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    """
    Baseline machine learning models for 6G OAM THz dataset analysis.
    """
    
    def __init__(self):
        """Initialize baseline models with default configurations."""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'svr': SVR(kernel='rbf'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        self.results = {}
        self.trained_models = {}
        
    def train_model(self, model_name, X_train, y_train, X_test, y_test):
        """
        Train a specific model and evaluate performance.
        
        Args:
            model_name (str): Name of the model to train
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Performance metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'model': model
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        self.results[model_name] = metrics
        self.trained_models[model_name] = model
        
        return metrics
        
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all baseline models and compare performance.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            pd.DataFrame: Comparison of all model performances
        """
        print("Training baseline models...")
        
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            try:
                self.train_model(model_name, X_train, y_train, X_test, y_test)
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                
        return self.get_results_summary()
        
    def get_results_summary(self):
        """
        Get summary of all model results.
        
        Returns:
            pd.DataFrame: Summary of model performances
        """
        summary_data = []
        
        for model_name, metrics in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Test R²': metrics['test_r2'],
                'Test MSE': metrics['test_mse'],
                'Test MAE': metrics['test_mae'],
                'CV R² Mean': metrics['cv_r2_mean'],
                'CV R² Std': metrics['cv_r2_std']
            })
            
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Test R²', ascending=False)
        
        return df_summary
        
    def plot_model_comparison(self, save_path=None):
        """
        Create visualization comparing model performances.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.results:
            print("No results to plot. Train models first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        model_names = list(self.results.keys())
        test_r2 = [self.results[name]['test_r2'] for name in model_names]
        test_mse = [self.results[name]['test_mse'] for name in model_names]
        test_mae = [self.results[name]['test_mae'] for name in model_names]
        cv_r2_mean = [self.results[name]['cv_r2_mean'] for name in model_names]
        
        # R² scores
        axes[0, 0].bar(model_names, test_r2, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Test R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MSE scores
        axes[0, 1].bar(model_names, test_mse, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Test MSE Scores')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE scores
        axes[1, 0].bar(model_names, test_mae, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Test MAE Scores')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Cross-validation R²
        axes[1, 1].bar(model_names, cv_r2_mean, color='gold', alpha=0.7)
        axes[1, 1].set_title('Cross-Validation R² Scores')
        axes[1, 1].set_ylabel('CV R² Mean')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
            
        plt.show()
        
    def hyperparameter_optimization(self, model_name, X_train, y_train, param_grid):
        """
        Perform hyperparameter optimization for a specific model.
        
        Args:
            model_name (str): Name of the model
            X_train, y_train: Training data
            param_grid (dict): Parameter grid for optimization
            
        Returns:
            GridSearchCV: Fitted grid search object
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for Track 2 of the competition.
    Optimizes multiple performance metrics simultaneously.
    """
    
    def __init__(self, objectives=['throughput', 'energy_efficiency', 'latency']):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives (list): List of objectives to optimize
        """
        self.objectives = objectives
        self.pareto_solutions = []
        
    def evaluate_solution(self, params, dataset):
        """
        Evaluate a solution across multiple objectives.
        
        Args:
            params (dict): Parameter configuration
            dataset (pd.DataFrame): Dataset for evaluation
            
        Returns:
            dict: Objective values
        """
        # Simplified objective evaluation
        # In practice, this would involve running simulations or trained models
        
        objectives = {}
        
        # Throughput objective (maximize)
        throughput_score = self._calculate_throughput_score(params, dataset)
        objectives['throughput'] = throughput_score
        
        # Energy efficiency objective (maximize)
        energy_score = self._calculate_energy_efficiency_score(params, dataset)
        objectives['energy_efficiency'] = energy_score
        
        # Latency objective (minimize - convert to maximization)
        latency_score = self._calculate_latency_score(params, dataset)
        objectives['latency'] = -latency_score  # Negative for minimization
        
        return objectives
        
    def _calculate_throughput_score(self, params, dataset):
        """Calculate throughput score based on parameters."""
        # Simplified scoring function
        base_score = 0.5
        freq_factor = (params.get('frequency', 400) - 300) / 300  # Normalize frequency
        power_factor = (params.get('tx_power', 20) - 10) / 20  # Normalize power
        
        return base_score + 0.3 * freq_factor + 0.2 * power_factor
        
    def _calculate_energy_efficiency_score(self, params, dataset):
        """Calculate energy efficiency score based on parameters."""
        # Simplified scoring function
        base_score = 0.6
        power_penalty = (params.get('tx_power', 20) - 10) / 50  # Higher power = lower efficiency
        distance_factor = (params.get('distance', 100) - 50) / 150
        
        return base_score - 0.3 * power_penalty + 0.1 * distance_factor
        
    def _calculate_latency_score(self, params, dataset):
        """Calculate latency score based on parameters."""
        # Simplified scoring function (lower is better)
        base_latency = 5.0
        distance_penalty = (params.get('distance', 100) - 50) / 100
        freq_benefit = (params.get('frequency', 400) - 300) / 300
        
        return base_latency + 2.0 * distance_penalty - 1.0 * freq_benefit
        
    def find_pareto_front(self, parameter_space, dataset, n_solutions=100):
        """
        Find Pareto optimal solutions.
        
        Args:
            parameter_space (dict): Parameter ranges
            dataset (pd.DataFrame): Dataset for evaluation
            n_solutions (int): Number of solutions to evaluate
            
        Returns:
            list: Pareto optimal solutions
        """
        solutions = []
        
        # Generate random solutions
        for _ in range(n_solutions):
            params = {}
            for param, (min_val, max_val) in parameter_space.items():
                params[param] = np.random.uniform(min_val, max_val)
                
            objectives = self.evaluate_solution(params, dataset)
            solutions.append({'params': params, 'objectives': objectives})
            
        # Find Pareto optimal solutions
        pareto_solutions = []
        
        for i, solution_i in enumerate(solutions):
            is_dominated = False
            
            for j, solution_j in enumerate(solutions):
                if i != j:
                    # Check if solution_j dominates solution_i
                    dominates = True
                    for obj in self.objectives:
                        if solution_j['objectives'][obj] <= solution_i['objectives'][obj]:
                            dominates = False
                            break
                            
                    if dominates:
                        is_dominated = True
                        break
                        
            if not is_dominated:
                pareto_solutions.append(solution_i)
                
        self.pareto_solutions = pareto_solutions
        return pareto_solutions
        
    def plot_pareto_front(self, save_path=None):
        """
        Visualize the Pareto front.
        
        Args:
            save_path (str): Path to save the plot
        """
        if len(self.pareto_solutions) == 0:
            print("No Pareto solutions found. Run find_pareto_front first.")
            return
            
        if len(self.objectives) >= 2:
            # Plot 2D Pareto front
            obj1 = self.objectives[0]
            obj2 = self.objectives[1]
            
            x_vals = [sol['objectives'][obj1] for sol in self.pareto_solutions]
            y_vals = [sol['objectives'][obj2] for sol in self.pareto_solutions]
            
            plt.figure(figsize=(10, 8))
            plt.scatter(x_vals, y_vals, c='red', s=50, alpha=0.7, label='Pareto Optimal')
            plt.xlabel(f'{obj1.capitalize()} Score')
            plt.ylabel(f'{obj2.capitalize()} Score')
            plt.title('Pareto Front - Multi-Objective Optimization')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Pareto front plot saved to {save_path}")
                
            plt.show()


def track1_throughput_optimization_example():
    """
    Example implementation for Track 1: Throughput Optimization
    """
    print("=== Track 1: Throughput Optimization Example ===")
    
    # Load sample data (replace with actual data loading)
    from dataset_loader import OAMDatasetLoader
    
    loader = OAMDatasetLoader()
    datasets = loader.load_all_datasets()
    
    if 'comprehensive' not in datasets:
        print("Comprehensive dataset not found. Using synthetic data for example.")
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'Frequency_GHz': np.random.uniform(300, 600, n_samples),
            'TX_Power_dBm': np.random.uniform(10, 30, n_samples),
            'Distance_m': np.random.uniform(50, 200, n_samples),
            'OAM_Mode': np.random.randint(1, 8, n_samples),
            'SNR_dB': np.random.uniform(5, 25, n_samples)
        })
        
        # Synthetic throughput based on realistic relationships
        y = (X['SNR_dB'] * 0.8 + 
             (X['Frequency_GHz'] - 300) / 300 * 5 +
             (30 - X['TX_Power_dBm']) / 20 * 2 +
             np.random.normal(0, 1, n_samples))
        
        # Preprocess data
        preprocessed = loader.preprocess_dataset(
            pd.concat([X, pd.DataFrame({'Throughput_Gbps': y})], axis=1)
        )
    else:
        preprocessed = loader.preprocess_dataset(datasets['comprehensive'])
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # Train all models
    results = baseline.train_all_models(
        preprocessed['X_train'], preprocessed['y_train'],
        preprocessed['X_test'], preprocessed['y_test']
    )
    
    print("\nModel Performance Summary:")
    print(results)
    
    # Create visualization
    baseline.plot_model_comparison('track1_model_comparison.png')
    
    return baseline, results


def track2_multiobjective_example():
    """
    Example implementation for Track 2: Multi-objective Optimization
    """
    print("\n=== Track 2: Multi-objective Optimization Example ===")
    
    # Define parameter space
    parameter_space = {
        'frequency': (300, 600),      # GHz
        'tx_power': (10, 30),         # dBm
        'distance': (50, 200),        # meters
        'oam_mode': (1, 8)            # OAM mode
    }
    
    # Create dummy dataset for optimization
    dummy_dataset = pd.DataFrame()  # In practice, use real dataset
    
    # Initialize optimizer
    optimizer = MultiObjectiveOptimizer()
    
    # Find Pareto optimal solutions
    pareto_solutions = optimizer.find_pareto_front(parameter_space, dummy_dataset)
    
    print(f"Found {len(pareto_solutions)} Pareto optimal solutions")
    
    # Display top 5 solutions
    print("\nTop 5 Pareto Optimal Solutions:")
    for i, solution in enumerate(pareto_solutions[:5]):
        print(f"Solution {i+1}:")
        print(f"  Parameters: {solution['params']}")
        print(f"  Objectives: {solution['objectives']}")
        print()
    
    # Visualize Pareto front
    optimizer.plot_pareto_front('track2_pareto_front.png')
    
    return optimizer, pareto_solutions


def main():
    """
    Main function demonstrating all baseline implementations.
    """
    print("6G OAM THz Dataset - Baseline Models Demo")
    print("=" * 50)
    
    # Track 1: Throughput Optimization
    baseline, results = track1_throughput_optimization_example()
    
    # Track 2: Multi-objective Optimization
    optimizer, pareto_solutions = track2_multiobjective_example()
    
    print("\n=== Summary ===")
    print(f"Track 1: Trained {len(baseline.results)} baseline models")
    print(f"Track 2: Found {len(pareto_solutions)} Pareto optimal solutions")
    print("\nBaseline implementations completed successfully!")
    print("\nGenerated files:")
    print("- track1_model_comparison.png")
    print("- track2_pareto_front.png")


if __name__ == "__main__":
    main()