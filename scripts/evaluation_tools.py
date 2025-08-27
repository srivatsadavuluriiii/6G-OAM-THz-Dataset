#!/usr/bin/env python3
"""
Performance Evaluation and Benchmarking Tools for 6G OAM THz Dataset
===================================================================

Comprehensive evaluation metrics and benchmarking utilities for the
6G OAM THz dataset competition. Includes statistical analysis,
visualization tools, and performance comparison frameworks.

Author: 6G Research Team
Version: 1.0
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
import json
import time
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class PerformanceEvaluator:
    """
    Comprehensive performance evaluation toolkit for ML models on 6G OAM THz data.
    """
    
    def __init__(self):
        """Initialize the performance evaluator."""
        self.evaluation_results = {}
        self.benchmark_data = {}
        
    def evaluate_regression_model(self, y_true, y_pred, model_name: str) -> Dict:
        """
        Evaluate regression model performance with comprehensive metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model being evaluated
            
        Returns:
            Dict: Comprehensive evaluation metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic regression metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Adjusted R-squared (assuming p features, n samples)
        n = len(y_true)
        p = 10  # Approximate number of features
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)) * 100
        
        # Maximum error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Explained variance
        explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        # Median absolute error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Prediction intervals (95%)
        errors = y_true - y_pred
        error_std = np.std(errors)
        pi_lower = y_pred - 1.96 * error_std
        pi_upper = y_pred + 1.96 * error_std
        coverage = np.mean((y_true >= pi_lower) & (y_true <= pi_upper))
        
        # Residual analysis
        residuals = y_true - y_pred
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        residual_skewness = stats.skew(residuals)
        residual_kurtosis = stats.kurtosis(residuals)
        
        # Compile results
        results = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2,
            'mape': mape,
            'smape': smape,
            'max_error': max_error,
            'explained_variance': explained_var,
            'median_ae': median_ae,
            'prediction_coverage_95': coverage,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_skewness': residual_skewness,
            'residual_kurtosis': residual_kurtosis,
            'n_samples': len(y_true)
        }
        
        self.evaluation_results[model_name] = results
        return results
        
    def evaluate_multiobjective_solution(self, solution: Dict, objectives: List[str]) -> Dict:
        """
        Evaluate multi-objective optimization solution.
        
        Args:
            solution: Dictionary containing objective values
            objectives: List of objective names
            
        Returns:
            Dict: Multi-objective evaluation metrics
        """
        # Normalize objectives (assuming higher is better for all)
        normalized_objectives = {}
        
        for obj in objectives:
            if obj in solution:
                # Simple min-max normalization (would use reference points in practice)
                if obj == 'throughput':
                    normalized_objectives[obj] = solution[obj] / 100.0  # Max 100 Gbps
                elif obj == 'energy_efficiency':
                    normalized_objectives[obj] = solution[obj] / 1000.0  # Max 1000 bits/J
                elif obj == 'latency':
                    # Lower is better for latency, so invert
                    normalized_objectives[obj] = max(0, 1 - solution[obj] / 100.0)
                else:
                    normalized_objectives[obj] = solution[obj]
                    
        # Calculate overall score (weighted sum)
        weights = {'throughput': 0.4, 'energy_efficiency': 0.3, 'latency': 0.3}
        overall_score = sum(normalized_objectives[obj] * weights.get(obj, 1.0) 
                          for obj in normalized_objectives)
        
        # Calculate dominance metrics
        results = {
            'objectives': solution,
            'normalized_objectives': normalized_objectives,
            'overall_score': overall_score,
            'n_objectives': len(objectives)
        }
        
        return results
        
    def statistical_significance_test(self, model1_scores: List, model2_scores: List, 
                                    alpha: float = 0.05) -> Dict:
        """
        Perform statistical significance test between two models.
        
        Args:
            model1_scores: Performance scores for model 1
            model2_scores: Performance scores for model 2
            alpha: Significance level
            
        Returns:
            Dict: Statistical test results
        """
        # Paired t-test
        statistic, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(model1_scores) - 1) * np.var(model1_scores, ddof=1) +
                             (len(model2_scores) - 1) * np.var(model2_scores, ddof=1)) /
                            (len(model1_scores) + len(model2_scores) - 2))
        
        cohens_d = (np.mean(model1_scores) - np.mean(model2_scores)) / pooled_std
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(model1_scores, model2_scores)
        
        results = {
            'ttest_statistic': statistic,
            'ttest_pvalue': p_value,
            'is_significant': p_value < alpha,
            'cohens_d': cohens_d,
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_pvalue': wilcoxon_p,
            'model1_mean': np.mean(model1_scores),
            'model2_mean': np.mean(model2_scores),
            'difference': np.mean(model1_scores) - np.mean(model2_scores)
        }
        
        return results
        
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
            
    def cross_validation_analysis(self, cv_scores: List, model_name: str) -> Dict:
        """
        Analyze cross-validation results.
        
        Args:
            cv_scores: List of cross-validation scores
            model_name: Name of the model
            
        Returns:
            Dict: CV analysis results
        """
        cv_scores = np.array(cv_scores)
        
        results = {
            'model_name': model_name,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_min': np.min(cv_scores),
            'cv_max': np.max(cv_scores),
            'cv_median': np.median(cv_scores),
            'cv_iqr': np.percentile(cv_scores, 75) - np.percentile(cv_scores, 25),
            'cv_coefficient_variation': np.std(cv_scores) / np.mean(cv_scores) if np.mean(cv_scores) != 0 else 0,
            'cv_scores': cv_scores.tolist()
        }
        
        return results
        
    def benchmark_comparison(self, results_dict: Dict) -> pd.DataFrame:
        """
        Create benchmark comparison table.
        
        Args:
            results_dict: Dictionary of model results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            if isinstance(results, dict) and 'r2' in results:
                comparison_data.append({
                    'Model': model_name,
                    'R²': results['r2'],
                    'RMSE': results['rmse'],
                    'MAE': results['mae'],
                    'MAPE': results['mape'],
                    'Max Error': results['max_error'],
                    'Samples': results['n_samples']
                })
                
        df_comparison = pd.DataFrame(comparison_data)
        
        if not df_comparison.empty:
            # Rank models
            df_comparison['R²_Rank'] = df_comparison['R²'].rank(ascending=False)
            df_comparison['RMSE_Rank'] = df_comparison['RMSE'].rank(ascending=True)
            df_comparison['MAE_Rank'] = df_comparison['MAE'].rank(ascending=True)
            
            # Overall rank (lower is better)
            df_comparison['Overall_Rank'] = (df_comparison['R²_Rank'] + 
                                           df_comparison['RMSE_Rank'] + 
                                           df_comparison['MAE_Rank']) / 3
            
            df_comparison = df_comparison.sort_values('Overall_Rank')
            
        return df_comparison
        
    def plot_model_comparison(self, results_dict: Dict, save_path: str = None):
        """
        Create comprehensive model comparison plots.
        
        Args:
            results_dict: Dictionary of model results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data
        model_names = list(results_dict.keys())
        metrics = ['r2', 'rmse', 'mae', 'mape', 'max_error', 'explained_variance']
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            values = [results_dict[name].get(metric, 0) for name in model_names]
            
            bars = axes[row, col].bar(model_names, values, color=colors, alpha=0.7)
            axes[row, col].set_title(f'{metric.upper()}')
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                                   
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
            
        plt.show()
        
    def plot_residual_analysis(self, y_true, y_pred, model_name: str, save_path: str = None):
        """
        Create residual analysis plots.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        residuals = np.array(y_true) - np.array(y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold')
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        
        axes[1, 1].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual analysis plot saved to {save_path}")
            
        plt.show()
        
    def generate_evaluation_report(self, output_file: str = 'evaluation_report.json'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_file: Output file name
        """
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_models_evaluated': len(self.evaluation_results),
            'evaluation_results': self.evaluation_results,
            'summary_statistics': self._calculate_summary_statistics()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"Evaluation report saved to {output_file}")
        
    def _calculate_summary_statistics(self) -> Dict:
        """Calculate summary statistics across all evaluated models."""
        if not self.evaluation_results:
            return {}
            
        all_r2 = [result['r2'] for result in self.evaluation_results.values()]
        all_rmse = [result['rmse'] for result in self.evaluation_results.values()]
        all_mae = [result['mae'] for result in self.evaluation_results.values()]
        
        return {
            'best_r2': max(all_r2),
            'worst_r2': min(all_r2),
            'mean_r2': np.mean(all_r2),
            'std_r2': np.std(all_r2),
            'best_rmse': min(all_rmse),
            'worst_rmse': max(all_rmse),
            'mean_rmse': np.mean(all_rmse),
            'best_mae': min(all_mae),
            'worst_mae': max(all_mae),
            'mean_mae': np.mean(all_mae)
        }


class CompetitionScorer:
    """
    Official scoring system for the 6G OAM THz dataset competition.
    """
    
    def __init__(self):
        """Initialize competition scorer."""
        self.track_weights = {
            'track1': {'throughput': 1.0},
            'track2': {'throughput': 0.4, 'energy_efficiency': 0.3, 'latency': 0.3},
            'track3': {'episode_reward': 0.6, 'final_performance': 0.4}
        }
        
    def score_track1_submission(self, y_true, y_pred) -> Dict:
        """
        Score Track 1 submission (Throughput Optimization).
        
        Args:
            y_true: True throughput values
            y_pred: Predicted throughput values
            
        Returns:
            Dict: Scoring results
        """
        evaluator = PerformanceEvaluator()
        results = evaluator.evaluate_regression_model(y_true, y_pred, "Track1_Submission")
        
        # Primary score is R²
        primary_score = results['r2']
        
        # Penalty for poor predictions
        penalty = 0
        if results['mape'] > 20:  # MAPE > 20%
            penalty += 0.1
        if results['max_error'] > 50:  # Max error > 50 Gbps
            penalty += 0.1
            
        final_score = max(0, primary_score - penalty)
        
        scoring_result = {
            'track': 'Track 1 - Throughput Optimization',
            'primary_metric': 'R²',
            'primary_score': primary_score,
            'penalty': penalty,
            'final_score': final_score,
            'rank_eligible': final_score > 0.5,  # Minimum threshold
            'detailed_metrics': results
        }
        
        return scoring_result
        
    def score_track2_submission(self, solutions: List[Dict]) -> Dict:
        """
        Score Track 2 submission (Multi-objective Optimization).
        
        Args:
            solutions: List of solution dictionaries with objective values
            
        Returns:
            Dict: Scoring results
        """
        if not solutions:
            return {'error': 'No solutions provided'}
            
        # Calculate scores for each solution
        solution_scores = []
        evaluator = PerformanceEvaluator()
        
        for i, solution in enumerate(solutions):
            result = evaluator.evaluate_multiobjective_solution(
                solution, ['throughput', 'energy_efficiency', 'latency']
            )
            solution_scores.append(result['overall_score'])
            
        # Primary score is mean of best 10 solutions
        top_solutions = sorted(solution_scores, reverse=True)[:10]
        primary_score = np.mean(top_solutions)
        
        # Diversity bonus
        diversity_score = np.std(solution_scores)  # Higher diversity is better
        diversity_bonus = min(0.1, diversity_score / 10)
        
        final_score = primary_score + diversity_bonus
        
        scoring_result = {
            'track': 'Track 2 - Multi-objective Optimization',
            'primary_metric': 'Average Top-10 Score',
            'primary_score': primary_score,
            'diversity_bonus': diversity_bonus,
            'final_score': final_score,
            'n_solutions': len(solutions),
            'best_solution_score': max(solution_scores),
            'rank_eligible': final_score > 0.6
        }
        
        return scoring_result
        
    def score_track3_submission(self, episode_rewards: List, final_performance: Dict) -> Dict:
        """
        Score Track 3 submission (Deep Reinforcement Learning).
        
        Args:
            episode_rewards: List of episode reward sums
            final_performance: Final performance metrics
            
        Returns:
            Dict: Scoring results
        """
        # Episode reward component
        reward_score = np.mean(episode_rewards[-100:])  # Average of last 100 episodes
        normalized_reward_score = max(0, min(1, (reward_score + 10) / 20))  # Normalize to [0,1]
        
        # Final performance component
        throughput_score = final_performance.get('avg_throughput', 0) / 100.0
        efficiency_score = final_performance.get('avg_energy_efficiency', 0) / 1000.0
        latency_score = max(0, 1 - final_performance.get('avg_latency', 50) / 100.0)
        
        performance_score = (0.5 * throughput_score + 
                           0.3 * efficiency_score + 
                           0.2 * latency_score)
        
        # Combine scores
        final_score = (0.6 * normalized_reward_score + 
                      0.4 * performance_score)
        
        # Stability bonus
        reward_stability = 1 / (1 + np.std(episode_rewards[-50:]))
        stability_bonus = min(0.1, reward_stability / 10)
        
        final_score += stability_bonus
        
        scoring_result = {
            'track': 'Track 3 - Deep Reinforcement Learning',
            'primary_metric': 'Combined Score',
            'reward_component': normalized_reward_score,
            'performance_component': performance_score,
            'stability_bonus': stability_bonus,
            'final_score': final_score,
            'episode_performance': {
                'mean_reward': np.mean(episode_rewards),
                'final_100_mean': reward_score,
                'best_episode': max(episode_rewards),
                'convergence_episode': self._find_convergence_episode(episode_rewards)
            },
            'rank_eligible': final_score > 0.5
        }
        
        return scoring_result
        
    def _find_convergence_episode(self, rewards: List, window: int = 50) -> int:
        """Find episode where rewards converged."""
        if len(rewards) < window * 2:
            return len(rewards)
            
        for i in range(window, len(rewards) - window):
            before_mean = np.mean(rewards[i-window:i])
            after_mean = np.mean(rewards[i:i+window])
            
            if abs(after_mean - before_mean) < 0.1:  # Convergence threshold
                return i
                
        return len(rewards)


def example_evaluation_workflow():
    """
    Example workflow demonstrating the evaluation tools.
    """
    print("6G OAM THz Dataset - Evaluation Workflow Example")
    print("=" * 55)
    
    # Initialize evaluator
    evaluator = PerformanceEvaluator()
    
    # Generate synthetic evaluation data
    np.random.seed(42)
    n_samples = 1000
    
    # True values (synthetic throughput data)
    y_true = np.random.lognormal(mean=3, sigma=0.5, size=n_samples)
    
    # Simulate different model predictions
    models = {
        'Random_Forest': y_true + np.random.normal(0, 2, n_samples),
        'XGBoost': y_true + np.random.normal(0, 1.5, n_samples),
        'Linear_Regression': y_true + np.random.normal(0, 3, n_samples),
        'Neural_Network': y_true + np.random.normal(0, 1.8, n_samples)
    }
    
    # Evaluate all models
    print("\nEvaluating models...")
    for model_name, y_pred in models.items():
        result = evaluator.evaluate_regression_model(y_true, y_pred, model_name)
        print(f"{model_name}: R² = {result['r2']:.3f}, RMSE = {result['rmse']:.3f}")
    
    # Create comparison
    comparison_df = evaluator.benchmark_comparison(evaluator.evaluation_results)
    print("\nModel Ranking:")
    print(comparison_df[['Model', 'R²', 'RMSE', 'MAE', 'Overall_Rank']])
    
    # Statistical significance test
    rf_scores = [0.85, 0.87, 0.86, 0.88, 0.84]  # Simulated CV scores
    xgb_scores = [0.82, 0.84, 0.83, 0.85, 0.81]
    
    sig_test = evaluator.statistical_significance_test(rf_scores, xgb_scores)
    print(f"\nStatistical Test (RF vs XGB):")
    print(f"P-value: {sig_test['ttest_pvalue']:.4f}")
    print(f"Significant: {sig_test['is_significant']}")
    print(f"Effect size: {sig_test['effect_size_interpretation']}")
    
    # Generate plots
    evaluator.plot_model_comparison(evaluator.evaluation_results, 'model_comparison.png')
    evaluator.plot_residual_analysis(y_true, models['XGBoost'], 'XGBoost', 'residual_analysis.png')
    
    # Competition scoring example
    scorer = CompetitionScorer()
    
    # Track 1 scoring
    track1_score = scorer.score_track1_submission(y_true, models['XGBoost'])
    print(f"\nTrack 1 Score: {track1_score['final_score']:.3f}")
    
    # Track 2 scoring (synthetic multi-objective solutions)
    mo_solutions = []
    for _ in range(50):
        solution = {
            'throughput': np.random.uniform(20, 80),
            'energy_efficiency': np.random.uniform(200, 800),
            'latency': np.random.uniform(2, 25)
        }
        mo_solutions.append(solution)
        
    track2_score = scorer.score_track2_submission(mo_solutions)
    print(f"Track 2 Score: {track2_score['final_score']:.3f}")
    
    # Track 3 scoring (synthetic RL results)
    episode_rewards = np.cumsum(np.random.normal(0.1, 1, 500))  # Improving rewards
    final_perf = {
        'avg_throughput': 65.0,
        'avg_energy_efficiency': 600.0,
        'avg_latency': 8.0
    }
    
    track3_score = scorer.score_track3_submission(episode_rewards, final_perf)
    print(f"Track 3 Score: {track3_score['final_score']:.3f}")
    
    # Generate evaluation report
    evaluator.generate_evaluation_report('evaluation_report.json')
    
    print("\nEvaluation workflow completed!")
    print("Generated files:")
    print("- model_comparison.png")
    print("- residual_analysis.png") 
    print("- evaluation_report.json")


if __name__ == "__main__":
    example_evaluation_workflow()