#!/usr/bin/env python3
"""
6G OAM THz Dataset Loader and Preprocessing Utilities
====================================================

Comprehensive data loading and preprocessing tools for the 6G OAM THz dataset.
Supports all dataset variants: comprehensive, lab_controlled, outdoor_urban, 
indoor_realistic, and high_mobility scenarios.

Author: 6G Research Team
Version: 1.0
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import os
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OAMDatasetLoader:
    """
    Comprehensive loader for 6G OAM THz datasets with preprocessing capabilities.
    """
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir (str): Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.datasets = {}
        self.scalers = {}
        
        # Define expected columns for validation
        self.expected_columns = [
            'Frequency_GHz', 'OAM_Mode', 'Beam_Divergence_mrad', 'Aperture_Size_m',
            'TX_Power_dBm', 'RX_Sensitivity_dBm', 'Distance_m', 'Elevation_Angle_deg',
            'Azimuth_Angle_deg', 'Atmospheric_Attenuation_dB_per_km', 'Rain_Rate_mm_per_hr',
            'Humidity_percent', 'Temperature_C', 'Wind_Speed_m_per_s', 'Pressure_hPa',
            'Multipath_Components', 'Doppler_Shift_Hz', 'Phase_Noise_dBc_per_Hz',
            'Antenna_Gain_dBi', 'VSWR', 'Polarization_Loss_dB', 'Pointing_Error_deg',
            'Jitter_deg', 'Scintillation_Index', 'Coherence_Time_ms', 'Coherence_BW_MHz',
            'Channel_Capacity_bps_per_Hz', 'SNR_dB', 'BER', 'Throughput_Gbps',
            'Latency_ms', 'Packet_Loss_Rate', 'Energy_Efficiency_bits_per_Joule'
        ]
        
    def load_dataset(self, filename: str, scenario: str = None) -> pd.DataFrame:
        """
        Load a specific dataset file.
        
        Args:
            filename (str): Name of the CSV file to load
            scenario (str): Scenario name for identification
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {filename}: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Validate columns
            missing_cols = set(self.expected_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns in {filename}: {missing_cols}")
                
            if scenario:
                self.datasets[scenario] = df
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
            
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available dataset files.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of all loaded datasets
        """
        dataset_files = {
            'comprehensive': 'comprehensive_dataset.csv',
            'lab_controlled': 'lab_controlled_comprehensive.csv',
            'outdoor_urban': 'outdoor_urban_comprehensive.csv',
            'indoor_realistic': 'indoor_realistic_comprehensive.csv',
            'high_mobility': 'high_mobility_comprehensive.csv'
        }
        
        loaded_datasets = {}
        
        for scenario, filename in dataset_files.items():
            try:
                df = self.load_dataset(filename, scenario)
                loaded_datasets[scenario] = df
                logger.info(f"Successfully loaded {scenario} dataset")
            except FileNotFoundError:
                logger.warning(f"Dataset file {filename} not found, skipping...")
            except Exception as e:
                logger.error(f"Failed to load {scenario} dataset: {str(e)}")
                
        return loaded_datasets
        
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive statistics for a dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            Dict: Statistics dictionary
        """
        stats = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numerical_stats': df.describe().to_dict(),
            'categorical_stats': {}
        }
        
        # Categorical column analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            stats['categorical_stats'][col] = df[col].value_counts().to_dict()
            
        return stats
        
    def preprocess_dataset(self, df: pd.DataFrame, 
                          scaler_type: str = 'standard',
                          target_column: str = 'Throughput_Gbps',
                          test_size: float = 0.2) -> Dict:
        """
        Preprocess dataset for machine learning.
        
        Args:
            df (pd.DataFrame): Dataset to preprocess
            scaler_type (str): Type of scaler ('standard', 'minmax', 'none')
            target_column (str): Name of target variable
            test_size (float): Proportion of data for testing
            
        Returns:
            Dict: Preprocessed data splits
        """
        # Handle missing values
        df_clean = df.dropna()
        logger.info(f"Removed {len(df) - len(df_clean)} rows with missing values")
        
        # Separate features and target
        if target_column not in df_clean.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Select only numerical features
        numerical_features = X.select_dtypes(include=[np.number]).columns
        X_numerical = X[numerical_features]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_numerical, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = None
            
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': list(X_train.columns)
        }
        
    def visualize_dataset_overview(self, datasets: Dict[str, pd.DataFrame], 
                                  save_path: str = None):
        """
        Create comprehensive visualization of dataset characteristics.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of datasets
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('6G OAM THz Dataset Overview', fontsize=16, fontweight='bold')
        
        # Dataset sizes
        scenario_names = list(datasets.keys())
        dataset_sizes = [len(datasets[scenario]) for scenario in scenario_names]
        
        axes[0, 0].bar(scenario_names, dataset_sizes, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Dataset Sizes')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput distributions
        axes[0, 1].hist([datasets[scenario]['Throughput_Gbps'] for scenario in scenario_names], 
                       label=scenario_names, alpha=0.7, bins=30)
        axes[0, 1].set_title('Throughput Distribution by Scenario')
        axes[0, 1].set_xlabel('Throughput (Gbps)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Frequency range coverage
        for i, scenario in enumerate(scenario_names):
            axes[0, 2].scatter(datasets[scenario]['Frequency_GHz'], 
                             datasets[scenario]['Throughput_Gbps'], 
                             alpha=0.5, label=scenario, s=1)
        axes[0, 2].set_title('Frequency vs Throughput')
        axes[0, 2].set_xlabel('Frequency (GHz)')
        axes[0, 2].set_ylabel('Throughput (Gbps)')
        axes[0, 2].legend()
        
        # OAM mode distribution
        if len(scenario_names) > 0:
            sample_dataset = datasets[scenario_names[0]]
            oam_counts = sample_dataset['OAM_Mode'].value_counts()
            axes[1, 0].pie(oam_counts.values, labels=oam_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('OAM Mode Distribution')
        
        # SNR vs BER relationship
        if len(scenario_names) > 0:
            sample_dataset = datasets[scenario_names[0]]
            axes[1, 1].scatter(sample_dataset['SNR_dB'], sample_dataset['BER'], 
                             alpha=0.6, s=1, color='red')
            axes[1, 1].set_title('SNR vs BER Relationship')
            axes[1, 1].set_xlabel('SNR (dB)')
            axes[1, 1].set_ylabel('BER')
            axes[1, 1].set_yscale('log')
        
        # Correlation heatmap (subset of features)
        if len(scenario_names) > 0:
            sample_dataset = datasets[scenario_names[0]]
            key_features = ['Frequency_GHz', 'TX_Power_dBm', 'Distance_m', 
                          'SNR_dB', 'BER', 'Throughput_Gbps']
            corr_matrix = sample_dataset[key_features].corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 2], square=True)
            axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
        
    def export_summary_report(self, datasets: Dict[str, pd.DataFrame], 
                            output_file: str = 'dataset_summary_report.txt'):
        """
        Export a comprehensive summary report of all datasets.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of datasets
            output_file (str): Output file name
        """
        with open(output_file, 'w') as f:
            f.write("6G OAM THz Dataset Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            total_samples = sum(len(df) for df in datasets.values())
            f.write(f"Total Datasets: {len(datasets)}\n")
            f.write(f"Total Samples: {total_samples:,}\n")
            f.write(f"Total Features: {len(self.expected_columns)}\n\n")
            
            for scenario, df in datasets.items():
                f.write(f"\n{scenario.upper()} DATASET\n")
                f.write("-" * 30 + "\n")
                
                stats = self.get_dataset_statistics(df)
                f.write(f"Shape: {stats['shape']}\n")
                f.write(f"Memory Usage: {stats['memory_usage_mb']:.2f} MB\n")
                
                # Key statistics
                f.write(f"\nKey Performance Metrics:\n")
                if 'Throughput_Gbps' in df.columns:
                    f.write(f"  Throughput Range: {df['Throughput_Gbps'].min():.2f} - {df['Throughput_Gbps'].max():.2f} Gbps\n")
                    f.write(f"  Throughput Mean: {df['Throughput_Gbps'].mean():.2f} Gbps\n")
                
                if 'SNR_dB' in df.columns:
                    f.write(f"  SNR Range: {df['SNR_dB'].min():.2f} - {df['SNR_dB'].max():.2f} dB\n")
                
                if 'Frequency_GHz' in df.columns:
                    f.write(f"  Frequency Range: {df['Frequency_GHz'].min():.1f} - {df['Frequency_GHz'].max():.1f} GHz\n")
                
                missing_count = sum(stats['missing_values'].values())
                f.write(f"  Missing Values: {missing_count}\n")
                
        logger.info(f"Summary report exported to {output_file}")


def main():
    """
    Example usage of the OAMDatasetLoader class.
    """
    # Initialize loader
    loader = OAMDatasetLoader()
    
    # Load all datasets
    print("Loading datasets...")
    datasets = loader.load_all_datasets()
    
    if not datasets:
        print("No datasets found. Please ensure dataset files are in the current directory.")
        return
    
    # Generate statistics for each dataset
    print("\nDataset Statistics:")
    for scenario, df in datasets.items():
        stats = loader.get_dataset_statistics(df)
        print(f"\n{scenario.capitalize()}: {stats['shape'][0]} samples, {stats['shape'][1]} features")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    loader.visualize_dataset_overview(datasets, 'dataset_overview.png')
    
    # Export summary report
    print("Exporting summary report...")
    loader.export_summary_report(datasets)
    
    # Example preprocessing
    if 'comprehensive' in datasets:
        print("\nPreprocessing comprehensive dataset...")
        preprocessed = loader.preprocess_dataset(datasets['comprehensive'])
        print(f"Training set shape: {preprocessed['X_train'].shape}")
        print(f"Test set shape: {preprocessed['X_test'].shape}")
    
    print("\nDataset loading and analysis complete!")


if __name__ == "__main__":
    main()