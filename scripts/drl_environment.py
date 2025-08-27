#!/usr/bin/env python3
"""
Deep Reinforcement Learning Environment for 6G OAM THz Dataset
============================================================

Custom OpenAI Gym environment for Track 3 of the competition.
Implements realistic 6G OAM THz communication scenarios with
continuous action spaces for parameter optimization.

Author: 6G Research Team
Version: 1.0
License: MIT
"""

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OAMTHzEnv(gym.Env):
    """
    OpenAI Gym environment for 6G OAM THz communication optimization.
    
    This environment simulates the process of optimizing communication parameters
    in a 6G OAM THz system to maximize throughput while considering constraints
    like energy efficiency and latency.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, dataset_path: Optional[str] = None, scenario: str = 'comprehensive'):
        """
        Initialize the 6G OAM THz environment.
        
        Args:
            dataset_path (str): Path to the dataset file
            scenario (str): Scenario type for environment configuration
        """
        super(OAMTHzEnv, self).__init__()
        
        self.scenario = scenario
        self.current_step = 0
        self.max_steps = 1000
        self.episode_rewards = []
        
        # Load reference dataset if available
        self.reference_data = None
        if dataset_path:
            try:
                self.reference_data = pd.read_csv(dataset_path)
                logger.info(f"Loaded reference dataset: {self.reference_data.shape}")
            except Exception as e:
                logger.warning(f"Could not load dataset: {e}")
        
        # Define action space (continuous parameters to optimize)
        self.action_space = spaces.Box(
            low=np.array([
                300.0,    # Frequency_GHz (300-600)
                1,        # OAM_Mode (1-8)
                10.0,     # TX_Power_dBm (10-30)
                50.0,     # Distance_m (50-200)
                0.0,      # Elevation_Angle_deg (0-90)
                0.0,      # Azimuth_Angle_deg (0-360)
                1.0,      # Beam_Divergence_mrad (1-10)
            ]),
            high=np.array([
                600.0,    # Frequency_GHz
                8,        # OAM_Mode
                30.0,     # TX_Power_dBm
                200.0,    # Distance_m
                90.0,     # Elevation_Angle_deg
                360.0,    # Azimuth_Angle_deg
                10.0,     # Beam_Divergence_mrad
            ]),
            dtype=np.float32
        )
        
        # Define observation space (system state)
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,      # SNR_dB (0-40)
                1e-6,     # BER (1e-6 to 1e-1)
                0.0,      # Throughput_Gbps (0-100)
                0.0,      # Latency_ms (0-100)
                0.0,      # Energy_Efficiency (0-1000)
                0.0,      # Packet_Loss_Rate (0-1)
                -50.0,    # Atmospheric_Attenuation_dB_per_km
                0.0,      # Rain_Rate_mm_per_hr (0-50)
                0.0,      # Temperature_C (-20 to 50)
                0.0,      # Humidity_percent (0-100)
            ]),
            high=np.array([
                40.0,     # SNR_dB
                1e-1,     # BER
                100.0,    # Throughput_Gbps
                100.0,    # Latency_ms
                1000.0,   # Energy_Efficiency
                1.0,      # Packet_Loss_Rate
                0.0,      # Atmospheric_Attenuation_dB_per_km
                50.0,     # Rain_Rate_mm_per_hr
                50.0,     # Temperature_C
                100.0,    # Humidity_percent
            ]),
            dtype=np.float32
        )
        
        # Initialize environment state
        self.state = None
        self.environmental_conditions = self._sample_environmental_conditions()
        
        # Performance tracking
        self.performance_history = {
            'throughput': [],
            'energy_efficiency': [],
            'latency': [],
            'reward': []
        }
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial observation
        """
        self.current_step = 0
        self.environmental_conditions = self._sample_environmental_conditions()
        
        # Sample initial system state
        self.state = np.array([
            np.random.uniform(5.0, 25.0),      # SNR_dB
            np.random.uniform(1e-5, 1e-2),    # BER
            np.random.uniform(1.0, 50.0),     # Throughput_Gbps
            np.random.uniform(1.0, 20.0),     # Latency_ms
            np.random.uniform(100.0, 800.0),  # Energy_Efficiency
            np.random.uniform(0.0, 0.1),      # Packet_Loss_Rate
            self.environmental_conditions['attenuation'],
            self.environmental_conditions['rain_rate'],
            self.environmental_conditions['temperature'],
            self.environmental_conditions['humidity']
        ], dtype=np.float32)
        
        # Reset performance tracking
        self.performance_history = {
            'throughput': [],
            'energy_efficiency': [],
            'latency': [],
            'reward': []
        }
        
        return self.state
        
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action (np.ndarray): Action to take
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_step += 1
        
        # Extract action parameters
        frequency = action[0]
        oam_mode = int(action[1])
        tx_power = action[2]
        distance = action[3]
        elevation = action[4]
        azimuth = action[5]
        beam_divergence = action[6]
        
        # Calculate new system state based on action
        new_state = self._calculate_system_response(
            frequency, oam_mode, tx_power, distance, 
            elevation, azimuth, beam_divergence
        )
        
        # Calculate reward
        reward = self._calculate_reward(new_state, action)
        
        # Update state
        self.state = new_state
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps or 
                self._is_terminal_state(new_state))
        
        # Update performance history
        self.performance_history['throughput'].append(new_state[2])
        self.performance_history['energy_efficiency'].append(new_state[4])
        self.performance_history['latency'].append(new_state[3])
        self.performance_history['reward'].append(reward)
        
        # Prepare info dictionary
        info = {
            'throughput': new_state[2],
            'energy_efficiency': new_state[4],
            'latency': new_state[3],
            'snr': new_state[0],
            'ber': new_state[1],
            'packet_loss': new_state[5],
            'step': self.current_step
        }
        
        return new_state, reward, done, info
        
    def _sample_environmental_conditions(self):
        """
        Sample realistic environmental conditions.
        
        Returns:
            dict: Environmental parameters
        """
        # Scenario-specific environmental conditions
        if self.scenario == 'outdoor_urban':
            conditions = {
                'attenuation': np.random.uniform(-30.0, -10.0),
                'rain_rate': np.random.uniform(0.0, 10.0),
                'temperature': np.random.uniform(5.0, 35.0),
                'humidity': np.random.uniform(30.0, 80.0),
                'wind_speed': np.random.uniform(2.0, 15.0)
            }
        elif self.scenario == 'indoor_realistic':
            conditions = {
                'attenuation': np.random.uniform(-20.0, -5.0),
                'rain_rate': 0.0,  # No rain indoors
                'temperature': np.random.uniform(18.0, 25.0),
                'humidity': np.random.uniform(40.0, 60.0),
                'wind_speed': 0.0  # No wind indoors
            }
        elif self.scenario == 'high_mobility':
            conditions = {
                'attenuation': np.random.uniform(-25.0, -8.0),
                'rain_rate': np.random.uniform(0.0, 20.0),
                'temperature': np.random.uniform(-10.0, 40.0),
                'humidity': np.random.uniform(20.0, 90.0),
                'wind_speed': np.random.uniform(5.0, 25.0)
            }
        else:  # comprehensive or lab_controlled
            conditions = {
                'attenuation': np.random.uniform(-25.0, -5.0),
                'rain_rate': np.random.uniform(0.0, 15.0),
                'temperature': np.random.uniform(0.0, 40.0),
                'humidity': np.random.uniform(20.0, 90.0),
                'wind_speed': np.random.uniform(0.0, 20.0)
            }
            
        return conditions
        
    def _calculate_system_response(self, frequency, oam_mode, tx_power, 
                                 distance, elevation, azimuth, beam_divergence):
        """
        Calculate system response based on input parameters.
        
        This is a simplified physics-based model. In practice, this would
        involve complex electromagnetic simulations.
        
        Args:
            frequency: Operating frequency in GHz
            oam_mode: OAM mode index
            tx_power: Transmit power in dBm
            distance: Communication distance in meters
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
            beam_divergence: Beam divergence in mrad
            
        Returns:
            np.ndarray: New system state
        """
        # Path loss calculation (simplified)
        free_space_loss = 20 * np.log10(distance) + 20 * np.log10(frequency) + 32.44
        atmospheric_loss = abs(self.environmental_conditions['attenuation']) * distance / 1000
        
        # OAM-specific losses
        oam_loss = 2.0 + 0.5 * (oam_mode - 1)  # Higher OAM modes have more loss
        
        # Total path loss
        total_loss = free_space_loss + atmospheric_loss + oam_loss
        
        # Received power
        rx_power = tx_power - total_loss
        
        # SNR calculation (simplified)
        noise_power = -174 + 10 * np.log10(1e9)  # Thermal noise for 1 GHz bandwidth
        snr = rx_power - noise_power
        
        # BER calculation (simplified QPSK)
        if snr > 0:
            ber = 0.5 * np.exp(-snr / 10)
        else:
            ber = 0.1  # High BER for negative SNR
            
        # Throughput calculation
        bandwidth_ghz = 1.0  # Assume 1 GHz bandwidth
        capacity = bandwidth_ghz * np.log2(1 + 10**(snr/10))  # Shannon capacity
        throughput = capacity * (1 - ber) * (1 - self.environmental_conditions['rain_rate'] / 100)
        
        # Latency calculation
        propagation_delay = distance / 3e8 * 1000  # Speed of light delay in ms
        processing_delay = 1.0 + 0.1 * oam_mode  # OAM processing overhead
        latency = propagation_delay + processing_delay
        
        # Energy efficiency calculation
        power_consumption = 10**(tx_power / 10) / 1000  # Convert dBm to Watts
        energy_efficiency = throughput / power_consumption if power_consumption > 0 else 0
        
        # Packet loss rate calculation
        packet_loss = ber + 0.01 * (self.environmental_conditions['rain_rate'] / 10)
        packet_loss = min(packet_loss, 1.0)
        
        # Construct new state
        new_state = np.array([
            max(0, min(40, snr)),                    # SNR_dB
            max(1e-6, min(1e-1, ber)),              # BER
            max(0, min(100, throughput)),            # Throughput_Gbps
            max(0, min(100, latency)),               # Latency_ms
            max(0, min(1000, energy_efficiency)),    # Energy_Efficiency
            max(0, min(1, packet_loss)),             # Packet_Loss_Rate
            self.environmental_conditions['attenuation'],
            self.environmental_conditions['rain_rate'],
            self.environmental_conditions['temperature'],
            self.environmental_conditions['humidity']
        ], dtype=np.float32)
        
        return new_state
        
    def _calculate_reward(self, state, action):
        """
        Calculate reward based on current state and action.
        
        Args:
            state (np.ndarray): Current system state
            action (np.ndarray): Action taken
            
        Returns:
            float: Reward value
        """
        # Extract state variables
        snr = state[0]
        ber = state[1]
        throughput = state[2]
        latency = state[3]
        energy_efficiency = state[4]
        packet_loss = state[5]
        
        # Multi-objective reward function
        # Maximize throughput and energy efficiency, minimize latency and packet loss
        
        # Throughput reward (0-1 normalized)
        throughput_reward = throughput / 100.0
        
        # Energy efficiency reward (0-1 normalized)
        efficiency_reward = energy_efficiency / 1000.0
        
        # Latency penalty (0-1 normalized, inverted)
        latency_penalty = max(0, 1.0 - latency / 100.0)
        
        # Packet loss penalty (0-1 normalized, inverted)
        packet_loss_penalty = max(0, 1.0 - packet_loss)
        
        # BER penalty
        ber_penalty = max(0, 1.0 - np.log10(ber / 1e-6) / np.log10(1e-1 / 1e-6))
        
        # Combine rewards with weights
        total_reward = (0.4 * throughput_reward +
                       0.3 * efficiency_reward +
                       0.1 * latency_penalty +
                       0.1 * packet_loss_penalty +
                       0.1 * ber_penalty)
        
        # Bonus for meeting performance thresholds
        if throughput > 50.0 and ber < 1e-3 and latency < 10.0:
            total_reward += 0.5  # Bonus for excellent performance
            
        # Penalty for unrealistic parameter combinations
        frequency = action[0]
        tx_power = action[2]
        distance = action[3]
        
        if tx_power > 25.0 and distance < 75.0:  # Too much power for short distance
            total_reward -= 0.2
            
        if frequency < 350.0 and throughput > 80.0:  # Unrealistic throughput for low freq
            total_reward -= 0.1
            
        return total_reward
        
    def _is_terminal_state(self, state):
        """
        Check if current state is terminal.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            bool: True if terminal state
        """
        # Terminal conditions
        ber = state[1]
        throughput = state[2]
        packet_loss = state[5]
        
        # System failure conditions
        if ber > 0.05:  # BER too high
            return True
        if throughput < 0.1:  # Throughput too low
            return True
        if packet_loss > 0.8:  # Packet loss too high
            return True
            
        return False
        
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            if self.state is not None:
                print(f"Step: {self.current_step}")
                print(f"SNR: {self.state[0]:.2f} dB")
                print(f"BER: {self.state[1]:.2e}")
                print(f"Throughput: {self.state[2]:.2f} Gbps")
                print(f"Latency: {self.state[3]:.2f} ms")
                print(f"Energy Efficiency: {self.state[4]:.2f}")
                print(f"Packet Loss: {self.state[5]:.4f}")
                print("-" * 30)
                
    def close(self):
        """Close the environment."""
        pass
        
    def get_performance_summary(self):
        """
        Get summary of episode performance.
        
        Returns:
            dict: Performance statistics
        """
        if not self.performance_history['throughput']:
            return {}
            
        summary = {
            'avg_throughput': np.mean(self.performance_history['throughput']),
            'max_throughput': np.max(self.performance_history['throughput']),
            'avg_energy_efficiency': np.mean(self.performance_history['energy_efficiency']),
            'avg_latency': np.mean(self.performance_history['latency']),
            'min_latency': np.min(self.performance_history['latency']),
            'total_reward': np.sum(self.performance_history['reward']),
            'avg_reward': np.mean(self.performance_history['reward']),
            'episode_length': len(self.performance_history['reward'])
        }
        
        return summary
        
    def plot_performance(self, save_path=None):
        """
        Plot episode performance metrics.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.performance_history['throughput']:
            print("No performance data to plot.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Episode Performance - {self.scenario}', fontsize=14, fontweight='bold')
        
        steps = range(len(self.performance_history['throughput']))
        
        # Throughput
        axes[0, 0].plot(steps, self.performance_history['throughput'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Throughput over Time')
        axes[0, 0].set_ylabel('Throughput (Gbps)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy Efficiency
        axes[0, 1].plot(steps, self.performance_history['energy_efficiency'], 'g-', alpha=0.7)
        axes[0, 1].set_title('Energy Efficiency over Time')
        axes[0, 1].set_ylabel('Energy Efficiency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Latency
        axes[1, 0].plot(steps, self.performance_history['latency'], 'r-', alpha=0.7)
        axes[1, 0].set_title('Latency over Time')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward
        axes[1, 1].plot(steps, self.performance_history['reward'], 'm-', alpha=0.7)
        axes[1, 1].set_title('Reward over Time')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to {save_path}")
            
        plt.show()


def test_environment():
    """
    Test the OAM THz environment with random actions.
    """
    print("Testing 6G OAM THz Environment")
    print("=" * 40)
    
    # Create environment
    env = OAMTHzEnv(scenario='comprehensive')
    
    # Test multiple episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        print("-" * 20)
        
        state = env.reset()
        total_reward = 0
        
        for step in range(50):  # Short episodes for testing
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Render every 10 steps
            if step % 10 == 0:
                env.render()
                
            if done:
                print(f"Episode ended at step {step + 1}")
                break
                
        print(f"Total reward: {total_reward:.2f}")
        
        # Get performance summary
        summary = env.get_performance_summary()
        print(f"Average throughput: {summary.get('avg_throughput', 0):.2f} Gbps")
        print(f"Average latency: {summary.get('avg_latency', 0):.2f} ms")
        
        # Plot performance for last episode
        if episode == 2:
            env.plot_performance(f'episode_{episode + 1}_performance.png')
    
    env.close()
    print("\nEnvironment testing completed!")


def drl_training_example():
    """
    Example of training a DRL agent using stable-baselines3 (if available).
    """
    print("\nDRL Training Example")
    print("=" * 30)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        
        # Create environment
        env = OAMTHzEnv(scenario='comprehensive')
        
        # Check environment
        check_env(env)
        print("Environment check passed!")
        
        # Create DRL agent
        model = PPO('MlpPolicy', env, verbose=1)
        
        # Train agent (short training for example)
        print("Training DRL agent...")
        model.learn(total_timesteps=1000)
        
        # Test trained agent
        print("Testing trained agent...")
        state = env.reset()
        for _ in range(100):
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            if done:
                break
                
        # Get final performance
        summary = env.get_performance_summary()
        print(f"Trained agent performance:")
        print(f"  Average throughput: {summary.get('avg_throughput', 0):.2f} Gbps")
        print(f"  Average reward: {summary.get('avg_reward', 0):.2f}")
        
        # Save model
        model.save("oam_thz_ppo_model")
        print("Model saved as 'oam_thz_ppo_model'")
        
    except ImportError:
        print("stable-baselines3 not installed. Install with:")
        print("pip install stable-baselines3")
        print("Skipping DRL training example.")


if __name__ == "__main__":
    # Test the environment
    test_environment()
    
    # DRL training example
    drl_training_example()