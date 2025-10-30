#!/usr/bin/env python3
"""
Enhanced Experiment Script

This script provides a comprehensive interface for running experiments with
configurable parameters for data preprocessing, feature engineering,
dataset splitting, model selection, and hyperparameter tuning.

Usage:
    python enhanced_experiment.py --help
    python enhanced_experiment.py --config experiment_config.json
    python enhanced_experiment.py --outlier-method iqr --split-method chronological --model RandomForest
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

class EnhancedExperimentRunner:
    """Enhanced experiment runner with comprehensive parameter control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the experiment runner with configuration."""
        self.config = config or {}
        self.experiment_id = self._generate_experiment_id()
        self.results_dir = RESULTS_DIR / self.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_id = self.config.get("experiment_name", "exp")
        return f"{base_id}_{timestamp}"
    
    def run_preprocessing(self, **kwargs) -> bool:
        """Run data preprocessing with configurable parameters."""
        logger.info("Starting data preprocessing...")
        
        # Build preprocessing command
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "stage2_preprocessing.py"),
            "--outlier-method", kwargs.get("outlier_method", "both"),
            "--iqr-multiplier", str(kwargs.get("iqr_multiplier", 1.5)),
            "--z-threshold", str(kwargs.get("z_threshold", 3.0)),
            "--max-outlier-ratio", str(kwargs.get("max_outlier_ratio", 0.1)),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            if result.returncode == 0:
                logger.info("Data preprocessing completed successfully")
                return True
            else:
                logger.error(f"Preprocessing failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return False
    
    def run_experiments(self, **kwargs) -> bool:
        """Run model experiments with configurable parameters."""
        logger.info("Starting model experiments...")
        
        # Build experiment command
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "model_horizon_experiments.py"),
            "--split-method", kwargs.get("split_method", "random"),
            "--group-size", str(kwargs.get("group_size", 168)),
            "--time-features", kwargs.get("time_features", "full"),
            "--test-size", str(kwargs.get("test_size", 0.2)),
            "--random-state", str(kwargs.get("random_state", 42)),
        ]
        
        # Add optional parameters
        if kwargs.get("window_multiplier"):
            cmd.extend(["--window-multiplier", str(kwargs["window_multiplier"])])
        if kwargs.get("transformer"):
            cmd.extend(["--transformer", str(kwargs["transformer"])])
        if kwargs.get("model") and kwargs["model"] != "all":
            cmd.extend(["--model", kwargs["model"]])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            if result.returncode == 0:
                logger.info("Model experiments completed successfully")
                self._save_experiment_output(result.stdout)
                return True
            else:
                logger.error(f"Experiments failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Experiments error: {e}")
            return False
    
    def _save_experiment_output(self, output: str):
        """Save experiment output to results directory."""
        output_file = self.results_dir / "experiment_output.txt"
        with open(output_file, 'w') as f:
            f.write(output)
        logger.info(f"Experiment output saved to {output_file}")
    
    def run_full_pipeline(self, **kwargs) -> bool:
        """Run complete experiment pipeline."""
        logger.info(f"Starting full experiment pipeline: {self.experiment_id}")
        
        # Step 1: Data preprocessing
        if not self.run_preprocessing(**kwargs):
            logger.error("Pipeline failed at preprocessing step")
            return False
        
        # Step 2: Model experiments
        if not self.run_experiments(**kwargs):
            logger.error("Pipeline failed at experiments step")
            return False
        
        logger.info("Full experiment pipeline completed successfully")
        return True
    
    def save_config(self):
        """Save experiment configuration."""
        config_file = self.results_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {config_file}")

def create_sample_config():
    """Create a sample configuration file."""
    sample_config = {
        "experiment_name": "enhanced_oil_temp_prediction",
        "description": "Enhanced experiment with configurable parameters",
        "preprocessing": {
            "outlier_method": "both",  # none, iqr, zscore, both, physical
            "iqr_multiplier": 1.5,
            "z_threshold": 3.0,
            "max_outlier_ratio": 0.1
        },
        "experiments": {
            "split_method": "chronological",  # random, chronological, group
            "group_size": 168,  # hours (1 week)
            "time_features": "full",  # none, basic, full, minimal
            "test_size": 0.2,
            "random_state": 42,
            "window_multiplier": null,  # 1, 2, 4, 8
            "transformer": null,  # 1, 2, or null for both
            "model": "all"  # LinearRegression, Ridge, RandomForest, MLP, all
        },
        "notes": "This configuration demonstrates all available parameters"
    }
    
    config_file = RESULTS_DIR / "sample_config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample configuration created: {config_file}")
    return sample_config

def main():
    """Main function to run enhanced experiments."""
    parser = argparse.ArgumentParser(
        description="Enhanced experiment runner for oil temperature prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with configuration file
  python enhanced_experiment.py --config my_config.json
  
  # Run with command line parameters
  python enhanced_experiment.py --outlier-method iqr --split-method chronological --model RandomForest
  
  # Run full pipeline with custom parameters
  python enhanced_experiment.py --outlier-method both --split-method group --time-features basic --window-multiplier 2
  
  # Create sample configuration file
  python enhanced_experiment.py --create-sample-config
  
  # Run specific transformer and model
  python enhanced_experiment.py --transformer 1 --model MLP --split-method random
        """
    )
    
    # Configuration options
    parser.add_argument("--config", type=str,
                       help="JSON configuration file path")
    parser.add_argument("--create-sample-config", action="store_true",
                       help="Create a sample configuration file")
    
    # Preprocessing parameters
    parser.add_argument("--outlier-method", choices=["none", "iqr", "zscore", "both", "physical"],
                       default="both", help="Outlier detection method")
    parser.add_argument("--iqr-multiplier", type=float, default=1.5,
                       help="IQR multiplier for outlier detection")
    parser.add_argument("--z-threshold", type=float, default=3.0,
                       help="Z-score threshold for outlier detection")
    parser.add_argument("--max-outlier-ratio", type=float, default=0.1,
                       help="Maximum allowed outlier ratio")
    
    # Experiment parameters
    parser.add_argument("--split-method", choices=["random", "chronological", "group"],
                       default="random", help="Data splitting method")
    parser.add_argument("--group-size", type=int, default=168,
                       help="Group size for group-based splitting (hours)")
    parser.add_argument("--time-features", choices=["none", "basic", "full", "minimal"],
                       default="full", help="Time features to include")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size ratio")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility")
    parser.add_argument("--window-multiplier", type=int, choices=[1, 2, 4, 8],
                       help="Window size multiplier for lookback")
    parser.add_argument("--transformer", type=int, choices=[1, 2],
                       help="Specific transformer to process")
    parser.add_argument("--model", type=str, choices=["LinearRegression", "Ridge", "RandomForest", "MLP", "all"],
                       default="all", help="Model to train")
    
    # Experiment control
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip preprocessing step")
    parser.add_argument("--experiment-name", type=str, default="enhanced_exp",
                       help="Experiment name for results directory")
    
    args = parser.parse_args()
    
    # Handle sample config creation
    if args.create_sample_config:
        create_sample_config()
        return
    
    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return
    else:
        # Build configuration from command line arguments
        config = {
            "experiment_name": args.experiment_name,
            "preprocessing": {
                "outlier_method": args.outlier_method,
                "iqr_multiplier": args.iqr_multiplier,
                "z_threshold": args.z_threshold,
                "max_outlier_ratio": args.max_outlier_ratio,
            },
            "experiments": {
                "split_method": args.split_method,
                "group_size": args.group_size,
                "time_features": args.time_features,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "window_multiplier": args.window_multiplier,
                "transformer": args.transformer,
                "model": args.model,
            }
        }
    
    # Initialize and run experiment
    runner = EnhancedExperimentRunner(config)
    runner.save_config()
    
    # Run pipeline
    success = False
    if args.skip_preprocessing:
        success = runner.run_experiments(**config.get("experiments", {}))
    else:
        success = runner.run_full_pipeline(**{**config.get("preprocessing", {}), **config.get("experiments", {})})
    
    if success:
        logger.info(f"Experiment completed successfully. Results saved to: {runner.results_dir}")
    else:
        logger.error("Experiment failed")
        sys.exit(1)

if __name__ == "__main__":
    main()