"""
Beautiful logging system using Rich library
"""
import os
import sys
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.tree import Tree
from rich.text import Text
from rich import box
import logging

class EHRLogger:
    """Enhanced logger for EHR experiments with beautiful output"""
    
    def __init__(self, name: str = "EHR", log_file: Optional[str] = None):
        self.console = Console()
        self.name = name
        self.log_file = log_file
        
        # Setup logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add Rich handler for console output
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        self.logger.addHandler(rich_handler)
        
        # Add file handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(f"[bold blue][/bold blue] {message}")
    
    def success(self, message: str):
        """Log success message"""
        self.logger.info(f"[bold green]✅[/bold green] {message}")
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[bold yellow]⚠️[/bold yellow] {message}")
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(f"[bold red]❌[/bold red] {message}")
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(f"[dim]🐛[/dim] {message}")
    
    def start_experiment(self, args: Dict[str, Any]):
        """Display experiment start information"""
        tree = Tree("🚀 [bold green]Starting EHR Experiment[/bold green]")
        
        # Add experiment details
        tree.add(f"📊 [bold]Dataset:[/bold] {args.get('dataset', 'N/A')}")
        tree.add(f"🔧 [bold]Method:[/bold] {args.get('method', 'N/A')}")
        tree.add(f"🎯 [bold]Task:[/bold] {args.get('task', 'N/A')}")
        
        if args.get('toy_dataset'):
            tree.add("🎮 [bold yellow]Using toy dataset for quick testing[/bold yellow]")
        
        self.console.print(tree)
    
    def data_preparation_start(self, dataset_name: str):
        """Display data preparation start"""
        self.info(f"Preparing EHR data: [bold]{dataset_name}[/bold]")
    
    def data_preparation_complete(self, train_samples: int, val_samples: int, test_samples: int):
        """Display data preparation completion with statistics"""
        tree = Tree("📋 [bold green]Data Preparation Complete[/bold green]")
        
        # Create statistics table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Split", style="cyan", no_wrap=True)
        table.add_column("Samples", justify="right", style="green")
        
        table.add_row("Train", f"{train_samples:,}")
        table.add_row("Val", f"{val_samples:,}")
        table.add_row("Test", f"{test_samples:,}")
        table.add_row("Total", f"{train_samples + val_samples + test_samples:,}", style="bold")
        
        tree.add(table)
        self.console.print(tree)
    
    def toy_dataset_stats(self, train_samples: int, val_samples: int, test_samples: int):
        """Display toy dataset statistics"""
        tree = Tree("🎮 [bold yellow]Toy Dataset Statistics[/bold yellow]")
        
        table = Table(show_header=True, header_style="bold yellow", box=box.ROUNDED)
        table.add_column("Split", style="cyan", no_wrap=True)
        table.add_column("Samples", justify="right", style="yellow")
        
        table.add_row("Train", f"{train_samples}")
        table.add_row("Val", f"{val_samples}")
        table.add_row("Test", f"{test_samples}")
        
        tree.add(table)
        self.console.print(tree)
    
    def processing_start(self, task_name: str):
        """Display processing start"""
        self.info(f"Starting {task_name}...")
    
    def processing_complete(self, task_name: str):
        """Display processing completion"""
        self.success(f"{task_name} completed")
    
    def show_message_example(self, message: str, title: str, max_length: int = 20000):
        """Display a formatted message example"""
        # Truncate message if too long
        display_message = message[:max_length] + "..." if len(message) > max_length else message
        
        panel = Panel(
            display_message,
            title=f"[bold blue]📝 {title}[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
        
        # Also write to log file if available
        if self.log_file:
            # Write title and message to log file
            self.logger.info(f"📝 {title}")
            self.logger.info("-" * 80)
            # Write the full message (not truncated) to log file
            full_message = message if len(message) <= max_length else message
            self.logger.info(full_message)
            self.logger.info("-" * 80)
    
    def create_progress(self, description: str, total: int) -> Progress:
        """Create a rich progress bar"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
    
    def print_separator(self, char: str = "─", length: int = 50):
        """Print a separator line"""
        self.console.print(char * length, style="dim")
    
    def log_metrics(self, metrics_dict: Dict[str, Any], title: str = "Model Evaluation Results"):
        """Display model evaluation metrics in a beautiful table format"""
        tree = Tree(f"📊 [bold green]{title}[/bold green]")
        
        # Check if this is bootstrap results (complex nested structure)
        is_bootstrap = any(isinstance(v, dict) and 'bootstrap_mean' in v for v in metrics_dict.values())
        
        if is_bootstrap:
            # Create bootstrap metrics table
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Original", justify="right", style="green")
            table.add_column("Bootstrap Mean ± Std", justify="right", style="yellow")
            table.add_column("95% CI", justify="right", style="blue")
            
            # Add bootstrap metrics to table
            for metric_name, metric_data in metrics_dict.items():
                if isinstance(metric_data, dict) and 'bootstrap_mean' in metric_data:
                    original_val = metric_data['value']
                    bootstrap_mean = metric_data['bootstrap_mean']
                    bootstrap_std = metric_data['bootstrap_std']
                    ci_lower = metric_data['ci_lower']
                    ci_upper = metric_data['ci_upper']
                    
                    # Format values, handling NaN
                    original_str = f"{original_val:.4f}" if not np.isnan(original_val) else "NaN"
                    
                    if np.isnan(bootstrap_mean) or np.isnan(bootstrap_std):
                        bootstrap_str = "NaN ± NaN"
                    else:
                        bootstrap_str = f"{bootstrap_mean:.4f} ± {bootstrap_std:.4f}"
                    
                    if np.isnan(ci_lower) or np.isnan(ci_upper):
                        ci_str = "[NaN, NaN]"
                    else:
                        ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                    
                    table.add_row(metric_name, original_str, bootstrap_str, ci_str)
                elif isinstance(metric_data, (int, float)):
                    # Simple numeric values (like Threshold)
                    table.add_row(metric_name, f"{metric_data:.4f}", "-", "-")
                else:
                    # Non-numeric values
                    table.add_row(metric_name, str(metric_data), "-", "-")
        else:
            # Create simple metrics table
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", justify="right", style="green")
            
            # Add metrics to table
            for metric_name, metric_value in metrics_dict.items():
                if isinstance(metric_value, (int, float)):
                    # Format numeric values with appropriate precision
                    if abs(metric_value) < 0.01:
                        formatted_value = f"{metric_value:.6f}"
                    elif abs(metric_value) < 1:
                        formatted_value = f"{metric_value:.4f}"
                    else:
                        formatted_value = f"{metric_value:.4f}"
                    
                    table.add_row(metric_name, formatted_value)
                else:
                    # Non-numeric values
                    table.add_row(metric_name, str(metric_value))
        
        tree.add(table)
        self.console.print(tree)
        
        # Also log to file if available (simplified)
        self.info(f"Evaluation completed - {title}")

# Global logger instance
logger = EHRLogger()

def get_logger(name: str = "EHR", log_file: Optional[str] = None, 
               experiment_info: Optional[Dict[str, Any]] = None) -> EHRLogger:
    """Get a logger instance with optional automatic log file generation"""
    
    # If no log_file specified but experiment_info provided, generate filename
    if log_file is None and experiment_info:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset = experiment_info.get('dataset', 'unknown')
        model = experiment_info.get('model', 'unknown').replace('/', '_').replace('\\', '_')
        method = experiment_info.get('method', 'unknown')
        log_file = f"logs/{timestamp}_{dataset}_{model}_{method}.log"
    
    return EHRLogger(name, log_file)
