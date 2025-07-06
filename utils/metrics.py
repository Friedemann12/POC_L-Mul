#!/usr/bin/env python
"""
Metrics Tracking Module for L-Mul Evaluation

This module provides centralized metrics tracking functionality to ensure
consistent measurement across different evaluation scripts.
"""

import time
import psutil
import os
import torch
from codecarbon import OfflineEmissionsTracker


class TaskTracker:
    """
    A context manager to track comprehensive metrics for evaluation tasks.
    
    Features:
    - Tokens per second calculation
    - Memory usage tracking (RAM and VRAM)
    - CPU usage monitoring
    - CO2 emissions estimation
    - Raw energy consumption measurement
    """
    def __init__(self, project_name, output_dir):
        self.project_name = project_name
        self.output_dir = output_dir
        self.process = psutil.Process(os.getpid())
        self.metrics = {}
        self.total_new_tokens = 0

    def __enter__(self):
        # Reset memory stats and get baseline readings
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        self.process.cpu_percent(interval=None)  # Start the CPU usage interval
        
        # Start the emissions tracker
        self.tracker = OfflineEmissionsTracker(
            project_name=self.project_name, 
            output_dir=self.output_dir,
            log_level="ERROR"  # Suppress verbose codecarbon output
        )
        self.tracker.start()
        return self

    def add_generated_tokens(self, token_count):
        """
        Accumulates the number of newly generated tokens for throughput calculation.
        
        Args:
            token_count (int): Number of tokens generated in the current operation
        """
        if isinstance(token_count, int) and token_count > 0:
            self.total_new_tokens += token_count

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop the tracker and calculate final metrics
        emissions_data = self.tracker.stop()
        duration = time.time() - self.start_time
        
        # Final hardware usage readings
        cpu_usage = self.process.cpu_percent(interval=None)
        ram_usage_gb = self.process.memory_info().rss / (1024 ** 3)
        vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

        # Calculate performance metrics
        tokens_per_second = (self.total_new_tokens / duration) if duration > 0 else 0
        
        # Get raw energy consumption from the tracker's data
        # Try multiple ways to access energy data as codecarbon API may vary
        raw_energy_kwh = 0
        if hasattr(self.tracker, 'final_emissions_data') and self.tracker.final_emissions_data:
            raw_energy_kwh = getattr(self.tracker.final_emissions_data, 'energy_consumed', 0)
        elif hasattr(self.tracker, '_measure') and hasattr(self.tracker._measure, 'energy'):
            raw_energy_kwh = self.tracker._measure.energy.kWh
        
        self.metrics = {
            "duration_seconds": round(duration, 4),
            "tokens_per_second": round(tokens_per_second, 2),
            "total_new_tokens": self.total_new_tokens,
            "cpu_percent_usage": round(cpu_usage, 2),
            "ram_usage_gb": round(ram_usage_gb, 4),
            "vram_peak_gb": round(vram_peak_gb, 4),
            "emissions_kg_co2": emissions_data,
            "energy_consumed_kwh": round(raw_energy_kwh, 6),
        } 