#!/usr/bin/env python3
"""
Data Utilities for FLAN-T5 Finetuning
Handles dataset loading, preprocessing, and validation
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset, DatasetDict
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processor for FLAN-T5 finetuning"""
    
    def __init__(self):
        self.supported_formats = ['.json', '.csv', '.jsonl', '.tsv']
    
    def load_dataset(self, data_path: str) -> Dataset:
        """
        Load dataset from various formats
        
        Args:
            data_path: Path to dataset file
            
        Returns:
            HuggingFace Dataset object
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        file_extension = data_path.suffix.lower()
        
        if file_extension == '.json':
            return self._load_json(data_path)
        elif file_extension == '.csv':
            return self._load_csv(data_path)
        elif file_extension == '.jsonl':
            return self._load_jsonl(data_path)
        elif file_extension == '.tsv':
            return self._load_tsv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_json(self, file_path: Path) -> Dataset:
        """Load JSON dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            # List of dictionaries
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            # Dictionary with data arrays
            if 'data' in data:
                return Dataset.from_list(data['data'])
            elif 'train' in data:
                return Dataset.from_list(data['train'])
            else:
                # Assume it's a single example
                return Dataset.from_list([data])
        else:
            raise ValueError("Invalid JSON format")
    
    def _load_csv(self, file_path: Path) -> Dataset:
        """Load CSV dataset"""
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)
    
    def _load_jsonl(self, file_path: Path) -> Dataset:
        """Load JSONL dataset"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return Dataset.from_list(data)
    
    def _load_tsv(self, file_path: Path) -> Dataset:
        """Load TSV dataset"""
        df = pd.read_csv(file_path, sep='\t')
        return Dataset.from_pandas(df)
    
    def validate_dataset(self, dataset: Dataset, required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate dataset format
        
        Args:
            dataset: Dataset to validate
            required_columns: Required column names
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        columns = dataset.column_names
        
        # Check for required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for common input/output column patterns
        input_columns = ['input', 'question', 'text', 'source', 'context']
        output_columns = ['output', 'answer', 'summary', 'target', 'response']
        
        has_input = any(col in columns for col in input_columns)
        has_output = any(col in columns for col in output_columns)
        
        if not has_input:
            raise ValueError(f"No input column found. Available columns: {columns}")
        
        if not has_output:
            raise ValueError(f"No output column found. Available columns: {columns}")
        
        # Check for empty values
        for column in columns:
            empty_count = sum(1 for item in dataset[column] if not item or str(item).strip() == '')
            if empty_count > 0:
                logger.warning(f"Column '{column}' has {empty_count} empty values")
        
        logger.info(f"Dataset validation passed. Shape: {len(dataset)} rows, {len(columns)} columns")
        return True
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> DatasetDict:
        """
        Split dataset into train/validation/test sets
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # Shuffle dataset
        shuffled_dataset = dataset.shuffle(seed=42)
        
        # Split dataset
        train_dataset = shuffled_dataset.select(range(train_size))
        val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
        test_dataset = shuffled_dataset.select(range(train_size + val_size, total_size))
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
    
    def create_sample_dataset(self, output_path: str, num_samples: int = 100):
        """
        Create a sample dataset for testing
        
        Args:
            output_path: Path to save sample dataset
            num_samples: Number of sample entries to create
        """
        sample_data = []
        
        # Sample tasks for demonstration (in this sample, it just translates phrases English to French)
        tasks = [
            ("Translate this to French:", "Traduisez ceci en français:"),
            ("Summarize this text:", "Résumez ce texte:"),
            ("Answer this question:", "Répondez à cette question:"),
            ("Complete this sentence:", "Complétez cette phrase:"),
            ("Explain this concept:", "Expliquez ce concept:")
        ]
        
        for i in range(num_samples):
            task_idx = i % len(tasks)
            task_input, task_output = tasks[task_idx]
            
            sample_data.append({
                "input": f"{task_input} This is sample input {i+1}.",
                "output": f"{task_output} This is sample output {i+1}."
            })
        
        # Save as JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sample dataset created with {num_samples} samples at {output_path}")
    
    def convert_format(self, input_path: str, output_path: str, 
                      input_format: str = 'auto', output_format: str = 'json'):
        """
        Convert dataset between different formats
        
        Args:
            input_path: Path to input dataset
            output_path: Path to output dataset
            input_format: Input format (auto, json, csv, jsonl, tsv)
            output_format: Output format (json, csv, jsonl, tsv)
        """
        # Load dataset
        if input_format == 'auto':
            dataset = self.load_dataset(input_path)
        else:
            # Handle specific format loading
            pass
        
        # Convert to desired format
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)
        
        elif output_format == 'csv':
            df = dataset.to_pandas()
            df.to_csv(output_path, index=False)
        
        elif output_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        elif output_format == 'tsv':
            df = dataset.to_pandas()
            df.to_csv(output_path, sep='\t', index=False)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Dataset converted from {input_path} to {output_path}")


def create_sample_data():
    """Create sample datasets for testing"""
    processor = DataProcessor()
    
    # Create sample training data
    processor.create_sample_dataset("data/sample_training_data.json", num_samples=100)
    
    # Create sample evaluation data
    processor.create_sample_dataset("data/sample_eval_data.json", num_samples=20)
    
    print("Sample datasets created successfully!")


if __name__ == "__main__":
    create_sample_data() 