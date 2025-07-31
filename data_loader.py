# data_loader.py

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants and configuration
DATA_DIR = 'data'
DATA_FILE = 'renal_health_data.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42

class DataConfig:
    def __init__(self, data_dir: str, data_file: str, test_size: float, random_state: int):
        self.data_dir = data_dir
        self.data_file = data_file
        self.test_size = test_size
        self.random_state = random_state

class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.data_file = config.data_file
        self.test_size = config.test_size
        self.random_state = config.random_state

    def _load_data(self) -> pd.DataFrame:
        """Load the dataset from the specified file."""
        try:
            data_path = os.path.join(self.data_dir, self.data_file)
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                raise FileNotFoundError
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess the data by splitting it into training and testing sets."""
        try:
            X = data.drop('target', axis=1)
            y = data['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            raise

    def _scale_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale the data using StandardScaler."""
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.error(f"Failed to scale data: {e}")
            raise

    def _impute_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Impute missing values using SimpleImputer."""
        try:
            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            return X_train_imputed, X_test_imputed
        except Exception as e:
            logger.error(f"Failed to impute data: {e}")
            raise

    def _transform_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform the data using ColumnTransformer."""
        try:
            numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X_train.select_dtypes(include=['object']).columns
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', 'passthrough', categorical_features)])
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            return X_train_transformed, X_test_transformed
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            raise

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the dataset."""
        try:
            data = self._load_data()
            X_train, X_test, y_train, y_test = self._preprocess_data(data)
            X_train_scaled, X_test_scaled = self._scale_data(X_train, X_test)
            X_train_imputed, X_test_imputed = self._impute_data(X_train_scaled, X_test_scaled)
            X_train_transformed, X_test_transformed = self._transform_data(X_train_imputed, X_test_imputed)
            return X_train_transformed, X_test_transformed, y_train, y_test
        except Exception as e:
            logger.error(f"Failed to load and preprocess data: {e}")
            raise

def main():
    config = DataConfig(DATA_DIR, DATA_FILE, TEST_SIZE, RANDOM_STATE)
    data_loader = DataLoader(config)
    X_train, X_test, y_train, y_test = data_loader.load_data()
    logger.info(f"Loaded and preprocessed data: X_train shape={X_train.shape}, X_test shape={X_test.shape}, y_train shape={y_train.shape}, y_test shape={y_test.shape}")

if __name__ == "__main__":
    main()