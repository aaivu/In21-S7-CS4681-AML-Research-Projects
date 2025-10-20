"""
Data preprocessing utilities for Network Intrusion Detection System
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from config import FEATURE_NAMES, ATTACK_MAPPING, MODEL_CONFIG


class NSLKDDPreprocessor:
    """
    Preprocessor for NSL-KDD dataset with attack-specific feature selection
    """
    
    def __init__(self, random_seed: int = MODEL_CONFIG['random_seed']):
        self.random_seed = random_seed
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scalers = {}
        self.feature_selectors = {}
        self.categorical_columns = ['protocol_type', 'service', 'flag']
        self.feature_columns = None
        
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare NSL-KDD datasets"""
        train_df = pd.read_csv(train_path, header=None, names=FEATURE_NAMES)
        test_df = pd.read_csv(test_path, header=None, names=FEATURE_NAMES)
        
        return train_df, test_df
    
    def load_test_data(self, test_path: str) -> pd.DataFrame:
        """Load only test dataset for evaluation"""
        test_df = pd.read_csv(test_path, header=None, names=FEATURE_NAMES)
        return test_df
    
    def prepare_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare basic features including attack label mapping"""
        df_processed = df.copy()
        
        # Check for unmapped attacks before mapping
        unique_attacks = df_processed['attack'].unique()
        unmapped = [a for a in unique_attacks if a not in ATTACK_MAPPING]
        if unmapped:
            # print(f"Warning: Found unmapped attack types: {unmapped}")  # COMMENTED OUT
            # print("These will be mapped to NaN")  # COMMENTED OUT
            pass
        
        # Map attack labels to numeric codes
        df_processed['attack'] = df_processed['attack'].map(ATTACK_MAPPING)
        
        # Check for NaN values after mapping
        nan_count = df_processed['attack'].isna().sum()
        if nan_count > 0:
            # print(f"Warning: {nan_count} attack labels became NaN after mapping")  # COMMENTED OUT
            # print("First few NaN samples (original attack labels):")  # COMMENTED OUT
            # nan_mask = df_processed['attack'].isna()
            # original_attacks = df['attack'][nan_mask].head(10)
            # print(f"  {original_attacks.tolist()}")  # COMMENTED OUT
            pass

        
        return df_processed
    
    def encode_categorical_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features using consistent encoding"""
        train_categorical = train_df[self.categorical_columns].copy()
        test_categorical = test_df[self.categorical_columns].copy()
        
        # Label encoding
        for col in self.categorical_columns:
            le = LabelEncoder()
            combined_data = pd.concat([train_categorical[col], test_categorical[col]])
            le.fit(combined_data)
            train_categorical[col] = le.transform(train_categorical[col])
            test_categorical[col] = le.transform(test_categorical[col])
            self.label_encoders[col] = le
        
        # One-hot encoding
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit on combined data for consistency
        combined_categorical = pd.concat([train_categorical, test_categorical], ignore_index=True)
        self.onehot_encoder.fit(combined_categorical)
        
        # Transform data
        train_encoded = self.onehot_encoder.transform(train_categorical)
        test_encoded = self.onehot_encoder.transform(test_categorical)
        
        # Create column names
        feature_names = []
        for i, col in enumerate(self.categorical_columns):
            categories = self.onehot_encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_df.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=test_df.index)
        
        return train_encoded_df, test_encoded_df
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete preprocessing pipeline"""
        # Encode categorical features
        train_encoded, test_encoded = self.encode_categorical_features(train_df, test_df)
        
        # Combine with numerical features
        numerical_cols = [col for col in FEATURE_NAMES if col not in self.categorical_columns + ['attack', 'level']]
        
        train_processed = train_df[numerical_cols].join(train_encoded)
        test_processed = test_df[numerical_cols].join(test_encoded)
        
        # Encode attack labels
        train_processed['attack'] = train_df['attack'].map(ATTACK_MAPPING)
        test_processed['attack'] = test_df['attack'].map(ATTACK_MAPPING)
        
        # Store feature columns
        self.feature_columns = [col for col in train_processed.columns if col != 'attack']
        
        return train_processed, test_processed
    
    def preprocess_for_evaluation_only(self, train_path: str, test_path: str) -> pd.DataFrame:
        """
        Load training data to fit transformers, then process only test data for evaluation
        This ensures proper preprocessing consistency while only returning test data
        """
        train_df = pd.read_csv(train_path, header=None, names=FEATURE_NAMES)
        test_df = pd.read_csv(test_path, header=None, names=FEATURE_NAMES)
        
        # Prepare basic features
        train_processed = self.prepare_basic_features(train_df)
        test_processed = self.prepare_basic_features(test_df)
        
        # Store feature columns (include level, exclude only attack)
        self.feature_columns = [col for col in FEATURE_NAMES if col not in ['attack']]
        
        # Encode categorical features using training data to fit transformers
        train_encoded, test_encoded = self.encode_categorical_features(train_processed, test_processed)
        
        # Get numerical features (including level)
        numerical_cols = [col for col in self.feature_columns if col not in self.categorical_columns]
        train_numerical = train_processed[numerical_cols].values
        test_numerical = test_processed[numerical_cols].values
        
        # Combine numerical and encoded categorical features
        train_combined = np.column_stack([train_numerical, train_encoded.values])
        test_combined = np.column_stack([test_numerical, test_encoded.values])
        
        # Standard scaling - fit on training data, transform test data
        scaler = StandardScaler()
        scaler.fit(train_combined)  # Fit on training data
        test_scaled = scaler.transform(test_combined)  # Transform only test data
        
        # Store scaler for all attack types
        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
            self.scalers[attack_type] = scaler
        
        # Create feature names
        all_feature_names = numerical_cols + list(train_encoded.columns)
        
        # Create final test DataFrame only
        test_final = pd.DataFrame(test_scaled, columns=all_feature_names)
        
        # Use already mapped attack labels (they were mapped in prepare_basic_features)
        test_final['attack'] = test_processed['attack']

        return test_final
    
    def create_attack_specific_datasets(self, df: pd.DataFrame, attack_type: int) -> pd.DataFrame:
        """Create binary dataset for specific attack type"""
        attack_mapping = {1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
        
        if attack_type not in attack_mapping:
            raise ValueError(f"Invalid attack type: {attack_type}")
        
        # Create binary dataset (normal vs specific attack)
        binary_df = df[df['attack'].isin([0, attack_type])].copy()
        binary_df['attack'] = (binary_df['attack'] == attack_type).astype(int)
        
        return binary_df
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray, attack_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[attack_type] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def select_features(self, X: np.ndarray, y: np.ndarray, attack_type: str, 
                       method: str = 'rfe', n_features: int = MODEL_CONFIG['n_features_rfe']) -> Tuple[np.ndarray, List[int]]:
        """Feature selection using RFE or SelectPercentile"""
        if method == 'rfe':
            clf = DecisionTreeClassifier(random_state=self.random_seed)
            selector = RFE(estimator=clf, n_features_to_select=n_features, step=1)
        elif method == 'percentile':
            selector = SelectPercentile(f_classif, percentile=10)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        
        self.feature_selectors[attack_type] = selector
        
        return X_selected, selected_features
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for handling class imbalance"""
        smote = SMOTE(
            sampling_strategy=MODEL_CONFIG['smote_sampling_strategy'],
            random_state=self.random_seed,
            k_neighbors=MODEL_CONFIG['smote_k_neighbors']
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return X_resampled, y_resampled
    
    def prepare_attack_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           attack_type: int) -> Dict[str, Any]:
        """Complete data preparation pipeline for specific attack type"""
        attack_name = {1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}[attack_type]
        
        # Create binary datasets
        train_binary = self.create_attack_specific_datasets(train_df, attack_type)
        test_binary = self.create_attack_specific_datasets(test_df, attack_type)
        
        # Separate features and labels
        X_train = train_binary.drop('attack', axis=1).values
        y_train = train_binary['attack'].values
        X_test = test_binary.drop('attack', axis=1).values
        y_test = test_binary['attack'].values
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, attack_name)
        
        # Feature selection
        X_train_selected, selected_features = self.select_features(
            X_train_scaled, y_train, attack_name
        )
        X_test_selected = X_test_scaled[:, selected_features]
        
        # Train-validation split
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_selected, y_train,
            test_size=MODEL_CONFIG['test_size'],
            stratify=y_train,
            random_state=self.random_seed
        )
        
        # Apply SMOTE to training data only
        X_train_smote, y_train_smote = self.apply_smote(X_train_final, y_train_final)
        
        return {
            'X_train': X_train_smote,
            'y_train': y_train_smote,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test_selected,
            'y_test': y_test,
            'selected_features': selected_features,
            'feature_names': [self.feature_columns[i] for i in selected_features] if self.feature_columns else None
        }
    
    def transform_new_data(self, data: pd.DataFrame, attack_type: str, skip_feature_selection: bool = False) -> np.ndarray:
        """Transform new data using fitted preprocessors"""
        if not self.feature_columns:
            raise ValueError("Preprocessor not fitted. Call preprocess_data first.")
        
        # Encode categorical features
        data_categorical = data[self.categorical_columns].copy()
        for col in self.categorical_columns:
            data_categorical[col] = self.label_encoders[col].transform(data_categorical[col])
        
        # One-hot encode
        data_encoded = self.onehot_encoder.transform(data_categorical)
        
        # Combine with numerical features
        numerical_cols = [col for col in FEATURE_NAMES if col not in self.categorical_columns + ['attack', 'level']]
        data_numerical = data[numerical_cols].values
        data_combined = np.column_stack([data_numerical, data_encoded])
        
        # Scale
        data_scaled = self.scalers[attack_type].transform(data_combined)
        
        # Optional feature selection - skip if models expect full feature set
        if not skip_feature_selection and attack_type in self.feature_selectors:
            data_selected = self.feature_selectors[attack_type].transform(data_scaled)
            return data_selected
    
    def preprocess_for_evaluation(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simplified preprocessing for evaluation with pre-trained models
        Only does encoding and scaling, no feature selection
        """
        # Basic data preparation
        train_processed = self.prepare_basic_features(train_df)
        test_processed = self.prepare_basic_features(test_df)
        
        # Store feature columns for later use
        feature_cols = [col for col in FEATURE_NAMES if col not in ['attack', 'level']]
        self.feature_columns = feature_cols
        
        # Encode categorical features in training data
        train_categorical = train_processed[self.categorical_columns].copy()
        for col in self.categorical_columns:
            le = LabelEncoder()
            train_categorical[col] = le.fit_transform(train_categorical[col])
            self.label_encoders[col] = le
        
        # One-hot encode categorical features
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        train_encoded = self.onehot_encoder.fit_transform(train_categorical)
        
        # Combine numerical and encoded categorical features for training
        numerical_cols = [col for col in feature_cols if col not in self.categorical_columns]
        train_numerical = train_processed[numerical_cols].values
        train_combined = np.column_stack([train_numerical, train_encoded])
        
        # Scale training data and fit scalers
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_combined)
        
        # Store the same scaler for all attack types (since we're not doing attack-specific processing)
        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
            self.scalers[attack_type] = scaler
        
        # Process test data using the same pipeline
        test_categorical = test_processed[self.categorical_columns].copy()
        for col in self.categorical_columns:
            test_categorical[col] = self.label_encoders[col].transform(test_categorical[col])
        
        test_encoded = self.onehot_encoder.transform(test_categorical)
        test_numerical = test_processed[numerical_cols].values
        test_combined = np.column_stack([test_numerical, test_encoded])
        test_scaled = scaler.transform(test_combined)
        
        # Create final DataFrames
        onehot_feature_names = []
        for i, col in enumerate(self.categorical_columns):
            categories = self.onehot_encoder.categories_[i]
            onehot_feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        all_feature_names = numerical_cols + onehot_feature_names
        
        train_final = pd.DataFrame(train_scaled, columns=all_feature_names, index=train_processed.index)
        train_final['attack'] = train_processed['attack'] 
        train_final['level'] = train_processed['level']
        
        test_final = pd.DataFrame(test_scaled, columns=all_feature_names, index=test_processed.index)
        test_final['attack'] = test_processed['attack']
        test_final['level'] = test_processed['level']
        
        return train_final, test_final
    
    def preprocess_for_evaluation(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocessing pipeline matching the exact notebook implementation
        - Full feature set (no feature selection)
        - Standard scaling only
        """
        print("Starting evaluation preprocessing (matching notebook pipeline)...")
        
        # Store feature columns (include level, exclude only attack)
        self.feature_columns = [col for col in FEATURE_NAMES if col not in ['attack']]
        
        # Encode categorical features using existing method
        train_encoded, test_encoded = self.encode_categorical_features(train_df, test_df)
        
        # Get numerical features (including level)
        numerical_cols = [col for col in self.feature_columns if col not in self.categorical_columns]
        train_numerical = train_df[numerical_cols].values
        test_numerical = test_df[numerical_cols].values
        
        # Combine numerical and encoded categorical features
        train_combined = np.column_stack([train_numerical, train_encoded.values])
        test_combined = np.column_stack([test_numerical, test_encoded.values])
        
        # Standard scaling - fit on training data only (matching notebook)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_combined)
        test_scaled = scaler.transform(test_combined)
        
        # Store scaler for all attack types
        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
            self.scalers[attack_type] = scaler
        
        # Create feature names
        all_feature_names = numerical_cols + list(train_encoded.columns)
        
        # Create final DataFrames
        train_final = pd.DataFrame(train_scaled, columns=all_feature_names)
        test_final = pd.DataFrame(test_scaled, columns=all_feature_names)
        
        # Convert attack labels from strings to numeric codes (matching notebook)
        train_final['attack'] = train_df['attack'].map(ATTACK_MAPPING)
        test_final['attack'] = test_df['attack'].map(ATTACK_MAPPING)
        
        print(f"Preprocessing complete: {len(all_feature_names)} features (matching notebook pipeline)")
        print(f"Attack label conversion: strings -> numeric codes")
        return train_final, test_final
