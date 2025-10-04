"""
Unit tests for the house purchase prediction pipeline.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import (
    check_data_quality,
    encode_categorical_features,
    engineer_features,
    split_data,
)


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions."""

    def setUp(self):
        """Create sample data for testing."""
        # Create 100 samples for better testing
        np.random.seed(42)
        n_samples = 100

        self.sample_data = pd.DataFrame(
            {
                "property_id": range(1, n_samples + 1),
                "country": np.random.choice(
                    ["USA", "UK", "France", "Germany"], n_samples
                ),
                "city": np.random.choice(
                    ["New York", "London", "Paris", "Berlin"], n_samples
                ),
                "property_type": np.random.choice(
                    ["Apartment", "Villa", "Townhouse", "Studio"], n_samples
                ),
                "furnishing_status": np.random.choice(
                    ["Furnished", "Unfurnished", "Semi-Furnished"], n_samples
                ),
                "property_size_sqft": np.random.randint(800, 3000, n_samples),
                "price": np.random.randint(300000, 1500000, n_samples),
                "constructed_year": np.random.randint(2000, 2023, n_samples),
                "previous_owners": np.random.randint(0, 5, n_samples),
                "rooms": np.random.randint(2, 7, n_samples),
                "bathrooms": np.random.randint(1, 5, n_samples),
                "garage": np.random.randint(0, 2, n_samples),
                "garden": np.random.randint(0, 2, n_samples),
                "crime_cases_reported": np.random.randint(0, 3, n_samples),
                "legal_cases_on_property": np.random.randint(0, 2, n_samples),
                "customer_salary": np.random.randint(80000, 300000, n_samples),
                "loan_amount": np.random.randint(200000, 1200000, n_samples),
                "loan_tenure_years": np.random.choice([15, 20, 25, 30], n_samples),
                "monthly_expenses": np.random.randint(2000, 8000, n_samples),
                "down_payment": np.random.randint(50000, 300000, n_samples),
                "emi_to_income_ratio": np.random.uniform(0.2, 0.4, n_samples),
                "satisfaction_score": np.random.randint(1, 11, n_samples),
                "neighbourhood_rating": np.random.randint(1, 11, n_samples),
                "connectivity_score": np.random.randint(1, 11, n_samples),
                "decision": np.random.choice([0, 1], n_samples),
            }
        )

    def test_engineer_features(self):
        """Test feature engineering."""
        df_engineered = engineer_features(self.sample_data)

        # Check new features exist
        self.assertIn("property_age", df_engineered.columns)
        self.assertIn("price_per_sqft", df_engineered.columns)
        self.assertIn("loan_to_price_ratio", df_engineered.columns)
        self.assertIn("affordability_score", df_engineered.columns)
        self.assertIn("total_rooms", df_engineered.columns)

        # Check calculations are correct (using first row)
        original_row = self.sample_data.iloc[0]
        engineered_row = df_engineered.iloc[0]

        self.assertEqual(
            engineered_row["property_age"], 2025 - original_row["constructed_year"]
        )
        self.assertEqual(
            engineered_row["price_per_sqft"],
            original_row["price"] / original_row["property_size_sqft"],
        )
        self.assertEqual(
            engineered_row["total_rooms"],
            original_row["rooms"] + original_row["bathrooms"],
        )

    def test_encode_categorical_features(self):
        """Test categorical encoding."""
        categorical_cols = ["country", "city", "property_type", "furnishing_status"]
        df_encoded, encoders = encode_categorical_features(
            self.sample_data, categorical_cols, fit=True
        )

        # Check encoding
        for col in categorical_cols:
            self.assertTrue(df_encoded[col].dtype in [np.int32, np.int64])

        # Check encoders saved
        self.assertEqual(len(encoders), len(categorical_cols))

    def test_split_data(self):
        """Test data splitting."""
        df = engineer_features(self.sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, target_col="decision", test_size=0.2, val_size=0.2, random_state=42
        )

        # Check shapes
        total_samples = len(df)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)

        # Check no data leakage (no overlap in indices)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        self.assertEqual(len(train_idx & val_idx), 0)
        self.assertEqual(len(train_idx & test_idx), 0)
        self.assertEqual(len(val_idx & test_idx), 0)

    def test_check_data_quality(self):
        """Test data quality checks."""
        quality_report = check_data_quality(self.sample_data)

        self.assertIn("shape", quality_report)
        self.assertIn("missing_values", quality_report)
        self.assertIn("duplicates", quality_report)
        self.assertIn("target_distribution", quality_report)

        self.assertEqual(quality_report["shape"], (100, 25))
        self.assertEqual(quality_report["duplicates"], 0)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering logic."""

    def test_affordability_calculation(self):
        """Test affordability score calculation."""
        df = pd.DataFrame(
            {
                "customer_salary": [100000, 200000],
                "price": [500000, 1000000],
                "constructed_year": [2020, 2010],
                "property_size_sqft": [1000, 2000],
                "loan_amount": [400000, 800000],
                "down_payment": [100000, 200000],
                "rooms": [3, 5],
                "bathrooms": [2, 3],
                "garage": [1, 1],
                "garden": [0, 1],
                "crime_cases_reported": [0, 1],
                "legal_cases_on_property": [0, 0],
                "monthly_expenses": [3000, 5000],
                "loan_tenure_years": [30, 25],
            }
        )

        df_engineered = engineer_features(df)

        # Check affordability score
        expected_affordability = [100000 / 500000, 200000 / 1000000]
        np.testing.assert_array_almost_equal(
            df_engineered["affordability_score"].values, expected_affordability
        )

        # Check risk score
        expected_risk = [0, 1]
        np.testing.assert_array_equal(df_engineered["risk_score"].values, expected_risk)


class TestModelInputOutput(unittest.TestCase):
    """Test model input/output consistency."""

    def test_feature_shape_consistency(self):
        """Test that feature shapes remain consistent through pipeline."""
        # This test ensures no features are lost during preprocessing
        sample_data = pd.DataFrame(
            {
                "property_id": [1],
                "country": ["USA"],
                "city": ["NY"],
                "property_type": ["Apartment"],
                "furnishing_status": ["Furnished"],
                "property_size_sqft": [1000],
                "price": [500000],
                "constructed_year": [2010],
                "previous_owners": [2],
                "rooms": [3],
                "bathrooms": [2],
                "garage": [1],
                "garden": [0],
                "crime_cases_reported": [0],
                "legal_cases_on_property": [0],
                "customer_salary": [100000],
                "loan_amount": [400000],
                "loan_tenure_years": [30],
                "monthly_expenses": [3000],
                "down_payment": [100000],
                "emi_to_income_ratio": [0.3],
                "satisfaction_score": [8],
                "neighbourhood_rating": [7],
                "connectivity_score": [8],
                "decision": [1],
            }
        )

        # Apply feature engineering
        df_engineered = engineer_features(sample_data)

        # Original features (25) - property_id - decision + 10 engineered = 33
        expected_feature_count = 25 - 1 - 1 + 10  # 33
        actual_feature_count = (
            len(df_engineered.columns) - 2
        )  # -2 for property_id and decision

        self.assertEqual(actual_feature_count, expected_feature_count)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestModelInputOutput))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
