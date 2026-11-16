"""Unit tests for fixed_program.py (iris logistic regression)."""
import unittest

import numpy as np

from fixed_program import load_data, preprocess, train_model

FEATURES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


class TestFixedProgram(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = preprocess(cls.df)

    def test_data_loads_with_expected_shape(self):
        """The iris data has 150 rows: 4 feature columns plus the target."""
        self.assertEqual(self.df.shape[0], 150)
        self.assertIn('target', self.df.columns)

    def test_correct_column_names_exist(self):
        """The real iris column names (with units) are present, so preprocess() can find them."""
        for col in FEATURES:
            self.assertIn(col, self.df.columns)

    def test_no_missing_values(self):
        self.assertEqual(int(self.df.isnull().sum().sum()), 0)

    def test_split_sizes(self):
        """80/20 split of 150 rows gives 120 training and 30 test rows."""
        self.assertEqual(self.X_train.shape, (120, 4))
        self.assertEqual(self.X_test.shape, (30, 4))
        self.assertEqual(len(self.y_train), 120)
        self.assertEqual(len(self.y_test), 30)

    def test_scaling_returns_transformed_array(self):
        """fit_transform returns an array that is roughly centred on 0 (not a scaler object)."""
        self.assertIsInstance(self.X_train, np.ndarray)
        self.assertLess(abs(float(self.X_train.mean())), 0.5)

    def test_model_accuracy(self):
        """Logistic regression should classify iris well."""
        acc = train_model(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsInstance(acc, float)
        self.assertGreater(acc, 0.9)


if __name__ == '__main__':
    unittest.main()
