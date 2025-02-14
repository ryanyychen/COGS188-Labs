import random
import unittest
from src.estimate_pi import estimate_pi, create_animation_pi


class TestEstimatePi(unittest.TestCase):
    def setUp(self):
        # Optionally set a seed for reproducibility in tests that depend on randomness.
        random.seed(42)

    def test_estimate_pi_output_length(self):
        """Test that estimate_pi returns the correct number of snapshots."""
        num_samples = 500
        step = 100
        data = estimate_pi(num_samples, step)
        self.assertEqual(len(data), num_samples // step)

    def test_estimate_pi_data_structure(self):
        """Test that each snapshot in the data is a tuple with the expected structure."""
        num_samples = 200
        step = 50
        data = estimate_pi(num_samples, step)
        for frame in data:
            self.assertIsInstance(frame, tuple)
            self.assertEqual(len(frame), 5)
            x_inside, y_inside, x_outside, y_outside, pi_estimate = frame
            self.assertIsInstance(x_inside, list)
            self.assertIsInstance(y_inside, list)
            self.assertIsInstance(x_outside, list)
            self.assertIsInstance(y_outside, list)
            self.assertIsInstance(pi_estimate, float)
            # Check that the total points equals the iteration count for this snapshot
            total_points = len(x_inside) + len(x_outside)
            # The snapshot index (starting from 1)
            index = data.index(frame) + 1
            self.assertEqual(total_points, step * index)

    def test_pi_estimate_reasonable_range(self):
        """Test that the pi_estimate is within a reasonable range after a simulation."""
        num_samples = 1000
        step = 250
        data = estimate_pi(num_samples, step)
        # Check the final estimate is within an expected range (Monte Carlo estimates can be off)
        _, _, _, _, final_pi_estimate = data[-1]
        self.assertTrue(3.0 <= final_pi_estimate <= 3.3)
