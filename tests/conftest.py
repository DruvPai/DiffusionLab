import pytest
import torch
import random
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seeds():
    """
    Fixture to set random seeds before each test.
    The autouse=True means this fixture will be automatically used by all tests.
    """
    # Set seeds for all random number generators
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    # Handle CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
        # For completely reproducible results on CUDA, you might also want:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    yield  # This is where the test runs

    # After test cleanup (if needed)
    pass
