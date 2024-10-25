import unittest
from src.utils.utils import set_seed
import numpy as np

class TestUtils(unittest.TestCase):
    """测试工具模块"""

    def test_set_seed(self):
        set_seed(42)
        a = np.random.rand()
        set_seed(42)
        b = np.random.rand()
        self.assertEqual(a, b)

if __name__ == '__main__':
    unittest.main()

