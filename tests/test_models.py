import unittest
import paddle
from src.models.model import SimpleClassifier

class TestModels(unittest.TestCase):
    """测试模型模块"""

    def test_simple_classifier(self):
        model = SimpleClassifier(input_size=24, hidden_size=64, output_size=2)
        x = paddle.randn([10, 24], dtype='float32')
        out = model(x)
        self.assertEqual(out.shape, [10, 2])

if __name__ == '__main__':
    unittest.main()

