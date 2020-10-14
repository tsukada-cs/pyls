import unittest
from unittest import mock

class TestLineSegment(unittest.TestCase):
    """Test the LineSegment class."""
    def test__init__(self, *mocks):
        from pyls import LineSegment
        x1 = [0,2]
        y1 = [1,3]
        x2 = [0,2]
        y2 = [1,3]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.lines["x1"]), x1)
        self.assertEqual(list(ls.lines["x2"]), x2)
        self.assertEqual(list(ls.lines["y1"]), y1)
        self.assertEqual(list(ls.lines["y2"]), y2)
        self.assertEqual(len(ls), 2)

    def test__add__(self, *mocks):
        from pyls import LineSegment
        x1 = [0,1]
        y1 = [4,5]
        x2 = [2,3]
        y2 = [6,7]
        ls1 = LineSegment(x1, x2, y1, y2)
        ls2 = LineSegment(x2, x1, y2, y1)
        self.assertEqual(list((ls1 + ls2).lines["x1"]), [0,1,2,3])
        self.assertEqual(list((ls1 + ls2).lines["x2"]), [2,3,0,1])
        self.assertEqual(list((ls1 + ls2).lines["y1"]), [4,5,6,7])
        self.assertEqual(list((ls1 + ls2).lines["y2"]), [6,7,4,5])
        
    def test_length(self, *mocks):
        from pyls import LineSegment
        x1 = [0,2]
        y1 = [0,5]
        x2 = [0,2]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.length), [1, 3])

    def test_x_length(self, *mocks):
        from pyls import LineSegment
        x1 = [0,2]
        y1 = [0,5]
        x2 = [0,2]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.x_length), [0, 0])

    def test_y_length(self, *mocks):
        from pyls import LineSegment
        x1 = [0,2]
        y1 = [0,5]
        x2 = [0,2]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.y_length), [1, 3])

    def test_get_left(self, *mocks):
        import numpy as np
        from pyls import LineSegment
        img = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ])
        x1 = [1,1]
        y1 = [0,0]
        x2 = [1,1]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.get_left(img, gap=1)), [3.5, 5])
        
    def test_get_right(self, *mocks):
        import numpy as np
        from pyls import LineSegment
        img = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ])
        x1 = [1,1]
        y1 = [0,0]
        x2 = [1,1]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.get_right(img, gap=1)), [1.5, 3])
        
    def test_get_center(self, *mocks):
        import numpy as np
        from pyls import LineSegment
        img = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ])
        x1 = [1,1]
        y1 = [0,0]
        x2 = [1,1]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.get_right(img, gap=0)), [2.5, 4])

    def test_get_high(self, *mocks):
        import numpy as np
        from pyls import LineSegment
        img = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ])
        x1 = [1,1]
        y1 = [0,0]
        x2 = [1,1]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.get_high(img, gap=1)), [3.5, 5])

    def test_get_low(self, *mocks):
        import numpy as np
        from pyls import LineSegment
        img = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ])
        x1 = [1,1]
        y1 = [0,0]
        x2 = [1,1]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        self.assertEqual(list(ls.get_low(img, gap=1)), [1.5, 3])

    def test_limit_value_min(self, *mocks):
        import numpy as np
        from pyls import LineSegment
        img = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ])
        x1 = [1,1]
        y1 = [0,0]
        x2 = [1,1]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        ls = ls.limit_value(img, min_value=2, gap=1, which="low")
        self.assertEqual(list(ls.lines["x1"]), [1])
        self.assertEqual(list(ls.lines["x2"]), [1])
        self.assertEqual(list(ls.lines["y1"]), [0])
        self.assertEqual(list(ls.lines["y2"]), [2])

    def test_limit_value_max(self, *mocks):
        import numpy as np
        from pyls import LineSegment
        img = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ])
        x1 = [1,1]
        y1 = [0,0]
        x2 = [1,1]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        ls = ls.limit_value(img, max_value=2, gap=1, which="low")
        self.assertEqual(list(ls.lines["x1"]), [1])
        self.assertEqual(list(ls.lines["x2"]), [1])
        self.assertEqual(list(ls.lines["y1"]), [0])
        self.assertEqual(list(ls.lines["y2"]), [1])

    def test_sort_x(self, *mocks):
        import numpy as np
        from pyls import LineSegment
        x1 = [3,2]
        y1 = [0,0]
        x2 = [1,3]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        ls = ls.sort("x", large_2=True)
        self.assertEqual(list(ls.lines["x1"]), [1,2])
        self.assertEqual(list(ls.lines["x2"]), [3,3])
        self.assertEqual(list(ls.lines["y1"]), [1,0])
        self.assertEqual(list(ls.lines["y2"]), [0,2])

    def test_sort_y(self, *mocks):
        from pyls import LineSegment
        x1 = [3,2]
        y1 = [0,0]
        x2 = [1,3]
        y2 = [1,2]
        ls = LineSegment(x1, x2, y1, y2)
        ls = ls.sort("y", large_2=False)
        self.assertEqual(list(ls.lines["x1"]), [1,3])
        self.assertEqual(list(ls.lines["x2"]), [3,2])
        self.assertEqual(list(ls.lines["y1"]), [1,2])
        self.assertEqual(list(ls.lines["y2"]), [0,0])

    def test_limit_x(self, *mocks):
        from pyls import LineSegment
        x1 = [0,0]
        y1 = [0,0]
        x2 = [3,5]
        y2 = [0,0]
        ls = LineSegment(x1, x2, y1, y2)
        ls0 = ls.limit_xy(which="x", xymin=4)
        ls1 = ls.limit_x(xmin=4)
        ls2 = ls.limit_x(xmax=4, crop=False)
        ls3 = ls.limit_x(xmax=4, crop=True)
        self.assertEqual(list(ls0.lines["x2"]), [5])
        self.assertEqual(list(ls1.lines["x2"]), [5])
        self.assertEqual(list(ls2.lines["x2"]), [3,5])
        self.assertEqual(list(ls3.lines["x2"]), [3,4])

    def test_limit_y(self, *mocks):
        from pyls import LineSegment
        x1 = [0,0]
        y1 = [0,0]
        x2 = [0,0]
        y2 = [3,5]
        ls = LineSegment(x1, x2, y1, y2)
        ls0 = ls.limit_xy(which="y", xymin=4)
        ls1 = ls.limit_y(ymin=4)
        ls2 = ls.limit_y(ymax=4, crop=False)
        ls3 = ls.limit_y(ymax=4, crop=True)
        self.assertEqual(list(ls0.lines["y2"]), [5])
        self.assertEqual(list(ls1.lines["y2"]), [5])
        self.assertEqual(list(ls2.lines["y2"]), [3,5])
        self.assertEqual(list(ls3.lines["y2"]), [3,4])