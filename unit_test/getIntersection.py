'''

Matlab original code

  function [xIntersect, yIntersect, xIntercept1, xIntercept2] = getIntersection(p, q, u, v)
        % Return intersection of two lines given by fits p and q centered at u and v
        %
        %  Parameters:
        %      p   first linear fit:  y = p(1)*x + p(2)
        %      q   second linear fit:  y = q(1)*x + q(2)
        %      u   u(1) mean of first, u(2) std of first
        %      v   v(1) mean of first, v(2) std of first
        %
        if p(1) == 0
            xIntercept1 = NaN;
        else
            xIntercept1 = (p(1)*u(1) - p(2)*u(2))./p(1);
        end
        if q(1) == 0
            xIntercept2 = NaN;
        else
            xIntercept2 = (q(1)*v(1) - q(2)*v(2))./q(1);
        end
        if p(1) == p(2)
            xIntersect = NaN;
            yIntersect = NaN;
        else
            denom = p(1)*v(2) - q(1)*u(2);
            numer =  u(1)*p(1)*v(2) - v(1)*q(1)*u(2) ...
                + q(2)*v(2)*u(2) - p(2)*u(2)*v(2);
            xIntersect = numer./denom;
            yIntersect = p(1)*(xIntersect - u(1))./u(2) + p(2);
        end
'''


import unittest

import numpy as np
from pyblinkers.matlab_forking import get_intersection
class TestGetIntersection(unittest.TestCase):

    def test_get_intersection(self):
        # Test input
        p = [24.709207534790040, 47.429222106933594]
        q = [-23.652940750122070, 46.986415863037110]
        u = [45.500000000000000, 1.870828693386971]
        v = [59.500000000000000, 4.183300132670378]

        # Expected output
        expected_xIntercept1 = 41.90895080566406
        expected_xIntercept2 = 67.81009674072266
        expected_xIntersect = 49.67326354980469
        expected_yIntersect = 102.5481115

        # Actual output
        xIntersect, yIntersect, xIntercept1, xIntercept2 = get_intersection(p, q, u, v)

        # Assert the results (up to 3 decimal places)
        self.assertAlmostEqual(xIntercept1, expected_xIntercept1, places=3, msg="xIntercept1 mismatch")
        self.assertAlmostEqual(xIntercept2, expected_xIntercept2, places=3, msg="xIntercept2 mismatch")
        self.assertAlmostEqual(xIntersect, expected_xIntersect, places=3, msg="xIntersect mismatch")
        self.assertAlmostEqual(yIntersect, expected_yIntersect, places=3, msg="yIntersect mismatch")

    def test_parallel_lines(self):
        # Parallel lines, no intersection
        p = [2, 1]
        q = [2, 5]
        u = [0, 1]
        v = [0, 1]

        # Expected output
        expected_xIntersect = np.nan
        expected_yIntersect = np.nan

        # Actual output
        xIntersect, yIntersect, xIntercept1, xIntercept2 = get_intersection(p, q, u, v)

        # Assert the results
        self.assertTrue(np.isnan(xIntersect), "xIntersect should be NaN for parallel lines")
        self.assertTrue(np.isnan(yIntersect), "yIntersect should be NaN for parallel lines")

    def test_zero_slope(self):
        # One line with zero slope
        p = [0, 2]
        q = [1, 1]
        u = [0, 1]
        v = [0, 1]

        # Expected output
        expected_xIntercept1 = np.nan

        # Actual output
        xIntersect, yIntersect, xIntercept1, xIntercept2 = get_intersection(p, q, u, v)

        # Assert the results
        self.assertTrue(np.isnan(xIntercept1), "xIntercept1 should be NaN for zero slope")

if __name__ == '__main__':
    unittest.main()
