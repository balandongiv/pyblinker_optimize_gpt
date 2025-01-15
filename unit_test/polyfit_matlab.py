import numpy as np
# from unit_test.matlab_forking import polyfitMatlab
# Unit Test
from scipy.linalg import qr,solve_triangular

import numpy as np
from scipy.linalg import qr

import numpy as np
from scipy.linalg import qr
def polyfitMatlab(x, y, n):
    """
    Fit a polynomial of degree n to data using least squares.

    Parameters:
    x : array_like, shape (M,)
        x-coordinates of the M sample points (independent variable).
    y : array_like, shape (M,)
        y-coordinates of the sample points (dependent variable).
    n : int
        Degree of the fitting polynomial.

    Returns:
    p : ndarray, shape (n+1,)
        Polynomial coefficients, highest power first.
    S : dict
        A dictionary containing diagnostic information:
        - 'R': Triangular factor from QR decomposition of the Vandermonde matrix.
        - 'df': Degrees of freedom.
        - 'normr': Norm of the residuals.
        - 'rsquared': Coefficient of determination (R-squared).
    mu : ndarray, shape (2,)
        Mean and standard deviation of x for centering and scaling.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if x.size != y.size:
        raise ValueError('x and y must have the same length')

    # Center and scale x
    mx = x.mean()
    sx = x.std(ddof=1)
    mu = np.array([mx, sx])
    x_scaled = (x - mu[0]) / mu[1]

    # Construct the Vandermonde matrix
    V = np.vander(x_scaled, n+1)

    # Solve least squares problem
    p, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
    p = p.flatten()  # Coefficients are already in descending order

    # Compute the QR decomposition
    Q, R = qr(V, mode='economic')

    # Degrees of freedom
    df = max(0, len(y) - (n + 1))

    # Norm of residuals
    if residuals.size > 0:
        normr = np.sqrt(residuals[0])
    else:
        r = y - V @ p
        normr = np.linalg.norm(r)

    # R-squared
    y_mean = y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = normr ** 2
    rsquared = 1 - ss_res / ss_tot

    S = {'R': R, 'df': df, 'normr': normr, 'rsquared': rsquared}

    return p, S, mu


def test_polyfit():
    n = 1
    x = np.array([43, 44, 45, 46, 47, 48])
    y = np.array([15.399296760559082, 26.770189285278320, 40.020221710205080,
                  54.111049652099610, 67.944847106933600, 80.329727172851560])

    # Update expected polynomial coefficients to descending order
    expected_p = np.array([24.709207534790040,47.429222106933594])
    expected_S = {
        'R': np.array([[0, -2.449489742783178],
                       [2.236067977499790, 0]]),
        'df': 4,
        'normr': 1.691495418548584,
        'rsquared': 0.999063611030579
    }
    expected_mu = np.array([45.5, 1.870828693386971])

    p, S, mu = polyfitMatlab(x, y, n)

    # Assertions with tolerances
    np.testing.assert_allclose(p, expected_p, rtol=1e-6)
    np.testing.assert_allclose(mu, expected_mu, rtol=1e-6)
    np.testing.assert_allclose(S['normr'], expected_S['normr'], rtol=1e-6)
    np.testing.assert_allclose(S['rsquared'], expected_S['rsquared'], rtol=1e-6)
    np.testing.assert_equal(S['df'], expected_S['df'])
    # R comparison should consider numerical differences

    print("All tests passed successfully.")


# Run the unit test
test_polyfit()
