import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d


def make_some_data(loc=0, scale=1, N=10_000):
    return np.random.normal(loc, scale, size=N)


X = make_some_data()
X = np.sort(X)
Y = np.arange(1, len(X) + 1) / len(X)

spline = UnivariateSpline(X, Y, k=2, s=0.01, ext=0)
d_spline = spline.derivative()


grid = np.linspace(X.min(), X.max(), 1000)
plt.hist(X, density=True, bins=100)
plt.plot(grid, d_spline(grid))
plt.show()

# some


# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([12, 14, 22, 39, 58, 77])
# cs = CubicSpline(x, y, bc_type='natural')

# xs = np.linspace(0, 5, 100)
# fig, ax = plt.subplots(figsize=(6.5, 4))
# ax.plot(x, y, 'o', label='Data Points')
# ax.plot(xs, cs(xs), label='Spline')
# ax.plot(xs, cs(xs, 1), label="First Derivative")
# ax.plot(xs, cs(xs, 2), label="Second Derivative")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.legend(loc='best')
# plt.show()


# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([12, 14, 22, 39, 58, 77])
# from scipy.interpolate import UnivariateSpline


# spline = UnivariateSpline(x,y, k=4, s=0)
# d_spline = spline.derivative()
# x_eval = np.linspace(0, 5, 100)

# plt.figure(figsize=(10, 6))
# plt.plot(x, y, 'o', label='Data Points')
# plt.plot(x_eval, spline(x_eval), '-', label='Linear Spline')
# plt.plot(x_eval, d_spline(x_eval), '--', label='Derivative of Linear Spline')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Linear Spline and Its Derivative')
# plt.show()


# # Create linear spline interpolation function
# linear_interp = interp1d(x, y, kind='quadratic', fill_value="extrapolate")

# # Compute slopes (derivatives) between consecutive data points
# slopes = np.diff(y) / np.diff(x)

# # Function to evaluate the derivative at any point
# def linear_spline_derivative(x_eval):
#     # Ensure x_eval is an array for consistent processing
#     x_eval = np.atleast_1d(x_eval)
#     derivatives = np.zeros_like(x_eval, dtype=float)
#     for i, val in enumerate(x_eval):
#         # Find the segment index where x_eval[i] belongs
#         segment_index = np.searchsorted(x, val) - 1
#         # Handle boundary cases
#         if segment_index < 0:
#             segment_index = 0
#         elif segment_index >= len(slopes):
#             segment_index = len(slopes) - 1
#         # Assign the corresponding slope
#         derivatives[i] = slopes[segment_index]
#     return derivatives

# # Points to evaluate
# x_eval = np.linspace(0, 5, 100)
# y_eval = linear_interp(x_eval)
# y_deriv = linear_spline_derivative(x_eval)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, 'o', label='Data Points')
# plt.plot(x_eval, y_eval, '-', label='Linear Spline')
# plt.plot(x_eval, y_deriv, '--', label='Derivative of Linear Spline')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Linear Spline and Its Derivative')
# plt.show()
