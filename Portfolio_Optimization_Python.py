import numpy as np
from scipy.optimize import linprog

# Given Data
prices = np.array([109, 94.8, 99.5, 93.1, 97.2, 96.3, 92.9, 110, 104, 101, 107, 102])  # Prices of bonds
spot_rates = np.array([0.0504, 0.0594, 0.0636, 0.0718, 0.0789, 0.0839])  # Spot rates (in decimal form)
liabilities = np.array([500, 200, 800, 200, 800, 1200])  # Liabilities at each time t
cash_flows = np.array([
    [10, 7, 8, 6, 7, 6, 5, 10, 8, 6, 10, 7],
    [10, 7, 8, 6, 7, 6, 5, 10, 8, 6, 110, 107],
    [10, 7, 8, 6, 7, 6, 5, 110, 108, 106, 0, 0],
    [10, 7, 8, 6, 7, 106, 105, 0, 0, 0, 0, 0],
    [10, 7, 8, 106, 107, 0, 0, 0, 0, 0, 0, 0],
    [110, 107, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])  # Cash flows for each bond at each time t

# Compute PV and DD of liabilities
pv_liabilities = np.sum(liabilities / (1 + spot_rates)**np.arange(1, len(liabilities) + 1))
dd_liabilities = np.sum((np.arange(1, len(liabilities) + 1) * liabilities) / (1 + spot_rates)**(np.arange(1, len(liabilities) + 1) + 1))

# Compute PV and DD of each bond
pv_bonds = np.sum(cash_flows / (1 + spot_rates[:, None])**np.arange(1, len(liabilities) + 1)[:, None], axis=0)
dd_bonds = np.sum((np.arange(1, len(liabilities) + 1)[:, None] * cash_flows) / (1 + spot_rates[:, None])**(np.arange(1, len(liabilities) + 1)[:, None] + 1), axis=0)

# Formulate the linear programming problem
c = prices  # Minimize cost
A_eq = np.vstack([pv_bonds, dd_bonds])  # PV and DD constraints
b_eq = np.array([pv_liabilities, dd_liabilities])  # Target PV and DD
bounds = [(0, None)] * len(prices)  # Non-negativity constraints

# Solve the problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# Results
if result.success:
    print("Optimal Portfolio Cost:", result.fun)
    print("Units of Each Bond to Purchase:", result.x)
else:
    print("Optimization failed.")
