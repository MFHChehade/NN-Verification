n_buses = 9
n_lines = 9
n_gens = 3

# Line reactances, explicitly indicating connected buses

# make X a dictionary for easier access
# make X index less by 1 

X = {(3, 4): 0.0850,
     (3, 5): 0.0920,
     (6, 4): 0.1610,
     (5, 8): 0.1700,
     (6, 7): 0.0720,
     (7, 8): 0.1008,
     (0, 3): 0.0576,
     (1, 6): 0.0625,
     (2, 8): 0.0586}


gen_buses = [0, 1, 2]  # Assuming generator buses are explicitly defined
demand_buses = [4, 5, 7]  # Assuming demand buses are explicitly defined

# Generation limits (Pmin, Pmax) from MATLAB script
gen_limits = [(30, 100), (60, 200), (30, 100)]  # Adjusted for simplicity

# Cost coefficients (c, b, a), assuming MATLAB's (A, B, C) needs to be reordered for Python
cost_coeffs = [(0.028, 15, 640), (0.020, 12, 1230), (0.028, 15, 640)]

# Nominal demand at each bus, taken from MATLAB's nodes
demand = [0, 0, 0, 0, 125, 90, 0, 100, 0]  # Only the specified demands from MATLAB, assumed as nominal

