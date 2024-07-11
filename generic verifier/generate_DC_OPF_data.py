import numpy as np
import pandas as pd
import cvxpy as cp
from system_data import n_buses, n_gens, n_lines, X, gen_limits, cost_coeffs, min_demand, max_demand

def generate_DC_OPF_data(num_scenarios, seed=42):
    """
    Generates and solves power flow optimization problems for a given number of scenarios.

    Parameters:
        num_scenarios (int): Number of random demand scenarios to generate.
        seed (int, optional): Seed for random number generator for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the optimization.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate random demand scenarios for each bus
    demand_scenarios = np.random.randint(min_demand, max_demand, size=(num_scenarios, n_buses))

    # Initialize a DataFrame to store results
    columns = ['Demand1', 'Demand2', 'Demand3', 'Gen1', 'Gen2', 'Cost']
    results_list = []

    # Optimization model for each demand scenario
    for Pd in demand_scenarios:
        # Decision variables for bus angles and generation outputs
        theta = cp.Variable(n_buses)
        Pg = cp.Variable(n_gens)

        # Quadratic objective function for generation cost minimization
        objective = cp.Minimize(sum(cost_coeffs[i][0] * Pg[i] ** 2 + cost_coeffs[i][1] * Pg[i] + cost_coeffs[i][2] for i in range(n_gens)))

        # Power flow calculations between buses
        Pf = [(theta[i] - theta[j]) / X[i] for i, j in zip(range(n_lines), range(1, n_lines + 1))]

        # Constraints setup
        constraints = [
            Pg[0] - Pd[0] + Pf[0] == 0,  # Power balance at Bus 1
            -Pf[0] + Pf[1] - Pd[1] == 0, # Power balance at Bus 2
            Pg[1] - Pd[2] - Pf[1] == 0   # Power balance at Bus 3
        ]
        
        # Generation limits
        constraints += [Pg[i] >= gen_limits[i][0] for i in range(n_gens)] + [Pg[i] <= gen_limits[i][1] for i in range(n_gens)]

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Store results
        if problem.status == cp.OPTIMAL:
            results_list.append(pd.Series([Pd[0], Pd[1], Pd[2], Pg.value[0], Pg.value[1], problem.value], index=columns))
        else:
            results_list.append(pd.Series([Pd[0], Pd[1], Pd[2], None, None, None], index=columns))

    # Concatenate all results into the DataFrame
    df_results = pd.DataFrame(results_list, columns=columns)

    return df_results


