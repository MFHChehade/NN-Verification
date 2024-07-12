import pandas as pd
import numpy as np
import cvxpy as cp
import importlib
import pyomo.environ as pyo
from pyomo.environ import Var, ConstraintList
from plotting_norms import plot_3d_norm_balls

class DC_OPF:
    # some useful DC_OPF functions

    def __init__(self, network):

        system_data = importlib.import_module(network)
        self.system_data = system_data
        self.n_buses = system_data.n_buses
        self.n_gens = system_data.n_gens
        self.n_lines = system_data.n_lines
        self.gen_limits = system_data.gen_limits
        self.demand = system_data.demand
        self.gen_buses = system_data.gen_buses
        self.demand_buses = system_data.demand_buses
        self.cost_coeffs = system_data.cost_coeffs
        self.reactance = system_data.X
        
        self.input_size = len(self.demand_buses)
        self.output_size = len(self.gen_buses)

        self.num_constraints = 2 * self.n_gens + 2 * self.n_buses
        self.constraint_description = self._constraint_description()

    def input_output(self, data_path):
        # Load the dataset
        df_results = pd.read_csv(data_path)
        print(df_results.columns)  # Debugging: Print columns to ensure correct data handling

        # Prepare input features and target outputs
        self.X = df_results[['Demand'+str(i) for i in range(1, len(self.demand_buses)+1)]].values
        self.y = df_results[['Gen'+str(i) for i in range(1, self.n_gens+1)]].values 

        return self.X, self.y

    def generate_data(self, num_scenarios, seed=42):
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
        load_demand = np.zeros(len(self.demand_buses))
        for i in range(len(self.demand_buses)):
            load_demand[i] = self.demand[self.demand_buses[i]]
        demand_scenarios = np.random.uniform(np.array(load_demand) -  20, np.array(load_demand) + 20, size=(num_scenarios, len(self.demand_buses)))  # Uniform for simplicity, modify as needed

        # Initialize a DataFrame to store results
        columns = ['Scenario'] + [f'Demand{b+1}' for b in range(len(self.demand_buses))] + [f'Gen{g+1}' for g in range(self.n_gens)] + ['Cost']
        results_list = []

        # Optimization model for each demand scenario
        for scenario_index, Pd in enumerate(demand_scenarios):
            # Decision variables for generator outputs and slack bus voltage angle (set slack bus angle to zero)
            Pg = cp.Variable(self.n_gens)
            theta = cp.Variable(self.n_buses)

            demand_expanded = np.zeros(self.n_buses)
            for i in range(len(self.demand_buses)):
                demand_expanded[self.demand_buses[i]] = Pd[i]

            # Power flow calculations
            Pf = [cp.inv_pos(self.reactance[(i, j)]) * (theta[i] - theta[j]) for i, j in self.reactance.keys()]

            # Objective: Minimize generation cost (quadratic cost function)
            cost_function = sum(self.cost_coeffs[i][0] * Pg[i]**2 + self.cost_coeffs[i][1] * Pg[i] + self.cost_coeffs[i][2] for i in range(self.n_gens))
            objective = cp.Minimize(cost_function)

            # Constraints
            constraints = []

            # Power balance equations for each bus
            for b in range(self.n_buses):
                power_in = sum(Pf[k] for k, (i, j) in enumerate(self.reactance.keys()) if j == b)
                power_out = sum(Pf[k] for k, (i, j) in enumerate(self.reactance.keys()) if i == b)
                generation = sum(Pg[g] for g in range(self.n_gens) if self.gen_buses[g] == b)  # Assuming `gen_buses` lists bus index for each generator
                demand_power = demand_expanded[b]

                constraints.append(power_in - power_out + generation - demand_power == 0)

            # Generation limits
            for g in range(self.n_gens):
                constraints.append(Pg[g] >= self.gen_limits[g][0])
                constraints.append(Pg[g] <= self.gen_limits[g][1])

            # Solve the optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # Collect results
            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                result = [scenario_index] + list(Pd) + list(Pg.value) + [problem.value]
            else:
                result = [scenario_index] + list(Pd) + [None]*self.n_gens + [None]

            results_list.append(pd.Series(result, index=columns))

        # Return results as a DataFrame
        results_df = pd.DataFrame(results_list)

        # convert to csv
        results_df.to_csv('DC-OPF/DC-OPF_data.csv', index=False)

    
    def add_constraint(self, model, index, margin, center):
        """
        Adds a specified constraint to a Pyomo model based on the index provided.

        Parameters:
            model (pyomo.core.base.PyomoModel.ConcreteModel): The Pyomo model to which the constraint will be added.
            index (int): Index specifying which constraint to add.
            margin (float): Margin value to use in bound constraints.
            center (list): List of center values used to calculate dynamic upper and lower bounds.
            max_gen1 (float): Upper bound for generator 1.
            min_gen1 (float): Lower bound for generator 1.
            max_gen2 (float): Upper bound for generator 2.
            min_gen2 (float): Lower bound for generator 2.

        Raises:
            AssertionError: If the model does not have the required attributes.
            ValueError: If the index is out of range or if the model has missing variables.
        """

        # Validate the index and model requirements
        assert 0 <= index <= self.num_constraints, "Index must be between 0 and {self.num_constraints - 1}."
        required_attrs = ['input', 'output']
        for attr in required_attrs:
            assert hasattr(model, attr), f"Model is missing required attribute: {attr}"
        
        print(f"Adding constraint: {self.constraint_description[index]}")

        # Define dynamic bounds if applicable
        d_highest = max(center)
        ub_power_balance = max(10**-2, 10**-2 * d_highest)
        lb_power_balance = -ub_power_balance

        if index < self.n_gens:
            constraint_name = f'generator{index + 1}_ub_constraint'
            setattr(model, constraint_name, pyo.Constraint(expr=(self.gen_limits[index][1] - model.output[index] + margin <= 0)))
        elif index < 2*self.n_gens:
            constraint_name = f'generator{index + 1}_lb_constraint'
            setattr(model, constraint_name, pyo.Constraint(expr=(model.output[index - self.n_gens] - self.gen_limits[index - self.n_gens][0] + margin <= 0)))

        elif index < 2*self.n_gens + self.n_buses:
            shifted_bus_index = index - 2*self.n_gens
            model = self.find_net_power(model, slack_bus_index=0, bus_index = shifted_bus_index)
            constraint_name = f'power_balance_ub_constraint_{shifted_bus_index + 1}'
            setattr(model, constraint_name, pyo.Constraint(expr=(ub_power_balance - model.net_power[shifted_bus_index] <= 0)))
        
        else:
            shifted_bus_index = index - 2*self.n_gens - self.n_buses
            model = self.find_net_power(model, slack_bus_index=0, bus_index = shifted_bus_index)
            constraint_name = f'power_balance_lb_constraint_{shifted_bus_index + 1}'
            setattr(model, constraint_name, pyo.Constraint(expr=(model.net_power[shifted_bus_index] - lb_power_balance <= 0)))
        
        return model


    def _constraint_description(self):
        constraint_description = {}
        for i in range(self.n_gens):
            constraint_description[i] = f"Upper bound for generator {i+1}"
            constraint_description[self.n_gens + i] = f"Lower bound for generator {i+1}"
        for i in range(self.n_buses):
            constraint_description[2*self.n_gens + i] = f"Upper bound on power balance for bus {i+1}"
            constraint_description[2*self.n_gens + self.n_buses + i] = f"Lower bound on power balance for bus {i+1}"
        return constraint_description

    def compute_admittance_matrix(self, slack_bus_index=0):

        # Initialize the bus admittance matrix with zeros
        Ybus = np.zeros((self.num_buses, self.num_buses), dtype=complex)
        
        # Populate the matrix
        for (i, j), reactance in self.reactance.items():
            admittance = 1 / complex(0, reactance)  # Inverse of the reactance (assuming purely inductive lines)
            
            # Update off-diagonal elements
            Ybus[i, j] -= admittance
            Ybus[j, i] -= admittance  # Symmetry
            
            # Update diagonal elements
            Ybus[i, i] += admittance
            Ybus[j, j] += admittance
        
        self.B = Ybus.imag
        # Extract B from the class (assuming it's stored as a numpy array)
        B = np.array(self.B, dtype=float)

        # Modify B to account for the reference (slack) bus
        B[:, slack_bus_index] = 0
        B[slack_bus_index, :] = 0
        B[slack_bus_index, slack_bus_index] = 1

        # Invert the modified B matrix
        self.B_inv = np.linalg.inv(B)
        return Ybus


    def find_net_power(self, model, slack_bus_index=0, bus_index = 0):
        # Extract B from the class (assuming it's stored as a numpy array)
        B = np.array(self.B, dtype=float)

        # Modify B to account for the reference (slack) bus
        B[:, slack_bus_index] = 0
        B[slack_bus_index, :] = 0
        B[slack_bus_index, slack_bus_index] = 1

        # Invert the modified B matrix
        B_inv = np.linalg.inv(B)

        # Define the extended variables of size 9
        P_g = Var(range(self.n_buses), initialize=0)
        P_d = Var(range(self.n_buses), initialize=0)

        # Define constraints to extend model.input and model.output to P_d and P_g
        model.P_g_constraint = ConstraintList()
        for i, bus_index in enumerate(self.gen_buses):
            model.P_g_constraint.add(P_g[bus_index] == model.output[i])
        for i in range(self.n_buses):
            if i not in self.gen_buses:
                model.P_g_constraint.add(P_g[i] == 0)

        model.P_d_constraint = ConstraintList()
        for i, bus_index in enumerate(self.demand_buses):
            model.P_d_constraint.add(P_d[bus_index] == model.input[i])
        for i in range(self.n_buses):
            if i not in self.demand_buses:
                model.P_d_constraint.add(P_d[i] == 0)

        # Calculate P_net
        P_net = Var(range(self.n_buses), initialize=0)
        model.P_net_constraint = ConstraintList()
        for i in range(self.n_buses):
            model.P_net_constraint.add(P_net[i] == P_g[i] - P_d[i])

        # Define theta as a Pyomo variable
        theta = Var(range(self.n_buses), initialize=0)
        model.theta = theta

        # Set the power injection at slack bus to 0 in the modified system (since theta_slack = 0)
        model.P_net_constraint.add(P_net[slack_bus_index] == 0)

        # Define the matrix-vector multiplication constraint using B_inv
        model.theta_constraint = ConstraintList()
        for i in range(9):
            model.theta_constraint.add(theta[i] == sum(B_inv[i, j] * P_net[j] for j in range(9)))

        # Define net_power as a Pyomo variable
        model.net_power = Var(range(self.n_buses), initialize=0)

        # Define power flow constraints based on reactances and voltage angles
        model.power_flow_constraint = ConstraintList()
        for i in range(self.n_buses):
            # Initial net power injection at each bus is zero
            net_power_expr = P_g[i] - P_d[i]
            for (bus_i, bus_j), reactance in self.X.items():
                if i == bus_i:
                    net_power_expr += (theta[bus_i] - theta[bus_j]) / reactance
                elif i == bus_j:
                    net_power_expr -= (theta[bus_i] - theta[bus_j]) / reactance
            model.power_flow_constraint.add(model.net_power[i] == net_power_expr)

        return model
    
    def display_results(self, inputs, outputs, center, norm_type):
        """
        Displays the results of the optimization process, allowing for custom or default input and output values,
        including the calculated norm of the inputs relative to the center.

        Parameters:
        - inputs (list or None): Custom list of input values to display. If None, uses model's inputs.
        - outputs (list or None): Custom list of output values to display. If None, uses model's outputs.

        This method prints:
        - The type of norm used in the optimization.
        - The values of the input and output variables, including how they compare to predefined bounds.
        - The norm of the input differences from the center.
        - The power balance and how it relates to generation and load.
        - The value of the primary property being enforced by constraints.
        - The objective function value.
        """
        # Determine the norm type used for easier reference in the output
        norm_type_expression = "infinity norm ball" if norm_type == 'infinity' else \
                    "L1 norm ball" if norm_type == 'l1' else "L2 norm ball"
        print(f"\nUsing {norm_type}")


        # Display input variable values and their respective centers
        for i, input_value in enumerate(inputs):
            print(f"Input {i + 1}: {input_value:.2f} (Center: {center[i]})")

        # Calculate and display the norm of the input differences from the center
        norm_value = self.norm_calculator(inputs, center, norm_type)
        print(f"Norm of input differences from center ({norm_type_expression}): {norm_value:.2f}")

        # Define bounds for generators
        max_gen = [self.gen_limits[i][1] for i in range(len(self.gen_limits))]
        min_gen = [self.gen_limits[i][0] for i in range(len(self.gen_limits))]

        # Display output variable values and their bounds
        for i, output_value in enumerate(outputs):
            print(f"Output {i + 1}: {output_value:.2f} (Bounds: {min_gen[i]} - {max_gen[i]})")

        # Calculate and display the power balance difference if using model's data
        difference = sum(outputs) - sum(inputs)
        balance_msg = "exceeds the load by" if difference > 0 else "is exceeded by the load by"
        print(f"\nOverall power mismatch (difference between generation and load): {difference:.2f} kW")
        print(f"The generation {balance_msg} {abs(difference):.2f} kW")

        print("\n")



    def norm_calculator(self, input_vector, center, norm_type):
        """
        Calculates the norm of the difference between a given input vector and the center.

        Parameters:
        - input_vector (list or np.array): The input vector for which the norm needs to be calculated.
        - center (list or np.array, optional): The center vector to calculate the norm against. Defaults to self.center.
        - norm_type (str, optional): The type of norm to calculate ('l1', 'l2', or 'infinity'). 

        Returns:
        - float: The calculated norm.
        """
        # Default to class attributes if not specified
        # Calculate the difference vector
        diff_vector = np.array(input_vector) - np.array(center)
        # Calculate the specified norm
        if norm_type == 'l1':
            norm_value = np.linalg.norm(diff_vector, ord=1)
        elif norm_type == 'l2':
            norm_value = np.linalg.norm(diff_vector, ord=2)
        elif norm_type == 'infinity':
            norm_value = np.linalg.norm(diff_vector, ord=np.inf)
        else:
            raise ValueError("Invalid norm type specified. Choose 'l1', 'l2', or 'infinity'.")

        return norm_value
    
    def render(self, inputs, outputs, center, norm_type, norm_value):
        assert len(center) == 3, "Plotting is only supported for 3-dimensional input vectors."
        radius_dict = {norm_type: norm_value}
        plot_3d_norm_balls(center, norm_types=(norm_type,), radius_dict=radius_dict, same_figure=True)

