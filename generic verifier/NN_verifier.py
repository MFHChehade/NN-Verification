import numpy as np
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
# from norms.plotting_norms import plot_3d_norm_balls


import numpy as np

class NN_verifier:
    """
    A class for optimizing neural network constraints within a power system context,
    particularly focused on DC optimal power flow problems.
    """

    def __init__(self, onnx_file, center, output_limits = None, nn_model=None, constraint_function = None, norm_type = 'infinity', epsilon_infinity=10, margin=1e-3):
        """
        Attributes:
        onnx_file (str): Path to the ONNX model file.

        nn_model: The neural network model object, default is None.
        constraint_function: Function to add the constraints to the Pyomo model.

        epsilon_infinity (float): Radius for the infinity norm constraints around the center.
        center (np.array): Central point in the input space around which bounds are calculated.
        output_limits (list): List of tuples specifying the output limits for the generators.
        margin (float): Small margin used in constraints to ensure non-strict inequalities.

        input_bounds (dict): Calculated bounds for the input variables.
        network_definition: Loaded ONNX network definition.
        formulation: OMLT formulation created from the ONNX model.

        model: The Pyomo model object.
        norm_type (str): Type of norm used ('infinity', 'l1', or 'l2') for the optimization problem.
        """

        self.onnx_file = onnx_file
        self.nn_model = nn_model
        self.constraint_function = constraint_function

        self.center = np.array(center)
        self.output_limits = output_limits
        self.epsilon_infinity = epsilon_infinity
        self.margin = margin

        self.input_size = len(center)
        self.output_size = len(output_limits)

        self.input_bounds = self._calculate_bounds()
        self.load_onnx()

        self.model = None    
        self.norm_type = norm_type  # default norm type

    def _calculate_bounds(self):
        """Calculates the input bounds based on the center and epsilon_infinity."""
        lb = self.center - self.epsilon_infinity
        ub = self.center + self.epsilon_infinity
        return {i: (float(lb[i]), float(ub[i])) for i in range(len(self.center))}

    def load_onnx(self):
        """
        Loads the ONNX model and sets up the neural network formulation.

        This method writes the ONNX model with the specified input bounds, 
        loads the network definition, and creates the neural network formulation.
        """
        write_onnx_model_with_bounds(self.onnx_file, None, self.input_bounds)
        self.network_definition = load_onnx_neural_network_with_bounds(self.onnx_file)
        self.formulation = FullSpaceNNFormulation(self.network_definition)


    def declare_variables(self):
        """
        Declares variables in the Pyomo model based on the neural network definition.

        This method sets up a concrete model with input and output variables, auxiliary variables based on the norm type,
        and constraints to link these variables with the neural network inputs and outputs.
        """
        m = pyo.ConcreteModel()
        m.nn = OmltBlock()
        m.nn.build_formulation(self.formulation)

        # Create input and output variables
        m.input = pyo.Var(range(self.input_size), domain=pyo.NonNegativeReals)
        m.output = pyo.Var(range(self.output_size), domain=pyo.Reals)

        # Create auxiliary variable depending on the norm type
        if self.norm_type == 'infinity':
            m.auxiliary = pyo.Var(domain=pyo.NonNegativeReals)
        elif self.norm_type == 'l1':
            m.auxiliary = pyo.Var(range(self.input_size), domain=pyo.NonNegativeReals)

        # Constraints to connect neural network inputs and outputs
        m.connect_inputs = pyo.ConstraintList()
        m.connect_outputs = pyo.ConstraintList()
        for i in range(self.input_size):
            m.connect_inputs.add(m.input[i] == m.nn.inputs[i])
        for j in range(self.output_size):
            m.connect_outputs.add(m.output[j] == m.nn.outputs[j])

        self.model = m



    def create_opt_problem(self, index = 0):
        self.model = self.constraint_function(self.model, index = index, margin = self.margin, center = self.center, output_limits = self.output_limits, input_neurons = self.input_size, output_neurons = self.output_size)

        # Define norm constraints and objective function
        self.define_norm_constraints(self.input_size, self.model)

    def define_norm_constraints(self, input_neurons, m):
        """
        Define norm constraints based on the selected norm type and set up the objective function.

        Args:
            input_neurons (int): Number of input neurons.
            m (pyo.ConcreteModel): The Pyomo model.
        """
        if self.norm_type == 'l1':
            m.l1_constraints = pyo.ConstraintList()
            for i in range(input_neurons):
                m.l1_constraints.add(m.input[i] - self.center[i] <= m.auxiliary[i])
                m.l1_constraints.add(-m.auxiliary[i] <= m.input[i] - self.center[i])
            m.obj = pyo.Objective(expr=sum(m.auxiliary[i] for i in range(input_neurons)), sense=pyo.minimize)
        elif self.norm_type == 'l2':
            m.obj = pyo.Objective(expr=sum((m.input[i] - self.center[i])**2 for i in range(input_neurons)), sense=pyo.minimize)
        else:  # 'infinity' norm by default
            m.l1_constraints = pyo.ConstraintList()
            for i in range(input_neurons):
                m.l1_constraints.add(m.input[i] - self.center[i] <= m.auxiliary)
                m.l1_constraints.add(-m.auxiliary <= m.input[i] - self.center[i])
            m.obj = pyo.Objective(expr=m.auxiliary, sense=pyo.minimize)


    def solve_opt_problem(self, solver = None):
        """
        Solves the optimization problem using the CBC solver and returns the model's inputs and outputs.

        Prints the outcome of the optimization process. If the solver reaches an optimal solution,
        it reports success; otherwise, it indicates failure.

        Returns:
            list: Values of the input variables after solving the optimization problem.
            list: Values of the output variables after solving the optimization problem.
        """

        # assert that in l2 case, solver is bonmin
        if solver is None:
            if self.norm_type == 'l2':
                solver = 'bonmin'
            else:
                solver = 'cbc'

        if self.norm_type == 'l2':
            assert solver == 'bonmin', "Solver must be 'bonmin' for l2 norm."

        solver_status = pyo.SolverFactory(solver).solve(self.model, tee=False)  # tee=False suppresses solver output

        # Initialize lists to store input and output values
        inputs = []
        outputs = []

        # Check if the solver found an optimal solution
        if solver_status.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("Problem solved successfully.")
            # Retrieve and store input values
            inputs = [self.model.input[i].value for i in self.model.input]
            # Retrieve and store output values
            outputs = [self.model.output[i].value for i in self.model.output]
        else:
            print("The solver failed to solve the problem.")

        return inputs, outputs


    def display_results(self, inputs=None, outputs=None):
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
        norm_type = "infinity norm ball" if self.norm_type == 'infinity' else \
                    "L1 norm ball" if self.norm_type == 'l1' else "L2 norm ball"
        print(f"\nUsing {norm_type}")

        # Use model's inputs and outputs if none are provided
        if inputs is None:
            inputs = [self.model.input[i].value for i in range(len(self.model.input))]
        if outputs is None:
            outputs = [self.model.output[i].value for i in range(len(self.model.output))]

        # Display input variable values and their respective centers
        for i, input_value in enumerate(inputs):
            print(f"Input {i + 1}: {input_value:.2f} (Center: {self.center[i]})")

        # Calculate and display the norm of the input differences from the center
        norm_value = self.norm_calculator(inputs)
        print(f"Norm of input differences from center ({norm_type}): {norm_value:.2f}")

        # Define bounds for generators
        max_gen = [self.output_limits[i][1] for i in range(len(self.output_limits))]
        min_gen = [self.output_limits[i][0] for i in range(len(self.output_limits))]

        # Display output variable values and their bounds
        for i, output_value in enumerate(outputs):
            print(f"Output {i + 1}: {output_value:.2f} (Bounds: {min_gen[i]} - {max_gen[i]})")

        # Calculate and display the power balance difference if using model's data
        difference = sum(outputs) - sum(inputs)
        balance_msg = "exceeds the load by" if difference > 0 else "is exceeded by the load by"
        print(f"\nPower balance (difference between generation and load): {difference:.2f} kW")
        print(f"The generation {balance_msg} {abs(difference):.2f} kW")

        print("\n")



    def norm_calculator(self, input_vector, center=None, norm_type=None):
        """
        Calculates the norm of the difference between a given input vector and the center.

        Parameters:
        - input_vector (list or np.array): The input vector for which the norm needs to be calculated.
        - center (list or np.array, optional): The center vector to calculate the norm against. Defaults to self.center.
        - norm_type (str, optional): The type of norm to calculate ('l1', 'l2', or 'infinity'). Defaults to self.norm_type.

        Returns:
        - float: The calculated norm.
        """
        # Default to class attributes if not specified
        if center is None:
            center = self.center
        if norm_type is None:
            norm_type = self.norm_type

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


    def maximum_feasible_ball(self):
        """
        Determines the configuration that results in the smallest feasible region by varying constraint settings.

        Iterates over a predefined set of configurations, each differing in the constraints applied,
        to find which configuration yields the smallest objective value, indicating the tightest feasible region.

        Returns:
            float: The size of the smallest feasible region found.
            dict: The configuration that resulted in the smallest feasible region.
        """
        smallest_region = float('inf')
        best_config = None
        best_input = None
        best_output = None

        for index in range(6):
            # Create and solve the optimization problem with the current configuration
            self.declare_variables()
            self.create_opt_problem(index)
            input, output = self.solve_opt_problem()
            self.display_results()
            # Fetch the objective function value which quantifies the size of the feasible region
            region_size = self.model.obj() if self.model.obj.expr() is not None else float('inf')

            # Check if the current configuration results in a smaller feasible region
            if region_size < smallest_region:
                smallest_region = region_size
                best_config = index
                best_input = input
                best_output = output


        print(f"Smallest feasible region size: {smallest_region}")
        print(f"Best configuration: {best_config}")

        self.display_results(best_input, best_output)
        return smallest_region, best_config

