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

    def __init__(self, onnx_file, problem_class, center, nn_model=None, norm_type = 'infinity', epsilon_infinity=10, margin=1e-3):
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
        self.problem_class = problem_class
        self.nn_model = nn_model

        self.center = center
        self.output_limits = problem_class.gen_limits
        self.epsilon_infinity = epsilon_infinity
        self.margin = margin

        self.input_size = problem_class.input_size
        self.output_size = problem_class.output_size

        self.input_bounds = self._calculate_bounds()
        self.load_onnx()

        self.model = None    
        self.norm_type = norm_type  # default norm type

    def _calculate_bounds(self):
        """Calculates the input bounds based on the center and epsilon_infinity."""
        lb = self.center - self.epsilon_infinity
        ub = self.center + self.epsilon_infinity
        print(f"Input bounds: {lb} - {ub}")
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
        self.model = self.problem_class.add_constraint(self.model, index = index, margin = self.margin, center = self.center)

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
        
        if self.norm_type == 'infinity' or self.norm_type == 'l1':
            norm_value = self.model.obj()
        else:
            norm_value = np.sqrt(self.model.obj())

        return inputs, outputs, norm_value


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
            input, output, _ = self.solve_opt_problem()
            self.display_results()
            # Fetch the objective function value which quantifies the size of the feasible region
            if self.norm_type == 'infinity' or self.norm_type == 'l1':
                region_size = self.model.obj() if self.model.obj.expr() is not None else float('inf')
            else:
                region_size = np.sqrt(self.model.obj()) if self.model.obj.expr() is not None else float('inf')

            # Check if the current configuration results in a smaller feasible region
            if region_size < smallest_region:
                smallest_region = region_size
                best_config = index
                best_input = input
                best_output = output


        print(f"Smallest feasible region size: {smallest_region}")
        print(f"Best configuration: {self.problem_class.constraint_description[best_config]}")

        self.display_results(best_input, best_output)
        self.render(best_input, best_output, self.center, self.norm_type, smallest_region)
        return smallest_region, best_config

    def display_results(self, inputs = None, outputs = None, center = None, norm_type = None):
        """
        Displays the results of the optimization problem, including input and output values.

        Args:
            inputs (list): Values of the input variables.
            outputs (list): Values of the output variables.
            center (list): Central point in the input space.
        """
        if inputs is None:
            inputs = [self.model.input[i].value for i in self.model.input]
        if outputs is None:
            outputs = [self.model.output[i].value for i in self.model.output]
        if center is None:
            center = self.center
        if norm_type is None:
            norm_type = self.norm_type

        self.problem_class.display_results(inputs, outputs, center, norm_type)
    
    def render(self, inputs = None, outputs = None, center = None, norm_type = None, norm_value = None):
        if inputs is None:
            inputs = [self.model.input[i].value for i in self.model.input]
        if outputs is None:
            outputs = [self.model.output[i].value for i in self.model.output]
        if center is None:
            center = self.center
        if norm_type is None:
            norm_type = self.norm_type
        if norm_value is None:
            if norm_type == 'infinity' or norm_type == 'l1':
                norm_value = self.model.obj()
            else:
                norm_value = np.sqrt(self.model.obj())

        self.problem_class.render(inputs, outputs, center, norm_type, norm_value)

