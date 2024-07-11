import pyomo.environ as pyo

def add_DC_OPF_constraint(model, index, margin, center, output_limits = None, input_neurons=3, output_neurons=2):
    """
    Adds a specified constraint to a Pyomo model based on the index provided.

    Parameters:
        model (pyomo.core.base.PyomoModel.ConcreteModel): The Pyomo model to which the constraint will be added.
        index (int): Index specifying which constraint to add (0-5).
        margin (float): Margin value to use in bound constraints.
        center (list): List of center values used to calculate dynamic upper and lower bounds.
        max_gen1 (float): Upper bound for generator 1.
        min_gen1 (float): Lower bound for generator 1.
        max_gen2 (float): Upper bound for generator 2.
        min_gen2 (float): Lower bound for generator 2.
        input_neurons (int): Number of input neurons, default is 3.
        output_neurons (int): Number of output neurons, default is 2.

    Raises:
        AssertionError: If the model does not have the required attributes.
        ValueError: If the index is out of range or if the model has missing variables.
    """

    max_gen1 = output_limits[0][1]
    min_gen1 = output_limits[0][0]
    max_gen2 = output_limits[1][1]
    min_gen2 = output_limits[1][0]

    # Validate the index and model requirements
    assert 0 <= index <= 5, "Index must be between 0 and 5."
    required_attrs = ['input', 'output']
    for attr in required_attrs:
        assert hasattr(model, attr), f"Model is missing required attribute: {attr}"
    
    constraint_descriptions = [
        "Upper bound on power balance",
        "Lower bound on power balance",
        "Upper bound for generator 1",
        "Lower bound for generator 1",
        "Upper bound for generator 2",
        "Lower bound for generator 2"
    ]
    
    print(f"Adding constraint: {constraint_descriptions[index]}")

    # Define dynamic bounds if applicable
    d_highest = max(center)
    ub_power_balance = max(10**-2, 10**-2 * d_highest)
    lb_power_balance = -ub_power_balance

    # Add the specific constraint to the model
    if index in [0, 1]:  # Power balance constraints
        balance = sum(model.output[i] for i in range(output_neurons)) - sum(model.input[i] for i in range(input_neurons))
        if index == 0:
            model.power_balance_ub_constraint = pyo.Constraint(expr=(ub_power_balance - balance <= 0))
        elif index == 1:
            model.power_balance_lb_constraint = pyo.Constraint(expr=(balance - lb_power_balance <= 0))

    elif index in [2, 3, 4, 5]:  # Generator constraints
        gen_index = (index - 2) // 2
        upper_bound = max_gen1 if gen_index == 0 else max_gen2
        lower_bound = min_gen1 if gen_index == 0 else min_gen2
        if index % 2 == 0:  # Upper bound
            constraint_name = f'generator{gen_index + 1}_ub_constraint'
            setattr(model, constraint_name, pyo.Constraint(expr=(upper_bound - model.output[gen_index] + margin <= 0)))
        else:  # Lower bound
            constraint_name = f'generator{gen_index + 1}_lb_constraint'
            setattr(model, constraint_name, pyo.Constraint(expr=(model.output[gen_index] - lower_bound + margin <= 0)))
    
    return model

