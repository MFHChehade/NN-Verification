# import os
# from generate_DC_OPF_data import generate_DC_OPF_data
# from train_neural_network import train_neural_network
# from export_model_to_onnx import export_model_to_onnx
# from DC_OPF_constraints import add_DC_OPF_constraint
# from NN_verifier import NN_verifier
# from NeuralNet import NeuralNet
# import torch

# # File paths
# data_file = 'DC-OPF/results.csv'
# model_file = 'DC-OPF/model.pth'
# onnx_file = 'DC-OPF/model.onnx'

# # Generate DC OPF data only if it doesn't already exist
# if not os.path.exists(data_file):
#     df_results = generate_DC_OPF_data(1000)
#     df_results.to_csv(data_file, index=False)
#     print(df_results.head())
# else:
#     print(f"Data file {data_file} already exists, skipping data generation.")

# model = NeuralNet()

# # Train neural network only if the model file doesn't already exist
# if not os.path.exists(model_file):
#     model = train_neural_network(data_file, model_file)
# else:
#     print(f"Model file {model_file} already exists, skipping training.")
#     model.load_state_dict(torch.load(model_file))

# # Export model to ONNX format
# file_path = export_model_to_onnx(model)

# # # Solve DC OPF
# center = [50, 40, 75]
# output_limits = [(70, 95), (70, 95)]
# norm_type = 'infinity'
# NN_verifier_instance = NN_verifier(onnx_file = file_path, center = center, output_limits = output_limits, nn_model=model, constraint_function = add_DC_OPF_constraint, norm_type = norm_type, epsilon_infinity=50, margin=1e-3)
# NN_verifier_instance.maximum_feasible_ball()

# norm_type = 'l1'
# NN_verifier_instance = NN_verifier(onnx_file = file_path, center = center, output_limits = output_limits, nn_model=model, constraint_function = add_DC_OPF_constraint, norm_type = norm_type, epsilon_infinity=50, margin=1e-3)
# NN_verifier_instance.maximum_feasible_ball()

# norm_type = 'l2'
# NN_verifier_instance = NN_verifier(onnx_file = file_path, center = center, output_limits = output_limits, nn_model=model, constraint_function = add_DC_OPF_constraint, norm_type = norm_type, epsilon_infinity=50, margin=1e-3)
# NN_verifier_instance.maximum_feasible_ball()

import os
from DC_OPF import DC_OPF
from train_neural_network import train_neural_network
from export_model_to_onnx import export_model_to_onnx
from NeuralNet import NeuralNet
import torch
import numpy as np
from NN_verifier import NN_verifier
from plotting_norms import plot_3d_norm_balls

# File paths
data_file = 'DC-OPF/DC-OPF_data.csv'
model_file = 'DC-OPF/model.pth'
system_data_file = 'system_data'

# create DC_OPF instance
dc_opf = DC_OPF(system_data_file)

# Generate DC OPF data only if it doesn't already exist
if not os.path.exists(data_file):
    dc_opf.generate_data(1000)
else:
    print(f"Data file {data_file} already exists, skipping data generation.")

X,y = dc_opf.input_output(data_file)

model = NeuralNet()

# Train neural network only if the model file doesn't already exist
if not os.path.exists(model_file):
    model = train_neural_network(X, y, model_file)
else:
    print(f"Model file {model_file} already exists, skipping training.")
    model.load_state_dict(torch.load(model_file))


# Export model to ONNX format
file_path = export_model_to_onnx(model)

# create the center
expanded_center = dc_opf.demand
center = np.zeros(dc_opf.input_size)
for i, demand_bus in enumerate(dc_opf.demand_buses):
    center[i] = expanded_center[demand_bus]

# find the maximum feasible ball
norm_types = ('infinity', 'l1', 'l2')
radius_dict = {}

for norm_type in norm_types:
    NN_verifier_instance = NN_verifier(onnx_file = file_path, problem_class=dc_opf, nn_model=model, norm_type = norm_type, epsilon_infinity=100, margin=1e-3, center = center)
    radius,_ = NN_verifier_instance.maximum_feasible_ball()
    radius_dict[norm_type] = radius

plot_3d_norm_balls(center, norm_types=('infinity', 'l1','l2'), radius_dict=radius_dict, same_figure=True)