import os
from generate_DC_OPF_data import generate_DC_OPF_data
from train_neural_network import train_neural_network
from export_model_to_onnx import export_model_to_onnx
from DC_OPF_constraints import add_DC_OPF_constraint
from NN_verifier import NN_verifier
from NeuralNet import NeuralNet
import torch

# File paths
data_file = 'results.csv'
model_file = 'DC-OPF/model.pth'
onnx_file = 'DC-OPF/model.onnx'

# Generate DC OPF data only if it doesn't already exist
if not os.path.exists(data_file):
    df_results = generate_DC_OPF_data(100)
    df_results.to_csv(data_file, index=False)
    print(df_results.head())
else:
    print(f"Data file {data_file} already exists, skipping data generation.")

model = NeuralNet()

# Train neural network only if the model file doesn't already exist
if not os.path.exists(model_file):
    model = train_neural_network(data_file, model_file)
else:
    print(f"Model file {model_file} already exists, skipping training.")
    model.load_state_dict(torch.load(model_file))

# Export model to ONNX format
file_path = export_model_to_onnx(model)

# # Solve DC OPF
center = [50, 40, 75]
output_limits = [(70, 95), (70, 95)]
NN_verifier_instance = NN_verifier(onnx_file = file_path, center = center, output_limits = output_limits, nn_model=model, constraint_function = add_DC_OPF_constraint, norm_type = 'infinity', epsilon_infinity=50, margin=1e-3)
NN_verifier.maximum_feasible_ball()

input = torch.tensor([50.0, 40.0, 75.0])
output = model(input)
print(output)