import torch
import torch.nn as nn
import torch.onnx
import tempfile

def export_model_to_onnx(model, file_path=None, input_names=['input'], output_names=['output'], dtype=torch.float32):
    """
    Exports a trained PyTorch model to ONNX format. Automatically determines the input size if possible.

    Parameters:
        model (torch.nn.Module): The trained model to export.
        file_path (str, optional): The file path where the ONNX model will be saved.
                                   If not specified, a temporary file will be used.
        input_names (list of str): Names of the inputs for the ONNX model.
        output_names (list of str): Names of the outputs for the ONNX model.
        dtype (torch.dtype): Data type of the input tensor (default is torch.float32).

    Returns:
        str: The file path where the ONNX model is saved.
    """
    # Attempt to determine input size from the first layer
    first_layer = next(model.children())
    if isinstance(first_layer, nn.Conv2d):
        # Assuming format (batch_size, channels, height, width)
        input_shape = (1, first_layer.in_channels, 224, 224)  # You may need to adjust 224x224 depending on the model
    elif isinstance(first_layer, nn.Linear):
        # Assuming format (batch_size, features)
        input_shape = (1, first_layer.in_features)
    else:
        raise ValueError("Automatic input shape determination not supported for this layer type")

    # Generate a dummy input tensor based on the determined input shape
    dummy_input = torch.rand(input_shape, dtype=dtype)

    # Use a temporary file if no file path is provided
    if file_path is None:
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            file_path = tmp_file.name
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},  # Enable dynamic batch size for input
            'output': {0: 'batch_size'}  # Enable dynamic batch size for output
        }
    )
    
    print(f"Model successfully exported to {file_path}")
    return file_path
