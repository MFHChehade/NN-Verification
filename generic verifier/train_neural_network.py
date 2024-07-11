import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from NeuralNet import NeuralNet  # Ensure this module is correctly implemented and imported

def train_neural_network(data_path, model_path, model_class=NeuralNet, criterion=nn.MSELoss(), optimizer_class=optim.Adam,
                         num_epochs=100, batch_size=10, learning_rate=0.001):
    """
    Trains a neural network on data from a specified CSV file and saves the model.

    Parameters:
        data_path (str): Path to the CSV file containing the training data.
        model_path (str): Path to save the trained model.
        model_class (torch.nn.Module class): Class of the neural network model to be trained.
        criterion (torch.nn.modules.loss): Loss function to be used for training.
        optimizer_class (torch.optim.Optimizer): Optimizer class to be used for training.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        torch.nn.Module: Trained neural network model.
    """
    # Load the dataset
    df_results = pd.read_csv(data_path)
    print(df_results.columns)  # Debugging: Print columns to ensure correct data handling

    # Prepare input features and target outputs
    X = df_results[['Demand1', 'Demand2', 'Demand3']].values
    y = df_results[['Gen1', 'Gen2']].values

    # Convert arrays to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create a TensorDataset for better data handling
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader instances to handle data batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Neural network initialization
    model = model_class()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % 10 == 0:  # Print loss every 10 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model weights
    torch.save(model.state_dict(), model_path)

    return model

