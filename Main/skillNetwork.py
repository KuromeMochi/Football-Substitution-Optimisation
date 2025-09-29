import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from unidecode import unidecode
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from preprocessing import *

BATCH_SIZE = 64

num_epochs = 50

class PlayerRatingModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(PlayerRatingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1) # single output: player rating

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # no softmax, regression
        return x  # output is a continuous rating
    
def train_model(X, y, epochs=70, lr=0.001, batch_size=64):
    input_size = X.shape[1]
    print(f"Size: {input_size}")
    model = PlayerRatingModel(input_size)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # split into train & validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

    # convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimiser.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        validation_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, X_val_tensor, y_val_tensor, training_losses, validation_losses

def predict(model, X_sample):
    model.eval()
    with torch.no_grad():
        prediction = model(X_sample)
        return torch.round(prediction * 10) / 10  # 1dp

def preprocess(player_data):
    player_data["Player"] = player_data["Player"].apply(lambda x: unidecode(x))
    player_data = player_data.fillna(0)

    player_data = player_data.drop(columns=["Rk", "Player", "Nation", "Squad", "Comp", "Born"])

    player_data["Estimated_Rating"] = 0

    # Create a rating based on weighted stats
    for index, row in player_data.iterrows():
        if row["Pos"] == "FW":
            player_data.at[index, "Estimated_Rating"] = (
                70 * row["Goals"] + 
                50 * row["Assists"] +
                50 * row["Shots"] +
                60 * row["SoT"] +
                40 * row["PPA"] +
                10 * row["ShoPK"] +
                10 * row["PasTotCmp"]
            )
        elif row["Pos"] == "MF":
            player_data.at[index, "Estimated_Rating"] = (
                60 * row["Goals"] + 
                80 * row["Assists"] +
                40 * row["Shots"] +
                50 * row["PasTotCmp"] +
                50 * row["PasTotAtt"] +
                40 * row["PPA"] +
                10 * row["PasTotCmp"]
            )
        elif row["Pos"] == "DF":
            player_data.at[index, "Estimated_Rating"] = (
                80 * row["Goals"] + 
                40 * row["Assists"] +
                40 * row["Recov"] +
                40 * row["AerWon"] +
                40 * row["PasLonCmp"] +
                40 * row["PPA"] +
                10 * row["PasTotCmp"]
            )
        else:
            player_data.at[index, "Estimated_Rating"] = (
                50 * row["Goals"] + 
                40 * row["Assists"] +
                40 * row["Shots"] +
                40 * row["PPA"] +
                10 * row["PasTotCmp"]
            )

    print(player_data["Estimated_Rating"])

    # normalize ratings to 0-100 scale
    scaler = MinMaxScaler(feature_range=(70, 100))
    player_data["Estimated_Rating"] = scaler.fit_transform(player_data[["Estimated_Rating"]])

    player_data = player_data.drop(columns=["Pos"], errors="ignore")
    player_data = player_data.astype(np.float32)

    X = player_data.drop(columns=["Estimated_Rating"]).values
    # Target = Estimated Rating
    y = player_data["Estimated_Rating"].values

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, mae, r2

player_data = pd.read_csv("Main/2223_football_player_stats.csv", delimiter=";", encoding="ISO-8859-1")
X_tensor, y_tensor = preprocess(player_data)
model, X_test, y_test, training_losses, validation_losses = train_model(X_tensor, y_tensor)
#print(X_test)

predicted = []
expected = []

# Test on a sample
#sample_idx = 3
for sample_idx in range(100):
    sample_input = X_test[sample_idx].unsqueeze(0)

    predicted_rating = predict(model, sample_input)

    # print(f"Predicted Rating: {predicted_rating.item():.1f}")
    # print(f"Actual Rating: {y_test[sample_idx].item():.1f}")

    predicted.append(predicted_rating.item())
    expected.append(y_test[sample_idx].item())


# mse, rmse, mae, r2 = evaluate_model(np.array(expected), np.array(predicted))

# # Print metrics
# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")

# # plot training loss
# plt.plot(range(70), training_losses, label="Training Loss")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.plot(range(len(validation_losses)), validation_losses, color="orange", label="Validation Loss")
# plt.xlabel('Epoch')
# plt.ylabel('Validation Loss')
# plt.title('Validation Loss Curve')
# plt.legend()
# plt.grid(True)
# plt.show()
