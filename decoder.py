# IMPORT ALL NECESSARY LIBRARIES FOR THE OBJECT
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the dataset
data = [
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]
]

# Convert the data to tensors
data = torch.tensor(data, dtype=torch.float32)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create the model
model = Autoencoder()

# Define the criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Encode and decode the data
encoded_data = model.encoder(data)
decoded_data = model.decoder(encoded_data)

# Convert the decoded data to binary values
decoded_data_binary = (decoded_data > 0.5).int()

# Print the original and reconstructed data
print("Original Data:")
print(data)
print("Reconstructed Data:")
print(decoded_data_binary)

# Visualize the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# Visualize a few examples of original and reconstructed data
def show_images(original, reconstructed):
    fig, axs = plt.subplots(nrows=2, ncols=len(original), figsize=(12, 4))
    for i in range(len(original)):
        axs[0, i].imshow(original[i].numpy().reshape(1, -1), cmap='binary')
        axs[0, i].set_title('Original')
        axs[0, i].axis('off')
        axs[1, i].imshow(reconstructed[i].detach().numpy().reshape(1, -1), cmap='binary')
        axs[1, i].set_title('Reconstructed')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.show()

show_images(data, decoded_data_binary)
