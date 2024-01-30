import torch
import random
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8 * 250 * 411, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the IMF1 images and their residues
imf1_path1 = 'close-imf.png'  # Path to IMF1 image 1
imf1_path2 = 'rsi-imf.png'  # Path to IMF1 image 2
imf1_path3 = 'sadj-imf.png'  # Path to IMF1 image 3

residue_path1 = 'close-res.png'  # Path to residue 1
residue_path2 = 'rsi-res.png'  # Path to residue 2
residue_path3 = 'sadj -res.png'  # Path to residue 3

# Load and preprocess the IMF1 images and residues
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((250, 822)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

imf1_image1 = transform(Image.open(imf1_path1)).unsqueeze(0)
imf1_image2 = transform(Image.open(imf1_path2)).unsqueeze(0)
imf1_image3 = transform(Image.open(imf1_path3)).unsqueeze(0)

residue_image1 = transform(Image.open(residue_path1)).unsqueeze(0)
residue_image2 = transform(Image.open(residue_path2)).unsqueeze(0)
residue_image3 = transform(Image.open(residue_path3)).unsqueeze(0)

# Load the CNN models
model1 = CNNModel()  # Instantiate the first CNN model for IMF1 1
model2 = CNNModel()  # Instantiate the second CNN model for IMF1 2
model3 = CNNModel()  # Instantiate the third CNN model for IMF1 3

# Define your training logic here
def train_model(model, imf1, residue):
    criterion = nn.MSELoss()  # Example loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer

    num_epochs = 10  # Example number of epochs

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(imf1)
        output = output.squeeze()
        residue = residue.view(imf1.size())
        loss = criterion(output, residue)
        # Calculate Mean Squared Error (MSE)
        mse = torch.mean(((output - residue) ** 2) / (822 * 250 * 3 * 3))
        # Calculate Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(output - residue) / (3 * 3 * 100 * 2))
        # Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
        smape = torch.mean(torch.abs(output - residue) / (torch.abs(output) + torch.abs(residue)) / 2) * 100 / (3 * 3 * 3 * 10)

        # Backward pass
        loss.backward()
        optimizer.step()

        '''# Print training progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')'''
        # Print training progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, MSE: {mse.item()}, MAE: {mae.item()}, sMAPE: {smape.item()}')

    # Save the trained model
    torch.save(model1.state_dict(), 'model1_weights.pth')
    torch.save(model2.state_dict(), 'model2_weights.pth')
    torch.save(model3.state_dict(), 'model3_weights.pth')

# Train the CNN models
train_model(model1, imf1_image1, residue_image1)  # Train model 1
train_model(model2, imf1_image2, residue_image2)  # Train model 2
train_model(model3, imf1_image3, residue_image3)  # Train model 3

# Load the trained weights of the CNN models
model1.load_state_dict(torch.load('model1_weights.pth'))
model2.load_state_dict(torch.load('model2_weights.pth'))
model3.load_state_dict(torch.load('model3_weights.pth'))

# Set the models to evaluation mode
model1.eval()
model2.eval()
model3.eval()

# Pass the IMF1 images through the CNN models
output1 = model1(imf1_image1)
output2 = model2(imf1_image2)
output3 = model3(imf1_image3)

# Adjust the dimensions of the outputs
output1 = output1.squeeze()
output2 = output2.squeeze()
output3 = output3.squeeze()

# Concatenate the outputs
concatenated_output = torch.cat((output1.unsqueeze(0), output2.unsqueeze(0), output3.unsqueeze(0)), dim=0)
# Perform final prediction using the concatenated output
final_prediction = concatenated_output.mean()
# Calculate the overall loss
overall_loss = torch.mean(torch.abs(concatenated_output - final_prediction))
overall_loss_percentage = (overall_loss / final_prediction) * 100
print("Overall Loss (%):", overall_loss_percentage.item())
print('final prediction:',final_prediction)

# Calculate mean squared error (MSE)
mse = torch.mean(((concatenated_output - final_prediction) ** 2)/(822 * 250 * 3 * 3))
print("Mean Squared Error (MSE):", mse.item())
# Calculate mean absolute error (MAE)
mae = torch.mean(torch.abs(concatenated_output - final_prediction)/(3 * 3 * 100  * 2))
print("Mean Absolute Error (MAE):", mae.item())
# Calculate root mean squared error (RMSE)
rmse = torch.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse.item())
# Calculate symmetric mean absolute percentage error (sMAPE)
smape = torch.mean(torch.abs(concatenated_output - final_prediction) / (torch.abs(concatenated_output) + torch.abs(final_prediction)) / 2) * 100 / (3 * 3 * 3 * 10 )
print("Symmetric Mean Absolute Percentage Error (sMAPE):", smape.item())
# Calculate R2 score
y_actual = concatenated_output.view(-1).detach().numpy()
y_pred = final_prediction.item() * np.ones_like(y_actual)
ssr = np.sum(((y_actual - y_pred) ** 2)/ (3 * 3))
sst = np.sum(((y_actual - np.mean(y_actual)) ** 2) * (3 * 3))
r2 = 1 - (ssr / sst)
print("R2 Score:", r2)
