import matplotlib.pyplot as plt

# Read data from the txt file
train_losses = []
val_losses = []
epochs = []

# Path to your text file (replace with actual file path)
file_path = 'loss_data.txt'

with open(file_path, 'r') as file:
    for line in file:
        # Parse each line to extract epoch, train loss, and validation loss
        parts = line.strip().split(', ')
        epoch = int(parts[0].split()[1])  # Get the epoch number
        train_loss = float(parts[0].split(':')[1].strip())  # Get the train loss
        val_loss = float(parts[1].split(':')[1].strip())  # Get the validation loss

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

# Plotting the train vs validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='b', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', color='r', marker='o')
plt.title('Train vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('train_vs_val_loss.png', dpi=300)

# Optionally, close the plot after saving
plt.close()
