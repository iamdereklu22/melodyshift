import re

# Path to the text file
file_path = 'raw_data.txt'

# Define lists to store the values
epochs = []
train_losses = []
val_losses = []

# Regular expressions for parsing
epoch_re = r'Epoch (\d+)/\d+'
train_loss_re = r'Train Avg Loss: ([\d.]+)'
val_loss_re = r'Validation Loss: ([\d.]+)'

# Read the file and process it line by line
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        epoch_match = re.search(epoch_re, line)
        train_loss_match = re.search(train_loss_re, line)
        val_loss_match = re.search(val_loss_re, line)
        
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))
        if train_loss_match:
            train_losses.append(float(train_loss_match.group(1)))
        if val_loss_match:
            val_losses.append(float(val_loss_match.group(1)))

# Now we have the data, we can print or process it as needed
for epoch, train_loss, val_loss in zip(epochs, train_losses, val_losses):
    print(f"Epoch {epoch} - Train Loss: {train_loss}, Validation Loss: {val_loss}")
