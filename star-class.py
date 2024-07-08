import torch
import torch.nn as nn
import torch.nn.functional as f
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Create a Model class for your neural network


class Model(nn.Module):
    # Model will have 6 in features (Temperature, Luminosity, Radius, Absolute magnitude, Star color, Spectral Class)
    # Model will then use two hidden layers, one using 14 nodes, and another using 17 nodes
    # Finally the Model will have 6 choices of what type of star the data is showing (Brown Dwarf, Red Dwarf, White Dwarf, Main Sequence, Super Giants, Hyper Giants)
    # Using a linear transformation to the information given to the model
    def __init__(self, dataframe, in_features=6, h1=14, h2=17, out_features=6):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        # Set manual seed for Pytorch to use during training
        torch.manual_seed(41)

        # Set criterion for measuring the error by using the CrossEntropyLoss class given by PyTorch
        self.criterion = nn.CrossEntropyLoss()

        # Use Adam optimizer and set learning rate of model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Create X and y for test split
        # X is all the data except the star type
        # y is the star type itself, which we are training the model to classify
        X = dataframe.drop('Star type', axis=1)
        y = dataframe['Star type']

        # Converts X and y to numpy arrays
        X = X.values
        y = y.values

        # Set variable for training split using scikit
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.35, random_state=41)

        # Convert in variables to float tensors
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)

        # Convert classifications to long tensors
        self.y_train = torch.LongTensor(self.y_train)
        self.y_test = torch.LongTensor(self.y_test)

        self.epochs = 1500  # Number of times the model will run through the test set

    # Forward function which moves each piece of data through each hidden layer and to the out layer
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.out(x)

        return x

    def train(self):
        losses = []  # List of losses
        for i in range(self.epochs):
            # Move forward and get prediction
            y_pred = self.forward(self.X_train)

            # Measure the error in the epoch and keep track of losses
            loss = self.criterion(y_pred, self.y_train)
            losses.append(loss.detach().numpy())

            if i % 10 == 0:
                print(f'Epoch: {i} and loss: {loss}')

            # Back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):
        # Print statement to start
        print("Begin Test!\n")

        # Test the model
        with torch.no_grad():
            y_eval = self.forward(self.X_test)
            loss = self.criterion(y_eval, self.y_test)

        correct = 0
        with torch.no_grad():
            for i, data in enumerate(self.X_test):
                # Pushes test data through layers for testing
                y_val = self.forward(data)

                # Prints which star type our network thinks the star is and the actual star type of each star
                print(f"{i+1}.) {self.y_test[i]} \t {y_val.argmax().item()}")

                # Checks if our network is correct
                if y_val.argmax().item() == self.y_test[i]:
                    correct += 1

        print(f"Score: {correct}/{len(self.X_test)}")

    def graph_training_data(self, loss_arr):
        plt.plot(range(self.epochs, loss_arr))
        plt.ylabel("Loss/Error")
        plt.xlabel("Epoch")


if __name__ == "__main__":
    # Upload Stars classification dataset onto a pandas dataframe
    url = 'https://github.com/YBIFoundation/Dataset/raw/main/Stars.csv'
    star_df = pd.read_csv(url)

    # Remove star category column as star type gives us the same information
    star_df = star_df.drop('Star category', axis=1)

    # Instantiate dictionaries for converting string values to decimal values for the model's use
    color_dict = {}
    spec_dict = {}

    # Fill dictionary with each unique color found in the dataset
    for color in star_df['Star color']:
        if color not in color_dict.values():
            color_dict[len(color_dict)] = color

    # Replace each color with it's corresponding key, making it a decimal
    for key in color_dict:
        star_df['Star color'] = star_df['Star color'].replace(
            color_dict[key], key)

    # Repeat the same steps for the Spectral Class column
    for spec in star_df['Spectral Class']:
        if spec not in spec_dict.values():
            spec_dict[len(spec_dict)] = spec

    for key in spec_dict:
        star_df['Spectral Class'] = star_df['Spectral Class'].replace(
            spec_dict[key], key)

    # Create the Model object
    model = Model(dataframe=star_df)

    # Call the training function and graph the training data
    losses = model.train()

    # Call the testing function
    model.test()
