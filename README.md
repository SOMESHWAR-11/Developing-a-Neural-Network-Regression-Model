# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.
<img width="953" height="586" alt="image" src="https://github.com/user-attachments/assets/a00cb1ff-9538-4ea1-9f69-238fd16c5a4a" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: SOMESHWAR S

### Register Number: 212224040322

```python
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1,8)
    self.fc2 = nn.Linear(8,10)
    self.fc3 = nn.Linear(10,1)
    self.relu = nn.ReLU()
    self.history = {'loss':[]}

  def forward(self,x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train), y_train)
      loss.backward()
      optimizer.step()
      
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information
Include screenshot of the generated data

<img width="325" height="540" alt="image" src="https://github.com/user-attachments/assets/b8e9c857-958b-4bbc-94c1-01a4ce9e66af" />


### OUTPUT

### Training Loss Vs Iteration Plot
Include your plot here

<img width="799" height="754" alt="image" src="https://github.com/user-attachments/assets/b2c1d471-3767-46d4-98b8-726a2f1ae8ea" />


### New Sample Data Prediction
Include your sample input and output here

<img width="898" height="131" alt="image" src="https://github.com/user-attachments/assets/80a479a6-74d9-4607-aeb5-1220379b6559" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
