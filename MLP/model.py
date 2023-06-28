import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, \
    f1_score

filename = 'data/western_china_train/all_data.csv'
train_data = pd.read_csv(filename, header=[0, 1], index_col=0, low_memory=False)
filename = 'data/western_china_train/train_label.csv'
train_label = pd.read_csv(filename, index_col=0, low_memory=False)

filename = 'data/western_test/all_test.csv'
test_data = pd.read_csv(filename, header=[0, 1], index_col=0, low_memory=False)
filename = 'data/western_test/test_label.csv'
test_label = pd.read_csv(filename, index_col=0, low_memory=False)

data = train_data
label = train_label

data = data.loc[:, [col for col in data.columns if col[1] in ["mut_important", "cnv_del", "cnv_amp"]]]
data.columns = data.columns.get_level_values(0) + "_" + data.columns.get_level_values(1)

ls = ('AR'
      , 'PTEN'
      , 'TP53'
      , 'MUC4'
      , 'RB1'
      , 'RBBP5'
      , 'CD55'
      , 'ARHGEF9'
      , 'ASB12'
      , 'CTSE'
      , 'FGF2'
      , 'RNF213'
      , 'MUC16'
      , 'UBE2W'
      , 'PKP1'
      , 'PPP1CA'
      , 'ARHGAP21'
      , 'COL1A2'
      )
data_filtered = data.filter(regex='^(' + '|'.join(ls) + ')_')
print(data_filtered.shape)

data_flat = data_filtered.values.reshape((data.shape[0], -1))
X_train, X_test, y_train, y_test = train_test_split(data_flat, label.values.ravel(), test_size=0.2, random_state=42)

seed = 30
np.random.seed(seed)
torch.manual_seed(seed)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


# Set model parameters
input_size = X_train.shape[1]
hidden_size = 100
output_size = 2

# Create an MLP model instance
model = MLP(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training model
num_epochs = 300
for epoch in range(num_epochs):
    # forward propagation
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make predictions on the test set
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    y_pred = predicted.numpy()

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy : ", acc)

y_scores = torch.softmax(outputs, dim=1)[:, 1].numpy()
auc = roc_auc_score(y_test, y_scores)
print("AUC: ", auc)

aupr = average_precision_score(y_test, y_scores)
print("AUPR: ", aupr)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)

data = test_data
y_true = test_label

data = data.loc[:, [col for col in data.columns if col[1] in ["mut_important", "cnv_del", "cnv_amp"]]]
data.columns = data.columns.get_level_values(0) + "_" + data.columns.get_level_values(1)

data = data.reindex(columns=data_filtered.columns, fill_value=0)
print(data.shape)

data_flat = data.values.reshape((data.shape[0], -1))
X_test_tensor = torch.tensor(data_flat, dtype=torch.float32)

# Make predictions on an external validation set
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    y_pred = predicted.numpy()

acc = accuracy_score(y_true, y_pred)
print("Test Accuracy : ", acc)

y_scores = torch.softmax(outputs, dim=1)[:, 1].numpy()
auc = roc_auc_score(y_true, y_scores)
print("AUC: ", auc)

aupr = average_precision_score(y_true, y_scores)
print("AUPR: ", aupr)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)
