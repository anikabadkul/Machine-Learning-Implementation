import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0) #setting a random seed for reproducibility

def data(file_path): #loads and analyzes the training data
    #question 1
    train_data = pd.read_csv(file_path) #loads the training data
    #question 2
    description = train_data.describe() #describes the dataset
    print(description) #prints the statistics
    #question 3
    plt.hist(train_data['Price'], bins = 20, edgecolor = 'black') #plots histogram of price
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.title("Histogram of the Price")
    plt.show()
    #question 4
    columns = ['GrLivArea', 'BedroomAbvGr', 'TotalBsmtSF', 'FullBath'] #plots scatter matrix for GrLivArea, BedroomAbvGr, TotalBsmtSF, FullBath
    pd.plotting.scatter_matrix(train_data[columns], alpha = 0.5, figsize = (10,10), diagonal = 'hist')
    plt.show()

    return train_data
def pred(x,w): #question 5
    return np.dot(x,w) #dot product of features and weights

def loss(y,y_pred): #question 6
    return np.mean((y-y_pred)**2) #calculates MSE

def gradient(x,y,y_pred): #question 7
    return (2/len(y))*np.dot(x.T, (y_pred - y)) #computes gradient for gradient descent

def update(w, grad, learning_rate): #question 8
    return w - learning_rate*grad #updates the weights during training

def train(train_data, learning_rate, iterations): #function for train model
    features = train_data.iloc[:,1:26].values #extracts features
    y = train_data['Price'].values #extracts price
    weights = np.random.rand(25) #random weights initialized
    mse_history = [] #list to store MSE after each iteration

    for x in range(iterations):
        predictions = pred(features,weights) #makes predictions
        current_loss = loss(y, predictions) #calculates MSE
        grad = gradient(features, y, predictions) #calculates gradient
        weights = update(weights, grad, learning_rate) #updates the weights
        mse_history.append(current_loss) #appends the MSE

    final_mse = mse_history.pop() #Final MSE
    return mse_history, final_mse, weights

def test(test_data, weights): #function for test model
    features = test_data.iloc[:,1:26].values #extracts features
    y = test_data['Price'].values #extracts price
    y_pred = pred(features, weights) #makes predictions
    return loss(y, y_pred) #calculates and returns MSE

#file path for train and test model
file_path_train = 'train.csv'
file_path_test = 'test.csv'
#loads and analysis training and test data
train_data = data(file_path_train)
test_data = pd.read_csv(file_path_test)

#COMMENTED OUT FOR QUESTION 10
#learning_rate = 0.2
#iterations = 35000
#mse_history, final_mse, weights = train(train_data, learning_rate, iterations)
#plt.plot(range(iterations),mse_history,label = f'Learning Rate={learning_rate}')

#question 11
learning_rate_1 = 10e-11
learning_rate_2 = 10e-12
iterations = 35000

mse_history_1, final_mse1_train, final_weights_1 = train(train_data, learning_rate_1, iterations)
mse_history_2, final_mse2_train, final_weights_2 = train(train_data, learning_rate_2, iterations)
#prints MSE for training data
print(f"MSE of train model for learning rate 10e-11:{final_mse1_train}")
print(f"MSE of train model for learning rate 10e-12:{final_mse2_train}")

final_mse1_test = test(test_data, final_weights_1)
final_mse2_test = test(test_data, final_weights_2)
#prints MSE for test data
print(f"MSE of test model for learning rate 10e-11:{final_mse1_test}")
print(f"MSE of test model for learning rate 10e-12:{final_mse2_test}")

#Plots MSE history for question 11
plt.plot(mse_history_1, label = 'Learning Rate = 10e-11')
plt.plot(mse_history_2, label = 'Learning Rate = 10e-12')
plt.title('MSE Over 35,000 Iterations For Different Learning Rates')
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.legend()
plt.show()import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0) #setting a random seed for reproducibility

def data(file_path): #loads and analyzes the training data
    #question 1
    train_data = pd.read_csv(file_path) #loads the training data
    #question 2
    description = train_data.describe() #describes the dataset
    print(description) #prints the statistics
    #question 3
    plt.hist(train_data['Price'], bins = 20, edgecolor = 'black') #plots histogram of price
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.title("Histogram of the Price")
    plt.show()
    #question 4
    columns = ['GrLivArea', 'BedroomAbvGr', 'TotalBsmtSF', 'FullBath'] #plots scatter matrix for GrLivArea, BedroomAbvGr, TotalBsmtSF, FullBath
    pd.plotting.scatter_matrix(train_data[columns], alpha = 0.5, figsize = (10,10), diagonal = 'hist')
    plt.show()

    return train_data
def pred(x,w): #question 5
    return np.dot(x,w) #dot product of features and weights

def loss(y,y_pred): #question 6
    return np.mean((y-y_pred)**2) #calculates MSE

def gradient(x,y,y_pred): #question 7
    return (2/len(y))*np.dot(x.T, (y_pred - y)) #computes gradient for gradient descent

def update(w, grad, learning_rate): #question 8
    return w - learning_rate*grad #updates the weights during training

def train(train_data, learning_rate, iterations): #function for train model
    features = train_data.iloc[:,1:26].values #extracts features
    y = train_data['Price'].values #extracts price
    weights = np.random.rand(25) #random weights initialized
    mse_history = [] #list to store MSE after each iteration

    for x in range(iterations):
        predictions = pred(features,weights) #makes predictions
        current_loss = loss(y, predictions) #calculates MSE
        grad = gradient(features, y, predictions) #calculates gradient
        weights = update(weights, grad, learning_rate) #updates the weights
        mse_history.append(current_loss) #appends the MSE

    final_mse = mse_history.pop() #Final MSE
    return mse_history, final_mse, weights

def test(test_data, weights): #function for test model
    features = test_data.iloc[:,1:26].values #extracts features
    y = test_data['Price'].values #extracts price
    y_pred = pred(features, weights) #makes predictions
    return loss(y, y_pred) #calculates and returns MSE

#file path for train and test model
file_path_train = 'train.csv'
file_path_test = 'test.csv'
#loads and analysis training and test data
train_data = data(file_path_train)
test_data = pd.read_csv(file_path_test)

#COMMENTED OUT FOR QUESTION 10
#learning_rate = 0.2
#iterations = 35000
#mse_history, final_mse, weights = train(train_data, learning_rate, iterations)
#plt.plot(range(iterations),mse_history,label = f'Learning Rate={learning_rate}')

#question 11
learning_rate_1 = 10e-11
learning_rate_2 = 10e-12
iterations = 35000

mse_history_1, final_mse1_train, final_weights_1 = train(train_data, learning_rate_1, iterations)
mse_history_2, final_mse2_train, final_weights_2 = train(train_data, learning_rate_2, iterations)
#prints MSE for training data
print(f"MSE of train model for learning rate 10e-11:{final_mse1_train}")
print(f"MSE of train model for learning rate 10e-12:{final_mse2_train}")

final_mse1_test = test(test_data, final_weights_1)
final_mse2_test = test(test_data, final_weights_2)
#prints MSE for test data
print(f"MSE of test model for learning rate 10e-11:{final_mse1_test}")
print(f"MSE of test model for learning rate 10e-12:{final_mse2_test}")

#Plots MSE history for question 11
plt.plot(mse_history_1, label = 'Learning Rate = 10e-11')
plt.plot(mse_history_2, label = 'Learning Rate = 10e-12')
plt.title('MSE Over 35,000 Iterations For Different Learning Rates')
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.legend()
plt.show()
