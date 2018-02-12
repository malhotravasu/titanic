from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
#dataset = numpy.loadtxt("train.csv", delimiter=",", skiprows=1)
dataset = numpy.genfromtxt("train.csv", delimiter=",", skip_header=1, filling_values=0, autostrip=True)

# split into input (X) and output (Y) variables
X = dataset[:, [2,4,5,6,7,8]] #PClass,Sex,Age,SibSp,Parch,Fare
Y = dataset[:, [1]]

#testing data
test_data = numpy.genfromtxt("test_processed.csv", delimiter=",", skip_header=1, filling_values=0, autostrip=True)
P = test_data[:,[0]]
Q = test_data[:,[1,3,4,5,6,8]]

# create model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=50000, batch_size=50)

# calculate predictions
predictions = model.predict(Q)

# round predictions
result1 = P
result2 = numpy.array([int(round(x[0])) for x in predictions])
result3 = result2.reshape((418,1))
result = numpy.concatenate((result1, result3), axis=1).astype(numpy.int64)
#for x in range(0, 10):
#    print(str(x) + ' : ' + str(rounded[x]))
#print(result1)
#print(result)
numpy.savetxt("result.csv", result.astype(int), fmt='%i', delimiter=",")