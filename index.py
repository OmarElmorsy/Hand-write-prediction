import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt 
import numpy as np 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data();

# scaling data 
x_train = x_train/255
x_test = x_test/255 

# reshape data 
x_train_reshape =  x_train.reshape(len(x_train), len(x_train[0])*len(x_train[0]));
x_test_reshape =  x_test.reshape(len(x_test), len(x_test[0])*len(x_test[0]));

model = Sequential([
    Dense(100, input_shape=(len(x_train_reshape[0]),), activation='relu'),
    Dense(50, activation='sigmoid'),
    Dense(10, activation='sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
)

model.fit(x_train_reshape, y_train, epochs=10)
indexErorr = []
predict  = model.predict(x_test_reshape)
count=0;
for i in range(len(y_test)):
  if y_test[i]==np.argmax(predict[i]):
    count=count+1
  else:
    indexErorr.append(i)

print("\n The number of samples that were mistakenly predict : ", len(indexErorr))

# to show image in vs code  
# 1- right click on code
# 2- chose run in interactive window 
# 3- chose run current file in interactive window
print("Number of Error Predict are : ", len(indexErorr))
print("We can see any example from indexError list ----->")
print("One of them is first element : is predict ", np.argmax(predict[indexErorr[0]]))
print("But is in real  it is ", y_test[indexErorr[0]])
print("As shown in the picture below ")

plt.imshow(x_test[indexErorr[0]])
