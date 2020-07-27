from utils import*
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#getting data
path = 'myData'
data = importDataInfo(path)

#visualization
data= balanceData(data, display = False)


# converting to numpy
imagePath, steering = loadData(path,data)
print(imagePath[0], steering[0])

#split data

xTrain, xVal, yTrain, yVal= train_test_split(imagePath, steering, test_size= 0.2, random_state=5)
print("Total training images:", len(xTrain))
print("Total validation images:", len(xVal))

#Augmentation Data

#Preprocessing

#Create Model Nvidia
model= createModel()
model.summary()

#Train
history=model.fit(batchGen(xTrain, yTrain, 100, 1 ), steps_per_epoch = 20, epochs = 2,
          validation_data = batchGen(xVal,yVal, 10,0), validation_steps=20)


#save
model.save('model.h5')
print("Model is saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('loss')
plt.xlabel('Epoch')
plt.show()