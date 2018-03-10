from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from keras.optimizers import RMSprop
from numpy import genfromtxt,reshape,array
from os import listdir
from cv2 import imread,cvtColor,COLOR_RGB2GRAY,resize
from sklearn.model_selection import train_test_split

'''#Fetch data way 1
print('Reading data...')
data = genfromtxt('data.csv',delimiter=',')
X = reshape(data[:,:-1],(data.shape[0],150,150,3))
y = data[:,-1]
print('Data: ',X.shape)'''

#Fetch data way 2
X=[]
y=[]
for c in listdir('data'):
	i = listdir('data').index(c)
	print('Reading',c)
	for sample in listdir('data/'+c):
		path = 'data/' + c + '/' + sample
		img = imread(path)
		#img = cvtColor(img,COLOR_RGB2GRAY)
		img = resize(img,(150,150))
		X.append(img)
		y.append(i)
X = array(X)
y = array(y)
print('Data: ',X.shape)

#Model
model = Sequential([])
model.add(Conv2D(32,3,activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=4))

model.add(Conv2D(64,3,activation='relu'))
model.add(MaxPooling2D(pool_size=4))

model.add(Conv2D(128,3,activation='relu'))
model.add(MaxPooling2D(pool_size=4))

model.add(Flatten())
model.add(Dense(4,activation='softmax'))

RMSprop(lr=0.01)
model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

#Train
trainX,testX,trainy,testy = train_test_split(X,y,test_size=0.25)
model.fit(trainX,trainy,epochs=3)

#Test
print(model.evaluate(testX,testy,verbose=1))