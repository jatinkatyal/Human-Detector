"""Generate data.csv from data folder in working directory. """
#from skimage.data import imread
#from skimage.transform import rescale
from cv2 import imread,cvtColor,COLOR_RGB2GRAY,resize
from os import listdir
from numpy import array,savetxt,hstack

data=[]
for c in listdir('data'):
	i = listdir('data').index(c)
	print('Reading',c)
	for sample in listdir('data/'+c):
		path = 'data/' + c + '/' + sample
		img = imread(path)
		img = cvtColor(img,COLOR_RGB2GRAY)
		#print('read',path)

		img = resize(img,(150,150))
		row = hstack([img.ravel(),i])
		data.append(row)

data = array(data)
savetxt('data.csv',data,delimiter=',',fmt='%i')
exit(0)