

import urllib



# Download the data stored in a zipped numpy array from one of these two locations
# The uncommented one is likely to be faster. If you're running all your experiments
# on a machine at home rather than using colab, then make sure you save it 
# rather than repeatedly downloading it.
#!wget "https://sussex.box.com/shared/static/2nansy5fdps2dcycsqb7r06cddbbkskd.npz" -O training_images.npz
#!wget "http://users.sussex.ac.uk/~is321/training_images.npz" -O training_images.npz
response = urllib2.urlopen("http://users.sussex.ac.uk/~is321/training_images.npz", timeout = 5)
content = response.read()
f = open( "local/index.html", 'w' )

# The test images (without points)
!wget "http://users.sussex.ac.uk/~is321/test_images.npz" -O test_images.npz
# The example images are here
!wget "http://users.sussex.ac.uk/~is321/examples.npz" -O examples.npz
import numpy as np

# Load the data using np.load
data = np.load('training_images.npz', allow_pickle=True)

# Extract the images
images = data['images']
# and the data points
pts = data['points']



import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pandas as pd


scale = 250/96
training = 1498
testing = len(images)-training
npoints = pts.shape[1]
dimension = images.shape[1]


def flipPoints(p, l):
  points = p.copy()
  for i in range(0,len(points)):
    points[i][:,0] = (0 - points[i][:,0]) + l
  return points 

def flipImages(img):
  i = img.copy()
  for k in range(0, len(img)):
    i[k] = cv2.flip(i[k],1)
  return i

def augmentData(i,p):
  msk = np.random.rand(len(images)) < 0.3

  fi = i.copy()[msk]
  i = i.copy()[~msk]
  fp = p.copy()[msk]
  p = p.copy()[~msk]

  flipped = flipImages(fi)
  fi = np.append(fi, flipped)

  flippedp = flipPoints(fp, len(i[0]))
  fp = np.append(fp, flippedp)

  i = np.append(i, fi)
  p = np.append(p,fp)
  
  return i, p


im = images.copy()
pm = pts.copy()
im, pm = augmentData(im,pm)


im = np.reshape(im, (-1, dimension, dimension, 3))
pm = np.reshape(pm, (-1, npoints, 2))
print(im.shape)
print(pm.shape)

#plt.imshow(im[7000])
#plt.plot(pm[7000][:,0],pm[7000][:,1],"r+")


def processPoints(pts):
  p = pts.copy()
  p = p/scale
  p = p/96
  return p

processedPts = processPoints(pts)
print(processedPts.shape)
processedPts = np.reshape(processedPts, (-1, 1, 1, (2*npoints)))



#pre-processing

def processImages(images):
  img = images.copy()
  pi = np.array([])
  for x in range(0,len(img)): #len(img)
    i = cv2.cvtColor(img[x],cv2.COLOR_BGR2GRAY)
    i = cv2.resize(i, (96, 96))
    i = i/255
    pi = np.append(pi, i)
    if (x%500 == 0):
      percent = round(x/len(img)*100)
      print("{0}%".format(percent))

  return pi


processedImgs = processImages(images)
processedImgs = np.reshape(processedImgs , (-1,96,96,1))



#sampling
def testSampling(images, pts):
  msk = np.random.rand(len(images)) < 0.8
  x = images.copy()[msk]
  y = pts.copy()[msk]

  testx = images.copy()[~msk]
  testy = pts.copy()[~msk]
  print(len(x))
  print(len(testx))
  return x,y,testx,testy

trainx, trainy, testx, testy = testSampling(processedImgs, processedPts)


#pre-processing test 
testy = np.array([])
testx= np.array([])
for i in range(training,len(images)):
  for j in range(0,npoints):
    t = pts[i][j]/scale
    t=t/96
    testy=np.append(testy,(t))
  g = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
  r = cv2.resize(g, (96, 96))
  #r = cv2.rotate(r,cv2.ROTATE_90_COUNTERCLOCKWISE)
  #r = cv2.flip(r, 0)
  r=r/255
  testx = np.append(testx, r) 

testx=np.reshape(testx , (-1,96,96,1))
testy=np.reshape(testy , ( -1 , 1 , 1 , (2*npoints) ))


#pre-processing training 

y = np.array([])
x= np.array([])
for i in range(0,training):
  for j in range(0,npoints):
    t = pts[i][j]/scale
    t=t/96
    y=np.append(y,(t))
  g = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
  r = cv2.resize(g, (96, 96))
  #r = cv2.rotate(r,cv2.ROTATE_90_COUNTERCLOCKWISE)
  #r = cv2.flip(r, 0)
  r=r/255
  x = np.append(x, r) 

x=np.reshape(x , (-1,96,96,1))
y=np.reshape( y , ( -1 , 1 , 1 , (2*npoints) ))

print(y.shape)
print(x.shape)



#my one

print(trainx.shape)
print(trainy.shape)


model_layers = [
      
    tf.keras.layers.Conv2D( 256 , input_shape=( 96 , 96 , 1 ) , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.BatchNormalization(),

 

    tf.keras.layers.Conv2D( 84 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.Conv2D( 84 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    tf.keras.layers.Dense(30,activation="relu"),
    tf.keras.layers.Conv2D( 84 , kernel_size=( 3 , 3 ) , strides=1 ),
]

model = tf.keras.Sequential( model_layers )
model.compile( loss=tf.keras.losses.mean_squared_error , optimizer=tf.keras.optimizers.Adam( lr=0.0001 ) , metrics=[ 'mse' ] )
model.summary()




model.fit( trainx , trainy , epochs=80 , batch_size=32 , validation_data=( testx , testy ) )



fig = plt.figure(figsize=( 50 , 50 ))
for i in range( 1 , 12):
    sample_image = np.reshape( testx[i] * 255  , ( 96 , 96 ) ).astype( np.uint8 )
    #plt.imshow(sample_image)
    #plt.show()
    pred = model.predict( testx[ i : i +1  ] ) * 96
    pred = pred.astype( np.int32 )
    pred = np.reshape( pred[0 , 0 , 0 ] , ( npoints , 2 ) )
    fig.add_subplot( 1 , 12 , i )
    plt.imshow( sample_image , cmap='gray' )
    plt.scatter( pred[ : , 0 ] , pred[ : , 1 ] , c='red', marker="+" )
    #plt.scatter( pred[ 0:17 , 0 ] , pred[ 0:17 , 1 ] , c='red', marker="+" )
    prediction=pred
    
plt.show()





