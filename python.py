import streamlit as st
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# Function definitions (paste your function definitions here)

genuine_image_paths = "A:\\Signature Forgery Detection Project\\real"
forged_image_paths = "A:\\Signature Forgery Detection Project\\forged"


def rgbgrey(img):
    # Converts rgb to grayscale
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg

def greybin(img):
    # Converts grayscale to binary
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
#     img = ndimage.binary_erosion(img).astype(img.dtype)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg

def preproc(path, img=None, display=True):
    if img is None:
        img = mpimg.imread(path)
    if display:
        plt.imshow(img)
        plt.show()
    grey = rgbgrey(img) #rgb to grey
    if display:
        plt.imshow(grey, cmap = matplotlib.cm.Greys_r)
        plt.show()
    binimg = greybin(grey) #grey to binary
    if display:
        plt.imshow(binimg, cmap = matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg==1)
    # Now we will make a bounding box with the boundary as the position of pixels on extreme.
    # Thus we will get a cropped image with only the signature part.
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap = matplotlib.cm.Greys_r)
        plt.show()
    return signimg

def Ratio(img):
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                a = a+1
    total = img.shape[0] * img.shape[1]
    return a/total

def Centroid(img):
    numOfWhites = 0
    a = np.array([0,0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                b = np.array([row,col])
                a = np.add(a,b)
                numOfWhites += 1
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a/numOfWhites
    centroid = centroid/rowcols
    return centroid[0], centroid[1]

def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].eccentricity, r[0].solidity


def SkewKurtosis(img):
    h,w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    #calculate projections along the x and y axes
    xp = np.sum(img,axis=0)
    yp = np.sum(img,axis=1)
    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    sx = np.sqrt(np.sum(x2*xp)/np.sum(img))
    sy = np.sqrt(np.sum(y2*yp)/np.sum(img))
    
    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    skewx = np.sum(xp*x3)/(np.sum(img) * sx**3)
    skewy = np.sum(yp*y3)/(np.sum(img) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp*x4)/(np.sum(img) * sx**4) - 3
    kurty = np.sum(yp*y4)/(np.sum(img) * sy**4) - 3

    return (skewx , skewy), (kurtx, kurty)



def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc(path, display=display)
    ratio = Ratio(img)
    centroid = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)
    return retVal


def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features

def makeCSV():
    if not(os.path.exists('A:\\Signature Forgery Detection Project\\Features')):
        os.mkdir('A:\\Signature Forgery Detection Project\\Features')
        print('New folder "Features" created')
    if not(os.path.exists('A:\\Signature Forgery Detection Project\\Features/Training')):
        os.mkdir('A:\\Signature Forgery Detection Project\\Features/Training')
        print('New folder "Features/Training" created')
    if not(os.path.exists('A:\\Signature Forgery Detection Project\\Features/Testing')):
        os.mkdir('A:\\Signature Forgery Detection Project\\Features/Testing')
        print('New folder "Features/Testing" created')
    # genuine signatures path
    gpath = genuine_image_paths
    # forged signatures path
    fpath = forged_image_paths
    for person in range(1,13):
        per = ('00'+str(person))[-3:]
        print('Saving features for person id-',per)
        
        with open('A:\\Signature Forgery Detection Project\\Features\\Training/training_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Training set
            for i in range(0,3):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(0,3):
                source = os.path.join(fpath, '021'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
        
        with open('A:\\Signature Forgery Detection Project\\Features\\Testing/testing_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Testing set
            for i in range(3, 5):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(3,5):
                source = os.path.join(fpath, '021'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
            



def testing(path):
    feature = getCSVFeatures(path)
    if not(os.path.exists('A:\\Signature Forgery Detection Project/TestFeatures')):
        os.mkdir('A:\\Signature Forgery Detection Project/TestFeatures')
    with open('A:\\Signature Forgery Detection Project\\TestFeatures/testcsv.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature))+'\n')

    
makeCSV()

# ...

n_input = 9
def readCSV(train_path, test_path, type2=False):
    # Reading train data
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = keras.utils.to_categorical(correct, 2)  # Converting to one hot
    # Reading test data
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not type2:
        df = pd.read_csv(test_path, usecols=(n_input,))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = keras.utils.to_categorical(correct, 2)  # Converting to one hot
    if not type2:
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input


ops.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 1000

# Network Parameters
n_hidden_1 = 7  # 1st layer number of neurons
n_hidden_2 = 10  # 2nd layer number of neurons
n_hidden_3 = 30  # 3rd layer
n_classes = 2  # no. of classes (genuine or forged)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], seed=2))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes], seed=4))
}


# Create model
def multilayer_perceptron(x):
    layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer

loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# For accuracies
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

def evaluate(train_path, test_path, type2=False):
    if not type2:
        train_input, corr_train, test_input, corr_test = readCSV(train_path, test_path)
    else:
        train_input, corr_train, test_input = readCSV(train_path, test_path, type2)
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
            if cost < 0.0001:
                break
        
        if type2 is False:
            accuracy2 = accuracy.eval({X: test_input, Y: corr_test})
            return accuracy2
        else:
            prediction = pred.eval({X: test_input})
            if prediction[0][1] > prediction[0][0]:
                return "Genuine"  # "Genuine Image"
            else:
                return "Forged"  # "Forged Image"


# Streamlit app
st.title("Signature Verification")

with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used _supervised learning to classify forged and real images. '
             'Supervised learning involves training a model on a labeled dataset,'
             'which means that each training example is paired with an output label.'
             'In this code, the MLP is trained using labeled data where signatures are labeled as either '
             'genuine (1) or forged (0).'
             'I used a _multilayer perceptron_ model with 3 hidden layers. '
             )

    st.subheader('Results')
    st.write('I used Multilayer Perceptron (MLP) Algorithm and tested then implemented'
             'An MLP is a class of feedforward artificial neural network (ANN).'
             'It consists of at least three layers of nodes: an input layer,'
             'one or more hidden layers, and an output layer.'
             'Each node (except for the input nodes) is a neuron that uses a nonlinear activation function (in this case, tanh is used)')

    # Algorithm, Accuracy = st.columns(2)

    # with st.container():
    #     Algorithm.write("MLP")
    #     Accuracy.write("0.6666667")

    data = {'Algorithm': [ 'Multilayer Perceptron (MLP)' ],
        'Accuracy': [ 0.6666667 ],
        }

    index = ['MLP']

    df_results = pd.DataFrame(data=data, index=index)
    st.table(df_results)

    st.write('MLP --> Multilayer Perceptron')

# Input ID from user
train_person_id = st.text_input("Enter person's ID:", "")

# Upload the signature image
uploaded_file = st.file_uploader("Upload the signature image:", type=["png", "jpg", "jpeg"])

if st.button("Verify Signature") and uploaded_file is not None and train_person_id:
    # Save the uploaded image
    image_path = f"uploaded_image.{uploaded_file.name.split('.')[-1]}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        # st.image(image_path, caption='Uploaded Signature Image', use_column_width=True)
        st.success("Image saved!")

    # Display the uploaded image
    st.image(image_path, caption='Uploaded Signature Image', use_column_width=True)

    # Define paths
    train_path = f"A:\\Signature Forgery Detection Project\\Features\\Training\\training_{train_person_id}.csv"
    test_path = 'A:\\Signature Forgery Detection Project\\TestFeatures\\testcsv.csv'

    # Process the test image
    testing(image_path)

    # Evaluate the image
    result = evaluate(train_path, test_path, type2=True)
    # st.balloons()
    # Display the result
    st.text_input("The signature is:", result)
else:
    st.write("Please enter the ID and upload the image to verify.")

