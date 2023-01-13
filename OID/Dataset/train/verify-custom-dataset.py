"""task 1 
importing libraries"""



# Importing needed libraries
import matplotlib.pyplot as plt
import numpy as np
import h5py


"""task 2
Reading saved custom dataset"""


# Opening saved dataset from HDF5 binary file
# Initiating File object
# Opening file in reading mode by 'r'
with h5py.File('/home/souheil/OID-Image-Recognition/OID/Dataset/train/dataset_custom.hdf5', 'r') as f:
    # Showing all keys in the HDF5 binary file
    print(list(f.keys()))
    
    # Extracting saved arrays for training by appropriate keys
    # Saving them into new variables    
    x_train = f['x_train']  # HDF5 dataset
    y_train = f['y_train']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_train = np.array(x_train)  # Numpy arrays
    y_train = np.array(y_train)  # Numpy arrays
    
    
    # Extracting saved arrays for validation by appropriate keys
    # Saving them into new variables 
    x_validation = f['x_validation']  # HDF5 dataset
    y_validation = f['y_validation']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_validation = np.array(x_validation)  # Numpy arrays
    y_validation = np.array(y_validation)  # Numpy arrays
    
    
    # Extracting saved arrays for testing by appropriate keys
    # Saving them into new variables 
    x_test = f['x_test']  # HDF5 dataset
    y_test = f['y_test']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_test = np.array(x_test)  # Numpy arrays
    y_test = np.array(y_test)  # Numpy arrays

# """task 3 
# showing types and shapes of loaded arrays"""




print(type(x_train))
print(type(y_train))
print(type(x_validation))
print(type(y_validation))
print(type(x_test))
print(type(y_test))
print()


# Showing shapes of loaded arrays
print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)
print(x_test.shape)
print(y_test.shape)

# """task 4
# Defining list of classes' names"""



# Defining list of classes' names
# (!) Spell in the same way they are in Open Images Dataset
# (!) No need to use '_' if the name consists of two or more words
labels = ['Traffic sign' , 'Car' , 'Bus']

# """task 5
# plotting 100 images from custom dataset"""


# Magic function that renders the figure in a jupyter notebook
# instead of displaying a figure object
#%matplotlib inline


# Setting default size of the plot
plt.rcParams['figure.figsize'] = (10.0, 60.0)


# Defining a figure object with number of needed subplots
# ax is a (25, 4) numpy array
# To access specific subplot we call it by ax[0, 0]
figure, ax = plt.subplots(nrows=25, ncols=4)


# Plotting 100 examples along 25 rows and 4 columns
for i in range(25):
    for j in range(4):
        # Preparing random index
        ii = np.random.randint(low=0, high=x_train.shape[0])
        
        # Plotting current subplot
        ax[i, j].imshow(x_train[ii].astype('uint8'))
        
        # Giving name to current subplot
        # according to class's name in list 'labels'
        ax[i, j].set_title(labels[y_train[ii]], fontsize=16)
        
        # Hiding axis
        ax[i, j].axis('off')


# Adjusting distance between subplots
plt.tight_layout()


# Saving the plot
figure.savefig('plot_100_custom_images.png')


# Showing the plot
plt.show()

# """task 6
#  Plotting histogram to show distribution of images among classes"""



# Magic function that renders the figure in a jupyter notebook
# instead of displaying a figure object
#%matplotlib inline


# Setting default size of the plot
plt.rcParams['figure.figsize'] = (10.0, 7.0)


# Calculating number of images for every class
# Iterating all classes' indexes in 'y_train' array
# Using Numpy function 'unique'
# Returning sorted unique elements and their frequencies
classesIndexes, classesFrequency = np.unique(y_train, return_counts=True)


# Printing frequency (number of images) for every class
print('classes indexes:' , classesIndexes)
print('classes frequency:', classesFrequency)


# Plotting histogram of 5 classes with their number of images
# Defining a figure object 
figure = plt.figure()


# Plotting Bar chart
plt.bar(classesIndexes, classesFrequency, align='center', alpha=0.6)


# Giving name to Y axis
plt.ylabel('Class frequency', fontsize=16)


# Giving names to every Bar along X axis
plt.xticks(classesIndexes, labels, fontsize=16)


# Giving name to the plot
plt.title('Histogram of Custom Dataset', fontsize=20)


# Saving the plot
figure.savefig('histogram_custom_images.png')


# Showing the plot
plt.show()


# """"task 7"""

# print(help(np.unique))


# """task 8"""

# print(help(plt.bar))




# """task 9 

# comments 

# ### Some comments

# To get more details of usage Numpy function 'unique':  
# **print(help(np.unique))**  

# More details and examples are here:  
# https://numpy.org/doc/stable/reference/generated/numpy.unique.html  


# To get more details of usage Bar charts from matplotlib library:  
# **print(help(plt.bar))**  

# More details and examples are here:  
# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.bar.html


# """