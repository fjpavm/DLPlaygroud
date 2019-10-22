import keras
from keras.utils import np_utils as numpy_utils
from keras import backend as K
import numpy
    
if __name__ == '__main__':
    
    #dataset = keras.datasets.mnist
    dataset = keras.datasets.fashion_mnist
    #dataset = keras.datasets.cifar10
    #dataset = keras.datasets.cifar100
    channels_first_format = K.image_data_format() == 'channels_first'
    (train_input, train_output), (test_input, test_output) = dataset.load_data()
    
    dataset_in_shape=numpy.shape( train_input)
    print('in shape: ' + str(dataset_in_shape))
    print('in num dim: ' + str(numpy.ndim(train_input)))
    
    dataset_out_shape=numpy.shape( train_output)
    print('out shape: ' + str(dataset_out_shape))
    print('out num dim: ' + str(numpy.ndim(train_output)))
    train_output_categorical = numpy_utils.to_categorical(train_output)
    print(train_output[0])
    print(train_output_categorical[0])
    