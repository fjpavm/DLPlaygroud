import keras
from keras.utils import np_utils as numpy_utils
from keras import backend as K
import numpy

class DataLoader:
    def __init__(self):
        self.datasetInfo = dict()
        self.datasetInfo["mnist"] = { 'name' : 'mnist',
                                     'source' : 'keras',
                                     'type' : 'supervised',
                                     'input' : {'type' : 'bw_image',
                                                'image_size' : {'width' : 28, 'height': 28}
                                                },
                                     'output' : {'type' : 'categorical',
                                                 'categories' : ['0','1','2','3','4','5','6','7','8','9']
                                                 },
                                     'loader' : self.mnistLoader
                                     }
    
    def mnistLoader(self):
        print('loading mnist')
        return self.kerasBWImageLoader(keras.datasets.mnist)
        
    def kerasBWImageLoader(self, in_dataset):
        channels_first_format = K.image_data_format() == 'channels_first'
        
        # load data in with keras
        (trainInput, trainOutput), (testInput, testOutput) = in_dataset.load_data()
        
        # add missing channel shape index needed for keras conv2d layers where needed
        if channels_first_format:
            trainInput = trainInput.reshape(trainInput.shape[:1]+(1,)+trainInput.shape[1:])
            testInput = testInput.reshape(testInput.shape[:1]+(1,)+testInput.shape[1:])
        else:
            trainInput = trainInput.reshape(trainInput.shape+(1,))
            testInput = testInput.reshape(testInput.shape+(1,))
            
        # converting input images to float intensity
        trainInput = trainInput.astype('float32')
        trainInput /= 255
        testInput = testInput.astype('float32')
        testInput /= 255
        
        # converting outputs to categorical
        trainOutput = numpy_utils.to_categorical(trainOutput)
        testOutput = numpy_utils.to_categorical(testOutput)
        
        outDict = {'train' : {'input' : trainInput, 'output' : trainOutput},
                   'test' : {'input' : testInput, 'output' : testOutput}
                   }
        return outDict
            
        
    def loadByName(self, in_name):
        if in_name in self.datasetInfo:
            return self.datasetInfo[in_name]['loader']()
        
    def load(self, in_name):
        return self.loadByName(in_name)
    

if __name__ == '__main__':
    # Test Data loader
    dl = DataLoader()
    data =  dl.load('mnist')
    for dataset in data:
        for type in data[dataset]:
            print(dataset + ' ' + type + ' shape: ' + str(data[dataset][type].shape))
    