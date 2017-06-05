## Helper code to generate mini batches of CIFAR-10 dataset
import glob
import pickle as pkl
import numpy as np
from scipy import misc

class CIFAR(object):
    
    def __init__(self,path):
        self.path = path
        self.data = []
        self.labels = []
        for file_name in glob.glob(path):
            print file_name
            try:
                with open(file_name,"rb") as fp:
                    batch = pkl.load(fp)
                    self.data.append(batch['data'])
                    self.labels.append(np.asarray(batch['labels']))
            except:
                pass
        self.data = np.asarray(self.data)
        self.data = self.data.reshape(self.data.shape[0]*self.data.shape[1],self.data.shape[2])
        self.labels = np.asarray(self.labels)
        self.labels = self.labels.reshape(self.labels.shape[0]*self.labels.shape[1])

    def __str__(self):
        return str((self.data.shape,self.labels.shape))

    def generate_batches(self,cur_batch,batch_size):
        return (self.data[cur_batch:(cur_batch+1)*batch_size,:],self.labels[cur_batch:(cur_batch+1)*batch_size])

    def resize_image(self,new_size):
        new_data = []
        for row in self.data:
            img = row.reshape((32,32,3))
            img = misc.imresize(img,new_size)
            assert img.shape == new_size
            new_data.append(img.reshape(new_size[0]*new_size[1]*new_size[2]))
        return np.asarray(new_data)

    def drop_color_channel(self,data,old_size,new_size):
        new_data = []
        for row in data:
            img = row.reshape(old_size)[:,:,0]
            assert img.shape == new_size
            new_data.append(img.reshape(new_size[0]*new_size[1]))
        return np.asarray(new_data)

def main():
    
    path = '../../../Dataset/CIFAR-10/*'
    cifar = CIFAR(path)
    print cifar
    x_batch, y_batch = cifar.generate_batches(0,50)
    print x_batch.shape
    print y_batch.shape
    new_data = cifar.resize_image((28,28,3))
    print new_data.shape
    new_data = cifar.drop_color_channel(new_data,(28,28,3),(28,28))
    print new_data.shape

if __name__ == "__main__":
    main()