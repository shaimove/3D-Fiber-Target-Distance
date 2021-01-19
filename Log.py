# Log.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

class TrackingLog(object):
    
    
    
    def __init__(self,folder,image_resolution,image_stats):
        
        # define folder to save results and image resolution
        self.folder = folder
        self.image_resolution = image_resolution
        self.image_stats = image_stats
        
        # define list of losses and epoch variable for training
        self.training_running_batch_loss = []
        self.training_loss_batch = []
        self.training_loss_epoch = []
        self.training_epoch = -1
        
        # define list of losses and epoch variable for training
        self.validation_running_batch_loss = []
        self.validation_loss_batch = []
        self.validation_loss_epoch = []
        self.validation_epoch = -1
        
    def BatchUpdate(self,mode,epoch,batch,loss):
        '''
        Update batch loss, of L1 score. 
        when a new epoch starts, using self.epoch indicator, we store the recorded
        losses and create new list of running losses. 

        Parameters
        ----------
        mode: string, training or validation 
        epoch : integer, 0 to number of epochs-1
        batch : integer, int 0 to number of batchs per epoch-1
        loss : float32 Tensor size 1, loss of L1 score

        '''
        # Step 1: Transform all the data to CPU and Numpy array
        loss = loss.to('cpu').detach().numpy()
        
        if mode == 'Training':
            # Step 2 - Training: If we started new epoch
            if epoch != self.training_epoch:
                # we are at a new epoch, store all previous batchs loss
                self.training_loss_batch.append(self.training_running_batch_loss)
                self.training_running_batch_loss = [] # restart the running batch loss
                self.training_epoch = epoch # update epoch number
            else:
                self.training_running_batch_loss.append(loss) # append loss
                
        elif mode == 'Validation':
            # Step 2 - Validation: If we started new epoch
            if epoch != self.validation_epoch:
                # we are at a new epoch, store all previous batchs loss
                self.validation_loss_batch.append(self.validation_running_batch_loss)
                self.validation_running_batch_loss = [] # restart the running batch loss
                self.validation_epoch = epoch # update epoch number
            else:
                self.validation_running_batch_loss.append(loss) # append loss
        
        
        return self
    
    def EpochUpdate(self,mode,epoch,loss):
        '''
        Save the loss for training or validation, at the end of every epoch

        Parameters
        ----------
        mode : string, training or validation 
        epoch : integer, 0 to number of epochs-1
        loss : float32 Tensor size 1, loss of L1 score

        '''

        if mode == 'Training':
            self.training_loss_epoch.append(loss) # we only need to update the loss
        elif mode == 'Validation':
            self.validation_loss_epoch.append(loss) # we only need to update the loss
            
        return self 
    
    
    def PlotLoss(self):
        '''
        At the end of training loop, plot the L1 loss per epoch, for bouth 
        Training and Validation

        '''
        # Create numpy array and figure
        plt.figure()
        loss_train = np.array(self.training_loss_epoch)
        loss_validation = np.array(self.validation_loss_epoch)
        
        # Plot
        plt.plot(range(len(self.training_loss_epoch)),loss_train,label='Training Loss')
        plt.plot(range(len(self.validation_loss_epoch)),loss_validation,label='Validation Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(); plt.legend()
        plt.title('Training and Validation L1 loss')
        
        return None
    
    def SaveResultsLastEpoch(self,images,output,labels):
        # Step 1: Transform all the data to CPU and Numpy array
        images = images.to('cpu').detach().numpy()
        output = output.to('cpu').detach().numpy()
        labels = labels.to('cpu').detach().numpy()
        
        # calculate predication
        predication_x = np.array(output[:,0]*self.image_resolution[0])
        predication_y = np.array(output[:,1]*self.image_resolution[1])         
        prediction = np.column_stack((predication_x,predication_y))
        
        # calculate ground truth
        position_x = np.array(labels[:,0]*self.image_resolution[0])
        position_y = np.array(labels[:,1]*self.image_resolution[1])         
        positions = np.column_stack((position_x,position_y))
        
        # calculate number of images
        num_of_img = len(position_x)
        
        # for every image, save prediction and true position in folder
        for i in range(num_of_img):
            # take image
            img = images[i][0]
            # multiply by std and ten add mean to get real image value
            img = img * self.image_stats[1] + self.image_stats[0]
            # convert to uint8
            img = np.uint8(img*255)
            # convert to BGR
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            
            # Plot predications in Blue
            cv2.circle(img, (int(prediction[i,0]),int(prediction[i,1])),
                       radius=5,color=(255, 0, 0),thickness=2)
            
            # Plot true position in Red
            cv2.circle(img, (int(positions[i,0]),int(positions[i,1])),
                       radius=5,color=(0, 0, 255),thickness=2)
            
            # save image
            filename = self.folder + str(i) + '.jpg'
            cv2.imwrite(filename,img)
        
        
        return None
        
        
        
        
        
        
        
        
        
    