# Main.py
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from torchsummary import summary

#import model
import model_transfer
import utils
import Log
from Dataset import DatasetImage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Step 1: Create datasets
# define parameters for training
dataframe_path = '../Dataset/Dataset.csv'
image_size = (1024,1024)

# Divide dataset between training and validation randomly, 80-20 split
data_table = pd.read_csv(dataframe_path)
data_train = data_table.sample(frac = 0.8)
data_validation = data_table.drop(data_train.index)

# calculate stats from training data
mean = [0.485, 0.456, 0.406] # for imagenet
std = [0.229, 0.224, 0.225] # for imagenet

#%% Step 2: Create dataset objects
batch_size_train = 32
batch_size_validation = 32

# define datatransforms
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)])

# define dataset and dataloader for training
train_dataset = DatasetImage(data_train,image_size,transform)
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetImage(data_validation,image_size,transform)
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)

# Define logger parameters 
model_name = 'Version 3-26_01_2021.pt'
folder = '../Resnet101 strong maxpool 50 epochs/'
Logger = Log.TrackingLog(folder,image_size,mean,std)


#%% Step 3: Define model hyperparameters
# number of epochs
num_epochs = 50

# load model
model = model_transfer.TrackingModel().to(device)
utils.count_parameters(model)

# send parameters to optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# define loss function 
criterion = torch.nn.L1Loss()

#%% Step 4: Training Loop


for epoch in range(num_epochs):
    ##################
    ### TRAIN LOOP ###
    ##################
    # set the model to train mode
    model.train()
    
    # initiate training loss
    train_loss = 0
    i = 0 # index for log of batchs
    
    for batch in train_loader:
        # get batch images and labels
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # clear the old gradients from optimizer
        optimizer.zero_grad()
        
        # forward pass: feed inputs to the model to get outputs
        output = model(images)
        
        # calculate the training batch loss
        loss = criterion(output,labels)
        
        # add L1 regularization
        #loss = loss + utils.Regularization(model,2,0.001)
        
        # backward: perform gradient descent of the loss w.r. to the model params
        loss.backward()
        
        # update the model parameters by performing a single optimization step
        optimizer.step()
        
        # accumulate the training loss
        train_loss += loss.item()
        
        # update training log
        print('Epoch %d, Batch %d/%d, training loss: %.4f' % (epoch+1,i,len(train_loader),loss))
        Logger.BatchUpdate(mode='Training',epoch=epoch,batch=i,loss=loss)
        i += 1 # update index

            
    #######################
    ### VALIDATION LOOP ###
    #######################
    # set the model to eval mode
    model.eval()
    
    # initiate validation loss
    valid_loss = 0
    i = 0 # index for log of batchs
    
    # turn off gradients for validation
    with torch.no_grad():
        for batch in validation_loader:
            # get batch images and labels
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # forward pass
            output = model(images)
            
            # validation batch loss
            #loss = criterion(output, torch.max(labels, 1)[1]) 
            loss = criterion(output,labels)
            
            # accumulate the valid_loss
            valid_loss += loss.item()
            
            # update validation log
            print('Epoch %d, Batch %d/%d, validation loss: %.4f' % (epoch+1,i,len(validation_loader),loss))
            Logger.BatchUpdate(mode='Validation',epoch=epoch,batch=i,loss=loss)
            i += 1 # update index for log of batchs 
            
                
                
    #########################
    ## PRINT EPOCH RESULTS ##
    #########################
    train_loss /= len(train_loader)
    valid_loss /= len(validation_loader)
    # update training and validation loss
    Logger.EpochUpdate(mode='Training',epoch=epoch,loss=train_loss)
    Logger.EpochUpdate(mode='Validation',epoch=epoch,loss=valid_loss)
    # print results
    print('Epoch: %s/%s: Training loss: %.3f. Validation Loss: %.3f.'
          % (epoch+1,num_epochs,train_loss,valid_loss))
 

#%% Step 5: print results and save the model
# save images for example
Logger.SaveResultsLastEpoch(images,output,labels)

# Plot Traing and Validation Loss graph
Logger.PlotLoss()

# Plot distribution of weights values, to detect high value
weigths = utils.PlotWeightsHistogram(model)

# model summary
summary(model, images)

# Svae the model
#PATH = folder + model_name
#torch.save({'model_state_dict': model.state_dict(), PATH)


