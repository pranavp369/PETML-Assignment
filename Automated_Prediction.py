# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:09:19 2024

@author: prana
"""

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import docx
from docx import Document
import re
import copy
from collections import Counter

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import tqdm
import torch.optim as optim
from torch.optim import Adam
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


def load_data():
    #Loding the rCBF dynamic nifti dataset
    rcbf_dataset = nib.load(r"Assignment/training_images_rcbf.nii").get_fdata()
    
    #Loding the sbr dynamic nifti dataset
    sbr_dataset = nib.load(r"Assignment/training_images_sbr.nii").get_fdata()
    
    #Loding the rCBF dynamic nifti dataset
    rcbf_testing = nib.load(r"Assignment/test_images_rcbf.nii").get_fdata()
    
    #Loding the sbr dynamic nifti dataset
    sbr_testing = nib.load(r"Assignment/test_images_sbr.nii").get_fdata()

    #Loading the VOI data
    voi_template = nib.load(r"Assignment\VOI_template.nii").get_fdata()
    
    #Loading the Text Label Document
    label_doc = docx.Document(r"Assignment\Diagnoses of training data.docx")
    
    return rcbf_dataset,sbr_dataset,rcbf_testing,sbr_testing,voi_template,label_doc


def extract_selected_roi(input_dataset, voi_template,label_list,slice_no):

    label_find = np.in1d(voi_template, label_list).reshape(voi_template.shape)
            
    roi = np.where(label_find, 1, 0)
            
    #Create a nifti mask image of the segmented label.
    new_image = nib.Nifti1Image(roi, affine=np.eye(4))

    new_image = new_image.get_fdata()

    #Extract the VOI regions from the rCBF image
    input_data = input_dataset[:,:,:,slice_no]*new_image    
    
    input_data = input_data.astype(int)

    return input_data

def extract_labels(doc):
    pid = []
    labels = []
    count = 0

    for para in doc.paragraphs:
        line = para.text.strip()
        if count >3:
            label = re.findall(r'\b\d+\b',line)
            pid.append(int(label[0])-1)
            labels.append(int(label[1])-1)
        count +=1
    
    return pid,labels

def extract_features_test(rcbf_dataset,sbr_dataset,voi_template,mask_labels):
    
    for i in range(40):
        intensity = []
        
        for j in mask_labels:
            mask_rcbf = extract_selected_roi(rcbf_dataset, voi_template, [j], i)
            mask_sbr = extract_selected_roi(sbr_dataset, voi_template, [j], i)
            
            data_fft_rcbf = np.fft.fftn(mask_rcbf)
            data_fft_sbr = np.fft.fftn(mask_sbr)
    
            #Shift the zero frequency component to the center
            data_fft_shifted_rcbf = np.fft.fftshift(data_fft_rcbf)
            data_fft_shifted_sbr = np.fft.fftshift(data_fft_sbr)
    
            # Calculate the magnitude spectrum
            magnitude_spectrum_rcbf = np.abs(data_fft_shifted_rcbf)
            magnitude_spectrum_sbr = np.abs(data_fft_shifted_sbr)    
            
            #The Spatial Intensity based Features
            intensity.append(np.mean(mask_rcbf))
            intensity.append(np.mean(mask_sbr))
            intensity.append(np.std(mask_rcbf))
            intensity.append(np.std(mask_sbr))
            intensity.append(np.max(mask_rcbf))
            intensity.append(np.max(mask_sbr))
            
            #The Spectral Features
            intensity.append(np.mean(magnitude_spectrum_rcbf))
            intensity.append(np.mean(magnitude_spectrum_sbr))
            intensity.append(np.std(magnitude_spectrum_rcbf))
            intensity.append(np.std(magnitude_spectrum_sbr))
            intensity.append(np.max(magnitude_spectrum_rcbf))
            intensity.append(np.max(magnitude_spectrum_sbr))
            intensity.append(np.min(magnitude_spectrum_rcbf))
            intensity.append(np.min(magnitude_spectrum_sbr))
            
        
        #Getting the Histogram features
        
        min_rcbf = np.min(rcbf_dataset[:,:,:,i])
        max_rcbf = np.max(rcbf_dataset[:,:,:,i])
        min_sbr = np.min(sbr_dataset[:,:,:,i])
        max_sbr = np.max(sbr_dataset[:,:,:,i])
        
        # Normalize the data to the range [0, 255]
        normalized_rcbf = rcbf_dataset[:,:,:,i] #((rcbf_dataset[:,:,:,i] - min_rcbf) / (max_rcbf - min_rcbf)) * 255
        normalized_sbr = sbr_dataset[:,:,:,i] #((sbr_dataset[:,:,:,i] - min_sbr) / (max_sbr - min_sbr)) * 255

        # Convert the data to integers
        normalized_rcbf = normalized_rcbf.astype(np.uint8)
        normalized_sbr = normalized_sbr.astype(np.uint8)

        
        # Flatten the image array
        image_flat_rcbf = normalized_rcbf.flatten()
        image_flat_sbr = normalized_sbr.flatten()

        # Filter out zero values
        non_zero_values_rcbf = image_flat_rcbf[image_flat_rcbf >3]
        non_zero_values_sbr = image_flat_sbr[image_flat_sbr > 3]
        
        #Get the Histogram values
        hist_values_rcbf = plt.hist(non_zero_values_rcbf, bins=range(1,100))[0]
        hist_values_sbr = plt.hist(non_zero_values_sbr, bins=range(1,100))[0]
        
        #Convert to List
        hist_values_rcbf = hist_values_rcbf.tolist()
        hist_values_sbr = hist_values_sbr.tolist()
        #print(i)
        #print(labels[i])
        intensity.append(np.mean(hist_values_rcbf))
        intensity.append(np.std(hist_values_rcbf))
        intensity.append(np.mean(hist_values_sbr))
        intensity.append(np.std(hist_values_sbr))
        
        
        """
        for num in range(len(hist_values_rcbf)):
            intensity.append(hist_values_rcbf[num])
        for num in range(len(hist_values_sbr)):
            intensity.append(hist_values_sbr[num])
        """
        #intensity.append(int(pid[i]))
        #intensity.append(int(labels[i]))
        
        if i == 0:
           data_stack = np.array(intensity)
       
        else:
           data_stack = np.vstack((data_stack,np.array(intensity)))
      
        #print("Data #",i," : ",intensity)
    
    
    intensity_dataset = pd.DataFrame(data_stack)
    
    return intensity_dataset

def extract_features(rcbf_dataset,sbr_dataset,voi_template,mask_labels,pid,labels):
    
    for i in range(40):
        intensity = []
        
        for j in mask_labels:
            mask_rcbf = extract_selected_roi(rcbf_dataset, voi_template, [j], i)
            mask_sbr = extract_selected_roi(sbr_dataset, voi_template, [j], i)
            
            data_fft_rcbf = np.fft.fftn(mask_rcbf)
            data_fft_sbr = np.fft.fftn(mask_sbr)
    
            #Shift the zero frequency component to the center
            data_fft_shifted_rcbf = np.fft.fftshift(data_fft_rcbf)
            data_fft_shifted_sbr = np.fft.fftshift(data_fft_sbr)
    
            # Calculate the magnitude spectrum
            magnitude_spectrum_rcbf = np.abs(data_fft_shifted_rcbf)
            magnitude_spectrum_sbr = np.abs(data_fft_shifted_sbr)    
            
            #The Spatial Intensity based Features
            intensity.append(np.mean(mask_rcbf))
            intensity.append(np.mean(mask_sbr))
            intensity.append(np.std(mask_rcbf))
            intensity.append(np.std(mask_sbr))
            intensity.append(np.max(mask_rcbf))
            intensity.append(np.max(mask_sbr))
            
            #The Spectral Features
            intensity.append(np.mean(magnitude_spectrum_rcbf))
            intensity.append(np.mean(magnitude_spectrum_sbr))
            intensity.append(np.std(magnitude_spectrum_rcbf))
            intensity.append(np.std(magnitude_spectrum_sbr))
            intensity.append(np.max(magnitude_spectrum_rcbf))
            intensity.append(np.max(magnitude_spectrum_sbr))
            intensity.append(np.min(magnitude_spectrum_rcbf))
            intensity.append(np.min(magnitude_spectrum_sbr))
            
        
        #Getting the Histogram features
        
        min_rcbf = np.min(rcbf_dataset[:,:,:,i])
        max_rcbf = np.max(rcbf_dataset[:,:,:,i])
        min_sbr = np.min(sbr_dataset[:,:,:,i])
        max_sbr = np.max(sbr_dataset[:,:,:,i])
        
        #min_rcbf = np.min(mask_rcbf)
        #ax_rcbf = np.max(mask_rcbf)
        #min_sbr = np.min(mask_sbr)
        #max_sbr = np.max(mask_sbr)
        
        # Normalize the data to the range [0, 255]
        normalized_rcbf = mask_rcbf #((rcbf_dataset[:,:,:,i] - min_rcbf) / (max_rcbf - min_rcbf)) * 255
        normalized_sbr = mask_sbr #((sbr_dataset[:,:,:,i] - min_sbr) / (max_sbr - min_sbr)) * 255

        # Convert the data to integers
        normalized_rcbf = normalized_rcbf.astype(np.uint8)
        normalized_sbr = normalized_sbr.astype(np.uint8)

        
        # Flatten the image array
        image_flat_rcbf = normalized_rcbf.flatten()
        image_flat_sbr = normalized_sbr.flatten()

        # Filter out zero values
        non_zero_values_rcbf = image_flat_rcbf[image_flat_rcbf >3]
        non_zero_values_sbr = image_flat_sbr[image_flat_sbr > 3]
        
        #Get the Histogram values
        hist_values_rcbf = plt.hist(non_zero_values_rcbf, bins=range(1,100))[0]
        hist_values_sbr = plt.hist(non_zero_values_sbr, bins=range(1,100))[0]
        
        #Convert to List
        hist_values_rcbf = hist_values_rcbf.tolist()
        hist_values_sbr = hist_values_sbr.tolist()
        
        intensity.append(np.mean(hist_values_rcbf))
        intensity.append(np.std(hist_values_rcbf))
        intensity.append(np.mean(hist_values_sbr))
        intensity.append(np.std(hist_values_sbr))
        
        
        """
        for num in range(len(hist_values_rcbf)):
            intensity.append(hist_values_rcbf[num])
        for num in range(len(hist_values_sbr)):
            intensity.append(hist_values_sbr[num])
        """
        intensity.append(int(pid[i]))
        intensity.append(int(labels[i]))
        
        if i == 0:
           data_stack = np.array(intensity)
       
        else:
           data_stack = np.vstack((data_stack,np.array(intensity)))
           
        #print("Data #",i," : ",intensity)
    
    
    intensity_dataset = pd.DataFrame(data_stack)
    #print(intensity_dataset)
    
    return intensity_dataset


def combine_data(dataset1,dataset2):
    
    dataset_rcbf = pd.DataFrame(dataset1)
    dataset_sbr = pd.DataFrame(dataset2)
    
    comb_data = pd.concat([dataset_rcbf.iloc[:,:-2], dataset_sbr], axis=1)
    
    return comb_data



"""
Neural Networks
"""
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(228, 500),
            nn.ReLU(),
            nn.Linear(500,300),
            nn.ReLU(),
            nn.Linear(300,200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            #nn.ReLU(),
            #nn.Linear(10,1)
        )
        
        self.final = nn.Linear(10,4)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = self.final(logits)

        return logits

def get_accuracy(G, Y):
    return (G.argmax(dim=1) == Y).float().mean()
    
def Train_model(model,X_train,y_train,X_test,y_test,batchsize=1):
    
    model.to(device)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long).reshape(-1,1)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long).reshape(-1,1)


    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    

    # training parameters
    n_epochs = 2000  # number of epochs to run
    batch_size = 4  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    
    # Hold the best model
    best_loss = np.inf   # init to infinity
    best_weights = None
    best_epoch = 0
    best_acc =0
   
    
    #"""
    for i in range(n_epochs):
        train_history =[]
        test_history = []
        train_loss = []
        test_loss = []
        
        #print("Training")
        for j in range(X_train.shape[0]):
            
            X_training = X_train[j:j+batchsize].to(device)
            y_training = y_train[j:j+batchsize].to(device)
            #print(X_training)
            y_pred = model(X_training)
            
            loss = loss_fn(y_pred,y_training[0])
            
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            
            train_history.append(get_accuracy(y_pred,y_training[0]).item())
            train_loss.append(loss.cpu().detach().numpy())
        
        #print("Testing")
        for j in range(X_test.shape[0]):
            
            model.eval()
            
            X_testing = X_test[j:j+batchsize].to(device)
            y_testing = y_test[j:j+batchsize].to(device)
            
            test_pred = model(X_testing)
                
                
            loss_val = loss_fn(test_pred, y_testing[0])
            loss_val = loss_val
            
            test_history.append(get_accuracy(test_pred,y_testing[0]).item())
            test_loss.append(loss_val.cpu().detach().numpy())
        
        
        if np.mean(test_loss)<=best_loss:
            best_epoch = i
            best_loss = np.mean(test_loss)
            best_acc = np.mean(test_history)
            best_model = copy.deepcopy(model.state_dict())
            
    
    model.load_state_dict(best_model)
    
    return model

def majority_votes(list1, list2, list3, list4, list5):#
    # Ensure all lists are of the same length
    assert len(list1) == len(list2) == len(list3)== len(list4) == len(list5), "All lists must be of the same length"
    
    majority_votes = []
    
    # Iterate over indices
    for i in range(len(list1)):
        # Get the elements at the current index from all three lists
        element1 = list1[i]
        element2 = list2[i]
        element3 = list3[i]
        element4 = list4[i]
        element5 = list5[i]
        
        # Count occurrences of each element
        counts = Counter([element1, element2, element3,element4, element5])
        # Find the element with the highest count
        majority_element, majority_count = counts.most_common(1)[0]
        # Check if the count of the majority element is greater than half the total count
        if majority_count > 2:
            majority_votes.append(majority_element)
        else:
            majority_votes.append(majority_element)  # No majority vote
        
    return majority_votes     


if __name__ == '__main__':
    
    #Extract the given 16 labels from the 50 label VOI
    entire_mask_labels = [1,2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,54,80,90,105,110,120,125]
    label_list = [10,11,16,17,18,19,28,29,34,35,39,40,80,90,105,120]
    
    
    #Load the input data
    rcbf_dataset,sbr_dataset,rcbf_testing,sbr_testing,voi_template,label_doc = load_data()
    
    #Extract the labels from the text document
    pid,labels = extract_labels(label_doc)
    
    intensity_dataset = extract_features(rcbf_dataset,sbr_dataset,voi_template,label_list,pid,labels)
    
    test_dataset = extract_features_test(rcbf_dataset,sbr_dataset,voi_template,label_list)
    
    

    
    models = []
    predictions =[]
    
    for i in range(5):
        print("Training Model #",i)
        #Split data into training and testing sets
        training_dataset,testing_dataset = train_test_split(intensity_dataset,train_size=0.75,shuffle=True)

        X_train = training_dataset.iloc[:,:-2]
        y_train = training_dataset.iloc[:,-1]
        X_test = testing_dataset.iloc[:,:-2]
        y_test = testing_dataset.iloc[:,-1]
    
    
        model = Net()
        
        model_dict = Train_model(model,X_train,y_train,X_test,y_test,batchsize=1)
        
        name = ['model1.pt','model2.pt','model3.pt','model4.pt','model5.pt']
        torch.save(model.state_dict(),name[i])
    
        models.append(model_dict)
    
    for net in models:
        
        model = Net()
        
        #Tensor Convertion of Testing Dataset
        X_hidden = torch.tensor(test_dataset.values, dtype=torch.float32)
        test_prediction = net(X_hidden).argmax(dim=1)
        predictions.append(test_prediction.numpy().tolist())
        print(test_prediction)     
    
    print(majority_votes(predictions[0],predictions[1], predictions[2],predictions[3], predictions[4]))
        
        
        
        
        
        
        
        
        
        
        
     


