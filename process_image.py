import numpy as np 
import matplotlib.pyplot as plt 
import os, glob
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import cv2
image_size=64

def read_data():
    image=[]
    label=[]
    if not os.path.exists("/home/hanlin/Desktop/Cat_VS_Dog/data.p"):
        for filename in glob.glob("/home/hanlin/Desktop/Cat_VS_Dog/cats/*.jpg"):
            #x=Image.open(filename).resize((image_size,image_size)).convert("L")
            x=cv2.resize(cv2.imread(filename,cv2.IMREAD_GRAYSCALE),(image_size,image_size))
            image.append(np.array(x)/255.0)
            label.append([1,0])

        for filename in glob.glob("/home/hanlin/Desktop/Cat_VS_Dog/dogs/*.jpg"):
            #x=Image.open(filename).resize((image_size,image_size)).convert("L")
            x=cv2.resize(cv2.imread(filename,cv2.IMREAD_GRAYSCALE),(image_size,image_size))
            image.append(np.array(x)/255.0)
            label.append([0,1])



        print("finish reading all image .......")
        image=np.array(image).reshape(-1,image_size,image_size,1)
        label=np.array(label).reshape(-1,2)

        train_image,test_image,train_label,test_label=train_test_split(image,label,test_size=0.1,shuffle=True, random_state=42)

        print("Training image shape:{}".format(train_image.shape))
        print("Testing image shape:{}".format(test_image.shape))
        with open("/home/hanlin/Desktop/Cat_VS_Dog/data.p","wb") as file:
            try:
                print("pickling.......")
                dataset={
                    'train_image':train_image,
                    'train_label':train_label,
                    'test_image':test_image,
                    'test_label':test_label

                }
                pickle.dump(dataset,file)
            except:

                print ("unable to pickling")
    
    with open("/home/hanlin/Desktop/Cat_VS_Dog/data.p","rb") as file:

        data=pickle.load(file)
        train_image=data['train_image']
        train_label=data['train_label']
        test_image=data['test_image']
        test_label=data['test_label']
    

    return train_image,train_label,test_image,test_label
    


    

