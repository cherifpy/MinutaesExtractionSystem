import cv2 as cv
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import copy
from models import *

def DecoupageEnBlocs(image_path:str,bloc_size:int=45):


    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    image_height, image_width= image.shape


    width_to_add = int(27 - ((image_width-18) % 27))
    height_to_add = int(27 - ((image_height-18) % 27))
    new_width = image_width + width_to_add
    new_height = image_height + height_to_add
    new_image = np.ones((new_height, new_width), dtype=image.dtype) * 255
    new_image[0:image_height, 0:image_width] = image

    #cv.imshow("h",new_image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
    num_blocks_y = (new_image.shape[0] - 18) // 27
    num_blocks_x = (new_image.shape[1] - 18) // 27

    new_image = new_image/255
    coord_blocs = []
    blocks = []

    for y in range(9,new_height-9,27):
        for x in range(9,new_width-9,27):
            
            bloc = new_image[y-9:y+36, x-9:x+36]

            coord_blocs.append((y-9,x-9)) 
            
            blocks.append(bloc)
            
    return blocks, coord_blocs, new_image, width_to_add, height_to_add


def DecoupageEnBlocsV2(image_path:str,bloc_size:int=45):


    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    image_height, image_width= image.shape


    width_to_add = int(27 - ((image_width-18) % 27))
    height_to_add = int(27 - ((image_height-18) % 27))
    new_width = image_width + width_to_add
    new_height = image_height + height_to_add
    new_image = np.ones((new_height, new_width), dtype=image.dtype) * 255
    new_image[0:image_height, 0:image_width] = image

    #cv.imshow("h",new_image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
    num_blocks_y = (new_image.shape[0] - 18) // 27
    num_blocks_x = (new_image.shape[1] - 18) // 27

    new_image = new_image/255
    coord_blocs = []
    blocks = []

    for y in range(9,new_height-9,27):
        for x in range(9,new_width-9,27):
            
            block = new_image[y-9:y+36, x-9:x+36]

            coord_blocs.append((y-9,x-9)) 
            
        blocks.append(block)
            
    return blocks, coord_blocs, new_image, width_to_add, height_to_add

def ImageReconstruction(mask_list, num_blocks_y,num_blocks_x):
    
    #print(mask_list)
    bloc_size = mask_list[0].shape[0]
    h = num_blocks_y*27+18
    w = num_blocks_x*27+18
    image = np.ones((h,w,1),dtype=np.uint8) * 255

    count = 0

    for j in range(num_blocks_y):
        for i in range(num_blocks_x):
            if i == 0 and j == 0 :
                image[0:45-9,0:45] = mask_list[count][0:45-9,0:45]
            
            if i == 0 and j == num_blocks_y :
                image[0:45-9,0:45] = mask_list[count][0:45-9,0:45]
            
            if i == num_blocks_x and j == num_blocks_y :
                image[h-45-9:h,w-45-9:w] = mask_list[count][9:45,9:45]

            
            count+=1

    return image

def LoadModels(input_shape,codageM1, weight_fM1,codageM2, weight_fM2):


    if os.path.exists(weight_fM1) and os.path.exists(weight_fM2):
        model1 = CreateFirstModel(input_shape,individual=codageM1)
        model1.load_weights(weight_fM1)

        model2 = CreateSecondModel(input_shape,individual=codageM2)
        model2.load_weights(weight_fM2)

        return model1,model2

    else:
        raise Exception("Weight file do not exist")
    
    return None,None
    
def BlocsClassification(model, bloc):

    prediction = model.predict(bloc,verbose=0)
    softmax_predictions = tf.nn.softmax(prediction)
    predicted_labels = tf.argmax(softmax_predictions, axis=1)


    if int(predicted_labels[0]) == 0:
        return True, bloc
    else:
        return False, np.zeros(bloc[0].shape,dtype=np.uint8)
    

def MinutaeZoneDetection(model,bloc, image, coord_bloc,minutaes):

    pred_vector = model.predict(bloc,verbose=0)
    softmax_predictions = tf.nn.softmax(pred_vector)
    prediction_labels = np.argmax(softmax_predictions)

    prediction = int(prediction_labels)


    bloc_mask = np.zeros((45,45,1), dtype=np.uint8)

    i = int(prediction // 3)
    j = int(prediction % 3)
    
    
    bloc_mask[9*(i+1):9*(i+2),9*(j+1):9*(j+2)] = 255
    Y = coord_bloc[1]+9*(j+1)+5
    X = coord_bloc[0]+9*(i+1)+5
    image = cv.circle(image, (Y,X), 4,(255,0,0),1)
    
    minutaes[len(minutaes.keys())+1] = {"Num":len(minutaes.keys())+1,"X":X,"Y":Y}

    return bloc_mask, copy.deepcopy(image), minutaes
    
def MinutaeZoneDetectionV2(model,bloc, coord_bloc):

    pred_vector = model.predict(bloc,verbose=0)
    softmax_predictions = tf.nn.softmax(pred_vector)
    prediction_labels = np.argmax(softmax_predictions)

    prediction = int(prediction_labels)


    bloc_mask = np.zeros((45,45,1), dtype=np.uint8)

    i = int(prediction // 3)
    j = int(prediction % 3)
    
    
    bloc_mask[9*(i+1):9*(i+2),9*(j+1):9*(j+2)] = 255
    Y = coord_bloc[1]+9*(j+1)+5
    X = coord_bloc[0]+9*(i+1)+5

    return Y, X

def MinutaesExtraction(image_path, bloc_size, codageM1, weight_fM1, codageM2, weight_fM2):

    blocs_list,coords_blocs, new_image, width_to_add, height_to_add = DecoupageEnBlocs(image_path,bloc_size)
    image = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    
    model1, model2 = LoadModels(bloc_size,codageM1,weight_fM1, codageM2, weight_fM2)

    blocs_mask = []
    minutaes = {}
    count = 0
    for b_num, bloc in enumerate(blocs_list):
        
        reshaped_bloc = tf.reshape(bloc, shape=(1, 45,45,1))
        exist, bloc_classified = BlocsClassification(model1,reshaped_bloc)
        
        if exist:
            #model,bloc, image, coord_bloc
            black_mask, new_image, minutaes = MinutaeZoneDetection(model2,reshaped_bloc,new_image,coords_blocs[b_num], minutaes)
            
            blocs_mask.append(copy.deepcopy(black_mask))
            
            count+=1

        else:
            blocs_mask.append(copy.deepcopy(bloc_classified))
    
    return new_image*255,minutaes


def GenerateNighbords(num_bloc:int,coords_blocs,image,bloc_size=45):
    """
        Cette methode de decoupage construit des bloc avec chavauchement
    """

    y, x = coords_blocs[num_bloc] 

    blocs = [
        [image[y-9:y-9+bloc_size,x-9:x-9+bloc_size],image[y-9:y-9+bloc_size,x:x+bloc_size],image[y-9:y-9+bloc_size,x+9:x+9+bloc_size]],
        [image[y  :y+bloc_size  ,x-9:x-9+bloc_size],image[y  :y+bloc_size  ,x:x+bloc_size],image[y  :y+bloc_size  ,x+9:x+9+bloc_size]],
        [image[y+9:y+9+bloc_size,x-9:x-9+bloc_size],image[y+9:y+9+bloc_size,x:x+bloc_size],image[y+9:y+9+bloc_size,x+9:x+9+bloc_size]],
    ]

    neighbors_blocs_coords = [
        [(y-9,x-9),(y-9,x),(y-9,x+9)],
        [(y,  x-9),(y,  x),(y,  x+9)],
        [(y+9,x-9),(y+9,x),(y-9,x+9)]
    ]
    
    return blocs,neighbors_blocs_coords

def MinutiaesExtractionV2(image_path, bloc_size, codageM1, weight_fM1, codageM2, weight_fM2):
    blocs_list,coords_blocs, new_image, _, _ = DecoupageEnBlocs(image_path,bloc_size)
    image = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    
    model1, model2 = LoadModels(bloc_size,codageM1,weight_fM1, codageM2, weight_fM2)

    minutaes = {}
    count = 0

    for b_num, bloc in enumerate(blocs_list):
        
        reshaped_bloc = tf.reshape(bloc, shape=(1, 45,45,1))
        exist, bloc_classified = BlocsClassification(model1,reshaped_bloc)
        
        if exist:
            #model,bloc, image, coord_bloc
            #Here i have to generate 9 images //num_bloc:int,coords_blocs,image,bloc_size
            neighbors_blocs,neighbors_blocs_coords = GenerateNighbords(b_num,coords_blocs,image)
            minu_cords = []
            for i in range(len(neighbors_blocs)):
                for j in range(len(neighbors_blocs[i])):
                    reshaped_bloc = tf.reshape(neighbors_blocs[i][j], shape=(1, 45,45,1))
                    exist_2, bloc_classified_2 = BlocsClassification(model1,reshaped_bloc)

                    if exist_2:                                             #model,bloc, image, coord_bloc
                        Y, X = MinutaeZoneDetectionV2(model2,reshaped_bloc,neighbors_blocs_coords[i][j])
                        
                        minu_cords.append((Y,X))

            if len(minu_cords)!= 0:
                mean_y = sum(y for y, x in minu_cords) / len(minu_cords)
                mean_x = sum(x for y, x in minu_cords) / len(minu_cords)

                # Create a new tuple with the mean values
                mean_tuple = (mean_y, mean_x)

                new_image = cv.circle(new_image, (Y,X), 4,(255,0,0),1)

                minutaes[len(minutaes.keys())+1] = {"Num":len(minutaes.keys())+1,"X":X,"Y":Y}
    
                count+=1

    
    return new_image*255,minutaes     


