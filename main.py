
from systeme import *

weight_fM1 = "WeightsFiles/model1_weightsEXP4.h5"
weight_fM2 = "WeightsFiles/model2_weights.h5"
#image_path = "G:/CS Extraction minuties/ImagesTests/101_2.png"
excel_file_name = "results"


codage_M1 = [[[256 , 3 , 0 , 1] , [128 , 7 , 0 , 1] , [256 , 9 , 3 , 1] , [256 , 9 , 0 , 1] ,[32 , 3 , 4 , 1] , [256 , 3 , 3 , 0] , [128 , 7 , 3 , 1] , [128 , 7 , 5 , 1]], [[512 , 1] , [512 , 1] , [512 , 0] , [512 , 1]]]
codage_M2 = [[[512, 9, 0, 1], [128, 11, 2, 1], [128, 11, 2, 1], [512, 5, 3, 1], [128, 7, 0, 1], [256, 5, 5, 1]], [[512, 1], [512, 0], [512, 0]]]

bloc_size = (45,45,1)

if __name__ == "__main__":

    for fingerprint in os.listdir("ImagesTests/"):
        print("Extraction de "+fingerprint)
        image_path = "ImagesTests/"+fingerprint
        image_with_minutaes,minutaes = MinutiaesExtractionV2(image_path,bloc_size,codage_M1,weight_fM1,
                                                codage_M2,weight_fM2)
        print("Nombre de minuties trouvées: ",len(minutaes.keys()))

        image_name = image_path.split("/")[-1].split(".")[0]
        #Minutaes excel file creation
        df = pd.DataFrame(minutaes)
        df = df.transpose()
        excel_file_path = "Minutaes/"+excel_file_name+"_"+image_name+'2.xlsx'
        df.to_excel(excel_file_path, index=False)


        #Mask save
        
        cv.imwrite("ImagesMarked/"+image_name+"_marked2.png",image_with_minutaes)


    









