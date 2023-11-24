from utils import generate_segments_and_save, timer, convert_dir_other_to_wav
from preprocessing import preprocess_dir
import os
import shutil
import glob


@timer
def chanting_predict():
    
    dir = './audio/test'
    model_dir = './model'
    model_type = 'svm' # support vector machine
    model_name = model_dir+"/svm_chanting_041_005005_f"


    # generate the segmentation result on the testing data
    generate_segments_and_save(model_name, model_type, 
                                soureFolder=dir+"/source", 
                                targetFolder=dir+"/result",
                                filetype='txt',filter=False)

if __name__ == "__main__":
    chanting_predict()
