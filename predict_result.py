from utils import generate_segments_and_save, timer, convert_dir_other_to_wav
from preprocessing import preprocess_dir
import os
import shutil
import glob


@timer
def chanting_predict():
    
    dir = './audio'
    model_dir = './model'
    model_type = 'svm' # support vector machine
    model_name = model_dir+"/svm_chanting_041_enhance2"
    path = dir+"/test/source"

    path = r'D:\pian\cassette'
    targetfolder = r'D:\pian\all_mp3'
    # convert_dir_other_to_wav(path,targetfolder)


    # generate the segmentation result on the testing data
    # preprocess_dir(dir+"/testing_canton/", dir+"/testing_canton/processed/", file_type='wav')
    generate_segments_and_save(model_name, model_type, 
                                soureFolder=dir+"/test/source", 
                                targetFolder=dir+"/test/result",
                                filetype='txt',filter=True)

if __name__ == "__main__":
    chanting_predict()
