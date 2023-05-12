from utils import generate_segments_and_save, timer, convert_dir_other_to_wav
from preprocessing import preprocess_dir
import os
import shutil
import glob
from pyAudioAnalysis import audioTrainTest as aT

@timer
def chanting_predict():
    
    dir = r'D:\pian\all_mp3\1'
    model_dir = './model'
    model_type = 'svm' # support vector machine
    model_name = model_dir+"/svm_chanting_041_enhance2"
    # path = dir+"/test/source"

    # path = r'D:\pian\cassette'
    # targetfolder = r'D:\pian\all_mp3'
    # convert_dir_other_to_wav(path,targetfolder)


    # generate the segmentation result on the testing data
    # preprocess_dir(dir+"/testing_canton/", dir+"/testing_canton/processed/", file_type='wav')
    generate_segments_and_save(model_name, model_type, 
                                soureFolder=dir, 
                                targetFolder="./audio/test/result",
                                filetype='txt',filter=True)

@timer
def instrument_file_classification():
    
    dir = r'D:\pian\all_audio'
    # dir = './audio/test/source'
    model_dir = './model'
    model_type = 'svm' # support vector machine
    model_name = model_dir+"/svm_instr_041"
    file = glob.glob(os.path.join(dir, '*.wav'))
    
    # file = os.listdir(os.path.join(dir,'*wav'))
    # print(file[0])
    # print(file)
    #File Classification and move to class folder

    # src = r'C:\Users\dslab\Documents\Machine-Learning-Chanting\audio\test\source'
    music_dir = os.path.join(dir, 'music')
    others_dir = os.path.join(dir,'others')
    if not os.path.exists(music_dir):
            os.makedirs(music_dir)
    if not os.path.exists(others_dir):
            os.makedirs(others_dir) 
    for f in file:
        filename = f.split('\\')[-1]
        # print(filename)
        class_id, probability, classes = aT.file_classification(f, model_name, model_type)
        # print(f)
        # print(f'P({classes[0]})={probability[0]}')
  
                 
        if probability[0] >= probability[1]:
            shutil.move(f, os.path.join(music_dir, filename))
            print('Moved {} to Music folder'.format(filename))
        else:
            shutil.move(f, os.path.join(others_dir, filename))
            print('Moved {} to Others folder'.format(filename))


if __name__ == "__main__":
    # chanting_predict()
    instrument_file_classification()
