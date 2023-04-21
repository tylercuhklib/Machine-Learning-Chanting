from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioTrainTest as aT
# from utils import generate_segments_and_save2xlsx
from preprocessing import folderAnnotation2folders
from preprocessing import preprocess_dir

dir = './audio'
model_dir = './model'
# defind the model and train
model_type = 'svm' # support vector machine
model_name = model_dir+"/svm_chanting_041_2" #mid-widow = 4, mid step = 1

aT.extract_features_and_train([dir+r'\training\Chanting', 
                              dir+r'\training\Speech',
                              dir+r'\training\Silence'],
                              4, 1,
                              aT.shortTermWindow, 
                              aT.shortTermStep, model_type, model_name)

