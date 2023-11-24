from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioTrainTest as aT
# from utils import generate_segments_and_save2xlsx
from preprocessing import folderAnnotation2folders
from preprocessing import preprocess_dir

dir = './training_audio'
model_dir = './model'
# defind the model and train
model_type = 'svm' # support vector machine
model_name = model_dir+"/svm_chanting_041_005005_f2_fine" #mid-widow = 4, mid step = 1

# aT.extract_features_and_train([dir+r'\training_14dbfs\Chanting', 
#                               dir+r'\training_14dbfs\Speech',
#                               dir+r'\training_14dbfs\Silence'],
#                               2, 1,
#                               aT.shortTermWindow, 
#                               aT.shortTermStep, model_type, model_name)

aT.extract_features_and_train([dir+r'\training_mix\Chanting', 
                              dir+r'\training_mix\Speech',
                              dir+r'\training_mix\Silence'],
                              4, 1,
                              aT.shortTermWindow, 
                              aT.shortTermStep, model_type, model_name)