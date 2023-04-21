from utils import generate_segments_and_save, timer
from preprocessing import preprocess_dir




@timer
def chanting_predict():
    dir = './audio'
    model_dir = './model'
    model_type = 'svm' # support vector machine
    model_name = model_dir+"/svm_chanting_041_2"

    # generate the segmentation result on the testing data
    # preprocess_dir(dir+"/testing_canton/", dir+"/testing_canton/processed/", file_type='wav')
    generate_segments_and_save(model_name, model_type, 
                                soureFolder=dir+"/test/source", 
                                targetFolder=dir+"/test/result",
                                filetype='txt',filter=True)

if __name__ == "__main__":
    chanting_predict()
