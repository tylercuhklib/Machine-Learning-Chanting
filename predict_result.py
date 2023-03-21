from utils import generate_segments_and_save
from preprocessing import preprocess_dir
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

@timer
def chanting_predict():
    dir = './audio'
    model_dir = './model'
    model_type = 'svm' # support vector machine
    model_name = model_dir+"/svm_chanting_041"

    # generate the segmentation result on the testing data
    # preprocess_dir(dir+"/testing_canton/", dir+"/testing_canton/processed/", file_type='wav')
    generate_segments_and_save(model_name, model_type, 
                                soureFolder=dir+"/test", 
                                targetFolder=dir+"/test",
                                filetype='txt',filter=True)

if __name__ == "__main__":
    chanting_predict()
