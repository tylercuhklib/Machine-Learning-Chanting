"""
Created by John Yeung (cfyeung2357@gmail.com)

# This file is for doing preprocessing on the audio data, including
- larger sound (unified across all samples)
- denoise audio

Modified by Tyler Wan (tylerwan@cuhk.edu.hk)
# Fixed:
- denoiseMP3
- preprocessingPipelineMP3
- preprocessingPipelineMP3toWav

citcations:

1. noisereduce
@software{tim_sainburg_2019_3243139,  
  author       = {Tim Sainburg},  
  title        = {timsainb/noisereduce: v1.0},  
  month        = jun,   
  year         = 2019,  
  publisher    = {Zenodo},  
  version      = {db94fe2},  
  doi          = {10.5281/zenodo.3243139},  
  url          = {https://doi.org/10.5281/zenodo.3243139}  
}
@article{sainburg2020finding,  
  title={Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires},  
  author={Sainburg, Tim and Thielk, Marvin and Gentner, Timothy Q},  
  journal={PLoS computational biology},  
  volume={16},  
  number={10},     
  pages={e1008228},   
  year={2020},    
  publisher={Public Library of Science}  
}

2. pyAudioAnalysis
@article{giannakopoulos2015pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}

"""


import glob
import os
from pyAudioAnalysis import audioBasicIO
import sys
import csv
import scipy.io.wavfile as wavfile

import re

from scipy.io import wavfile
from pydub import AudioSegment
from pydub import effects
import numpy as np
import noisereduce as nr


def annotation2folders(wavFile: str, csvFile: str, folderPath: str):
    """
        Break an audio stream to segments of interest, 
        defined by a csv file
        
        - wavFile:    path to input wavfile
        - csvFile:    path to csvFile of segment limits
        - folderPath: path to class folders
        
        Input CSV file must be of the format <T1>\t<T2>\t<Label>
    """
    print(wavFile)
    print(csvFile)
    [Fs, x] = audioBasicIO.read_audio_file(wavFile)
    with open(csvFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for j, row in enumerate(reader):
            T1 = float(row[0].replace(",","."))
            T2 = float(row[1].replace(",","."))
            row[2] = row[2].replace(' ', '_')
            label = os.path.join(folderPath, row[2], "%s_%.2f_%.2f.wav" % (re.split(r'[/\\]', wavFile)[-1], T1, T2))
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            if not os.path.exists(os.path.join(folderPath, row[2])):
                os.makedirs(os.path.join(folderPath, row[2]))
            label = label.replace(" ", "_")
            xtemp = x[int(round(T1*Fs)):int(round(T2*Fs))]            
            print(T1, T2, label, xtemp.shape)
            wavfile.write(label, Fs, xtemp)
    
def folderAnnotation2folders(sourceFolder, targetFolder):
    """
        Break an audio stream to segments of interest for all files in the sourceFolder
        
        - sourceFolder:    path to Folder of all source audio file and .segments file
        - targetFolder:    path to Folder where user want to store the class folders
    """
    print(sourceFolder)
    for fileName in glob.glob(os.path.join(sourceFolder, '*.segments')):
        print(fileName)
        fileName = fileName.split('.')[0]
        print(fileName)
        annotation2folders('%s.wav' % (fileName), '%s.segments' % (fileName), targetFolder)
        
        
# https://github.com/jiaaro/pydub/issues/90#issuecomment-75551606
def matchTargetAmplitude(sound, target_dBFS): # not used
    change_in_dBFS = target_dBFS - sound.max_dBFS
    return sound.apply_gain(change_in_dBFS)

def denoise(data: np.ndarray, samplingRate: int, propDecrease=0.9, freqMaskSmoothHz=120)->np.ndarray:
    """denoise the audio data

    Args:
        data (np.ndarray): input audio data
        samplingRate (int): audio sampling rate
        propDecrease (float, optional): _description_. Defaults to 0.9.
        freqMaskSmoothHz (int, optional): I think it is the max cap for frequency to be smoothed. Defaults to 100 to avoid canceling male voices.

    Returns:
        np.ndarray: denoised audio data
    """
    
    reduced_noise = nr.reduce_noise(y=data, sr=samplingRate, prop_decrease=propDecrease, freq_mask_smooth_hz=freqMaskSmoothHz)
    return reduced_noise

def denoiseMP3(data, samplingRate: int, propDecrease=0.8, freqMaskSmoothHz=120):
    # np_data = np.array(data.get_array_of_samples())
    # np_data = np_data[::data.channels]
    reduced_noise = nr.reduce_noise(y=data, sr=samplingRate, prop_decrease=propDecrease, freq_mask_smooth_hz=freqMaskSmoothHz)
    return reduced_noise
    


def setVolume(data: np.ndarray, frameRate: int, targetVolume: float = -14):
    """set the data's volume to the target Volume (in dB)

    Args:
        data (np array): input audio data array
        frameRate (int): sampling rate
        targetVolume (float): a volume in dB
    """
    # print(data.dtype.itemsize)
    sound = AudioSegment(data.tobytes(), frame_rate=frameRate, sample_width=data.dtype.itemsize, channels=1)
    # print(sound.max_dBFS)
    # print(sound.dBFS)
    # increase average volume
    print(sound.dBFS)
    if sound.dBFS < -12:
        normalized_sound = sound - sound.dBFS
        normalized_sound = normalized_sound + (targetVolume - normalized_sound.dBFS)
        # normalized_sound = targetVolume - sound.dBFS
    else:
        normalized_sound = sound
    return normalized_sound

def setVolumeMP3(sound: np.ndarray, frameRate: int, targetVolume: float = -14):
    """set the data's volume to the target Volume (in dB)

    Args:
        data (AudioSegment object): input audio data, load from pydub
        frameRate (int): sampling rate, no use, leave here for consistency
        targetVolume (float): a volume in dB
    """

    sound = AudioSegment(sound.tobytes(), frame_rate=frameRate, sample_width=2, channels=1)
    print(sound.dBFS)
    # print(sound.max_dBFS)
    normalized_sound = sound - sound.dBFS
    # print(normalized_sound.dBFS)
    normalized_sound = normalized_sound + (targetVolume - normalized_sound.dBFS)
    print(normalized_sound.dBFS)

    return normalized_sound

# def preprocessingPipeline(sourceFolder, targetFolder, fileName): # from wav file
#     """Process the data using wav source files, called by #?def preprocess_dir()

#     Args:
#         sourceFolder (str): The raw data file location.
#         targetFolder (str): The target class folder location where the processed file are stored.
#         fileName (str): name of the file to be processed
#     """       
#     if not os.path.exists(targetFolder):
#         os.makedirs(targetFolder)
#     rate, data = wavfile.read(os.path.join(sourceFolder, fileName))
#     # data = denoise(data, rate)
#     data = setVolume(data, rate)
#     print('done')
#     data.export(os.path.join(targetFolder, fileName), format="wav")

def preprocessingPipeline(sourceFolder, targetFolder, fileName): # from wav file
    """Process the data using wav source files, called by #?def preprocess_dir()

    Args:
        sourceFolder (str): The raw data file location.
        targetFolder (str): The target class folder location where the processed file are stored.
        fileName (str): name of the file to be processed
    """       
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    data = AudioSegment.from_file(os.path.join(sourceFolder, fileName), format="wav")
    rate = data.frame_rate
    np_data = np.array(data.get_array_of_samples())
    np_data = np_data[::data.channels]
    # np_data = denoiseMP3(np_data, rate)    
    byte_data = setVolumeMP3(np_data, rate)
    np_data = np.array(byte_data.get_array_of_samples())
    wavfile.write(os.path.join(targetFolder, fileName.split(".")[0]+"_14dbfs"+".wav"), rate, np_data)
    
def preprocessingPipelineMP3(sourceFolder: str, targetFolder: str, fileName: str): # from map file
    """Process the data using mp3 source files, called by #?def preprocess_dir()

    Args:
        sourceFolder (str): The raw data file location.
        targetFolder (str): The target class folder location where the processed file are stored.
        fileName (str): name of the file to be processed
    """    
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    data = AudioSegment.from_file(os.path.join(sourceFolder, fileName), format="mp3")
    # rate, data = wavfile.read(os.path.join(sourceFolder, fileName))\
    # print(type(data))

    rate = data.frame_rate    
    # print(rate)
    # print(len(data))
    np_data = np.array(data.get_array_of_samples())
    np_data = np_data[::data.channels]
    np_data = denoiseMP3(np_data, rate)    
    byte_data = setVolumeMP3(np_data, rate)
    np_data = np.array(byte_data.get_array_of_samples())
    wavfile.write(os.path.join(targetFolder, fileName.split(".")[0]+".wav"), rate, np_data)

def preprocessingPipelineMP3toWav(sourceFolder: str, targetFolder: str, fileName: str): # from map file
    """Convert mp3 source to files to wav, then process the data by preprocessingPipeline, called by #?def preprocess_dir()

    Args:
        sourceFolder (str): The raw data file location.
        targetFolder (str): The target class folder location where the processed file are stored.
        fileName (str): name of the file to be processed
    """    
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    data = AudioSegment.from_file(os.path.join(sourceFolder, fileName), format="mp3")
    rate = data.frame_rate
    audioBasicIO.convert_dir_mp3_to_wav(sourceFolder,rate,1)
    fileName = fileName.split(".")[0]+".wav"
    preprocessingPipeline(sourceFolder=sourceFolder, targetFolder=targetFolder, fileName=fileName)

    
def preprocess_dir(sourceFolder: str, targetFolder: str, file_type: str='mp3'):
    """apply the preprocessing pipeline on entire directory

    Args:
        sourceFolder (str): The raw data folder location.
        targetFolder (str): The target folder location where the processed class files are stored.
        file_type (str, optional): the file type of the source files. Defaults to 'mp3'.
    """    
    
    
    if file_type == 'wav':
        glob_folder = os.path.join(sourceFolder + '*.wav')
        for file in glob.glob(glob_folder):
            file = file.split('\\')[-1]
            preprocessingPipeline(sourceFolder=sourceFolder, targetFolder=targetFolder, fileName=file)
    elif file_type == 'mp3':    
        glob_folder = os.path.join(sourceFolder + '*.mp3')
        for file in glob.glob(glob_folder):
            file = file.split('\\')[-1]
            preprocessingPipelineMP3(sourceFolder=sourceFolder, targetFolder=targetFolder, fileName=file)
            # preprocessingPipelineMP3toWav(sourceFolder=sourceFolder, targetFolder=targetFolder, fileName=file)

def main(sourceFolder, processedFolder, targetFolder, filetype, segment= False):
    # preprocess the training data
    # print(sourceFolder)
    # preprocess_dir(sourceFolder, processedFolder, filetype)
    # cut the source files in a folder according to their segments file
    
    if segment:
        folderAnnotation2folders(processedFolder, targetFolder) 

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4], sys.argv[5])

 