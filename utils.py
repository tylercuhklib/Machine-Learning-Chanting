"""
Created by John Yeung (cfyeung2357@gmail.com)
Edited by Tyler Wan (tylerwan@cuhk.edu.hk)

utils file for generating segmentation results using a pretrain model, and save the results to a targetFolder.

"""

from pickle import TRUE
from pyAudioAnalysis import audioSegmentation as aS
import os
from pydub import AudioSegment
import re
import glob
from typing import List
import xlsxwriter
import csv
import functools
import time

import wave
import contextlib

def get_wave_duration(wavFile: str):
    
    with contextlib.closing(wave.open(wavFile,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        # print(duration)
    return duration

def convert_dir_other_to_wav(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
        
    for filename in glob.glob(os.path.join(source_folder,'*','*.mp3')):
    # for filename in os.listdir(audio_folder):
        ext = filename.split('.')[-1]
        if ext != 'wav':
            
            output_filename = filename.split('\\')[-1]
            output_filename = output_filename.replace('.mp3','.wav')
            print(filename)
            print(output_filename)
            os.system("ffmpeg -loglevel level+warning -i {0} -vn {1}".format(filename, os.path.join(target_folder,output_filename)))
        else:
            continue
    print("Convected all non-WAV file to WAV")

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

def generate_segments_and_save(modelName: str, modelType: str, soureFolder: str, targetFolder: str, filetype: str,filter: bool=False):
    """generate segments for all files in the source folder, and save to the target folder

    Args:
        modelName (str): the model name to be used in generating segmentation
        modelType (str): type of the model (svm, NN, etc.)
        soureFolder (str): name of the folder containing the audio files to be segmented
        targetFolder (str): location of the file to store the generated segments
        filetype: save result as 'wav','xlsx' or 'txt'
    """ 
    stat = []   
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    for filename in glob.glob(soureFolder + '/*.wav'):
        
        file_name = re.split('[\\\\/.]', filename)[-2]
        # check_txt_exist = file_name+'.txt'
        # txt_path = os.path.join(targetFolder,check_txt_exist)
        # if not os.path.exists(txt_path):

        gt_filename= soureFolder+ '\\' + file_name+".segments"
        #     # print(file_name)
        
        if not os.path.exists(os.path.join(targetFolder, file_name)): # skip if the file is already processed
            flagsInd, classesAll, acc, CM, segs, classes = aS.mid_term_file_classification_chanting(filename, 
                                                                                           modelName, 
                                                                                           modelType, plot_results=True, gt_file=gt_filename)
            if filetype == 'wav':
                _segments2wav(filename, segs, classes, classesAll, targetFolder)
            elif filetype == 'txt':
                prob = _segments2txt(filename, segs, classes, classesAll, targetFolder, filter)
                _segments2xlsx(filename, segs, classes, classesAll, targetFolder)
            # aS.plot_segmentation_results = (flagsInd,flagsInd,classesAll,2)
            
    #         stat.append([file_name, prob])
    # probability_stat(stat)

def _segments2wav(wavFile: str, segments: List[List[float]], classes: List[float], classesAll: List[str], folderPath: str):
    """save the generated segments into corresponding audio wav files, with 2 seconds append at start/end

    Args:
        wavFile (str): the source wav file where the audio segments are generated
        segments (List[List[float]]): a list of segments time stamps
        classes (List[float]): a list of classified class ids corresponding to the segments
        classesAll (List[str]): name of the classes corresponding to the ids, i.e. classesAll[classes[index]] -> classified class name
        folderPath (str): file path where the audio segments are stored, with the name 'folderPath/wavFile_name/className_start_end.wav'
    """    
    time_append = 0 # the seconds to be append at head of tail in the audio slice
    file_name = re.split('[\\\\/.]', wavFile)[-2] # get file name from the full path
    # \\\\ become \\ inside re.split, and then \
    # print(classesAll)
    if not os.path.exists(os.path.join(folderPath, file_name)): # skip if the file is already processed
        os.makedirs(os.path.join(folderPath, file_name))
        audiofile = AudioSegment.from_file(wavFile)
        for i, (seg, cls) in enumerate(zip(segments, classes)):
            # include starting and ending 5 seconds
            start_seconds = seg[0] - time_append if seg[0] - time_append > 0                          else 0
            end_seconds   = seg[1] + time_append if seg[1] + time_append < audiofile.duration_seconds else audiofile.duration_seconds
            
            classname = classesAll[int(cls)] #to get exact classname(Tyler)
            classname = re.split('\\/',classname)[-1]

            # the path and filename to be saved
            # label = os.path.join(folderPath, file_name, "%s_%d.%d-%d.%d.wav" % (classesAll[int(cls)], int(start_seconds//60), int(start_seconds%60), int(end_seconds//60), int(end_seconds%60)))
            label = os.path.join(folderPath, file_name, "%s_%d.%d-%d.%d.wav" % (classname, int(start_seconds//60), int(start_seconds%60), int(end_seconds//60), int(end_seconds%60)))
            # print(label)
            audio_slice = audiofile[start_seconds*1000:end_seconds*1000]    # get the audio slice
            audio_slice.export(label, format="wav")

def _segments2mp3(wavFile: str, segments: List[List[float]], classes: List[float], classesAll: List[str], folderPath: str):
    """save the generated segments into corresponding audio mp3 files, with 2 seconds append at start/end

    Args:
        wavFile (str): the source wav file where the audio segments are generated
        segments (List[List[float]]): a list of segments time stamps
        classes (List[float]): a list of classified class ids corresponding to the segments
        classesAll (List[str]): name of the classes corresponding to the ids, i.e. classesAll[classes[index]] -> classified class name
        folderPath (str): file path where the audio segments are stored, with the name 'folderPath/wavFile_name/className_start_end.mp3'
    """    
    time_append = 0 # the seconds to be append at head of tail in the audio slice
    file_name = re.split('[\\\\/.]', wavFile)[-2] # get file name from the full path
    # \\\\ become \\ inside re.split, and then \
    # print(classesAll)
    if not os.path.exists(os.path.join(folderPath, file_name)): # skip if the file is already processed
        os.makedirs(os.path.join(folderPath, file_name))
        audiofile = AudioSegment.from_file(wavFile)
        for i, (seg, cls) in enumerate(zip(segments, classes)):
            # include starting and ending 5 seconds
            start_seconds = seg[0] - time_append if seg[0] - time_append > 0                          else 0
            end_seconds   = seg[1] + time_append if seg[1] + time_append < audiofile.duration_seconds else audiofile.duration_seconds
            
            classname = classesAll[int(cls)] #to get exact classname(Tyler)
            classname = re.split('\\/',classname)[-1]

            # the path and filename to be saved
            # label = os.path.join(folderPath, file_name, "%s_%d.%d-%d.%d.mp3" % (classesAll[int(cls)], int(start_seconds//60), int(start_seconds%60), int(end_seconds//60), int(end_seconds%60)))
            label = os.path.join(folderPath, file_name, "%s_%d.%d-%d.%d.mp3" % (classname, int(start_seconds//60), int(start_seconds%60), int(end_seconds//60), int(end_seconds%60)))
            # print(label)
            audio_slice = audiofile[start_seconds*1000:end_seconds*1000]    # get the audio slice
            audio_slice.export(label, format="mp3")

def _segments2xlsx(wavFile: str, segments: List[List[float]], classes: List[float], classesAll: List[str], folderPath: str):
    """save the generated segments into excel file
    Args:
        wavFile (str): the source wav file where the audio segments are generated
        segments (List[List[float]]): a list of segments time stamps
        classes (List[float]): a list of classified class ids corresponding to the segments
        classesAll (List[str]): name of the classes corresponding to the ids, i.e. classesAll[classes[index]] -> classified class name
        folderPath (str): file path where the audio segments are stored, with the name 'folderPath/wavFile_name/className_start_end.mp3'
    """    
    file_name = re.split('[\\\\/.]', wavFile)[-2] # get file name from the full path
    workbook = xlsxwriter.Workbook(folderPath+'/%s.xlsx'%(file_name))

    worksheet = workbook.add_worksheet()
    row = 0
    if not os.path.exists(os.path.join(folderPath, file_name)): # skip if the file is already processed
        # os.makedirs(os.path.join(folderPath, file_name))
        # audiofile = AudioSegment.from_file(wavFile)
        for i, (seg, cls) in enumerate(zip(segments, classes)):
            # starting and ending time
            # print(seg)
            start_seconds = '%02d:%02d:%02d'%(seg[0]//3600,seg[0]%3600//60,seg[0]%60)                       
            end_seconds   = '%02d:%02d:%02d'%(seg[1]//3600,seg[1]%3600//60,seg[1]%60)
            
            classname = classesAll[int(cls)] #to get exact classname(Tyler)
            classname = re.split('\\/',classname)[-1]
            worksheet.write(row,0,start_seconds)
            worksheet.write(row,1,end_seconds)
            worksheet.write(row,2,classname)
            row += 1

    workbook.close()    

def _segments2txt(wavFile: str, segments: List[List[float]], classes: List[float], classesAll: List[str], folderPath: str, filter: bool = False):
    """save the generated segments into text file
    Args:
        wavFile (str): the source wav file where the audio segments are generated
        segments (List[List[float]]): a list of segments time stamps
        classes (List[float]): a list of classified class ids corresponding to the segments
        classesAll (List[str]): name of the classes corresponding to the ids, i.e. classesAll[classes[index]] -> classified class name
        folderPath (str): file path where the audio segments are stored, with the name 'folderPath/wavFile_name/className_start_end.mp3'
    """    
    file_name = re.split('[\\\\/.]', wavFile)[-2] # get file name from the full path
    classname_result = []
    start_seconds = []
    end_seconds = []
    # print(segments)
    chanting_time = 0
    total_time = 0
    total_time = segments[-1][1]
    if filter == False:
        for i, (seg,cls) in enumerate(zip(segments, classes)):
        # starting and ending time
        # print(seg)
        # start_seconds = '%02d:%02d:%02d'%(seg[0]//3600,seg[0]%3600//60,seg[0]%60)                       
        # end_seconds   = '%02d:%02d:%02d'%(seg[1]//3600,seg[1]%3600//60,seg[1]%60)

            classname = classesAll[int(cls)] #to get exact classname(Tyler)
            classname = re.split('\\/',classname)[-1]

            classname_result.append(classname)
            start_seconds.append(seg[0])
            end_seconds.append(seg[1])

    if filter == True:

        for i, (seg,cls) in enumerate(zip(segments, classes)):
            duration = seg[1]-seg[0]
            classname = classesAll[int(cls)] #to get exact classname(Tyler)
            classname = re.split('\\/',classname)[-1]
            # if classname == 'Chanting':
            if classname == "Chanting" and duration >=8:
                # total_time += duration
                classname_result.append(classname)
                start_seconds.append(seg[0])
                end_seconds.append(seg[1])
                chanting_time += duration
    if total_time > 0:
        prob_chanting = int(100*chanting_time / total_time)
    else:
        prob_chanting = int(0)

    prob_chanting = str(prob_chanting)
    data = zip(start_seconds,end_seconds,classname_result)
    with open(folderPath+'/'+file_name+'.txt', 'w',newline='') as file:
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')
        for row in data:
            writer.writerow(row)
    return prob_chanting

def probability_stat(stat):
    # print(stat)
    # workbook_path = os.path.join(r'D:\pian\result', 'prob.xlsx')
    workbook_path = os.path.join(r'C:\Users\dslab\Documents\Machine-Learning-Chanting\audio\test\result', 'prob.xlsx')
    workbook = xlsxwriter.Workbook(workbook_path)
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'Filename')
    worksheet.write('B1', 'Prob(Chanting)')
    row = 1
    col = 0
    for filename, prob in stat:
        worksheet.write(row, col, filename)
        worksheet.write(row, col+1, prob)
        row += 1
    workbook.close()


    
