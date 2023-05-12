import os
import shutil
import glob

source_folder = r'D:\pian\cassette'
destination_folder = r'C:\Users\dslab\Documents\Machine-Learning-Chanting\audio\test\source'
# print(os.path.join(source_folder,'*','*.mp3'))
for path in glob.glob(os.path.join(source_folder,'*','*.mp3')):
    # print(fileName)'
    filename = path.split('\\')[-1]
    # filename = filename.split('.')[0]
    # print(filename)
    source = path
    destination = os.path.join(destination_folder, filename)
    shutil.copy(source, destination)
    print('copied')

                          
                          