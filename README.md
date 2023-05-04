This is based on the project ["Archive of 20th Century Cantonese Chanting in Hong Kong"「二十世紀香港粵語吟誦典藏」](https://dsprojects.lib.cuhk.edu.hk/en/projects/20th-cantonese-poetry-chanting/home/)

# Classification of Cantonese Chanting by Machine Learning.
Cantonese Chanting is a traditional Chinese scholarly practice of reading, composing, and teaching classical poetry and prose in a specific melodic style. It is characterized by its melodic nature and room for improvisation, with various schools formed by dialects, lineages, and personal preferences. The art of chanting is gradually fading, as the elderly who once learned in private schools pass away, making the preservation of this artform urgent.

Since many recordings are extracted from class sessions, with chanting intermingled with teaching. Researcher have to find out the chanting activities within these hours long recordings. The works could be time consuming. 

By using the python based module [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis.git), it is possible to identify speech and chanting by the machine learning method. This tool as a quick filter to identify if the chanting exists and to label the period in the audio.

As one of the default output file is txt, which include the labels of "start time", "end time" and "class". This is recommended to use [Audacity](https://www.audacityteam.org/) to import the txt for further audio editing.

## Quickstart
 * Clone the source of this library: `git clone https://github.com/tylercuhklib/Machine-Learning-Chanting.git`
 * Copy your test audio(in wav format) into folder "audio/test/source"
 * Open terminal and cd to the project's file path, e.g.
 ```
 cd C:\Users\Users\Documents\Machine-Learning-Chanting
 ```
 * Use the trained SVM model in folder "model" to predict the result. Run the following in terminal:
 ```python
 pip install -r ./requirements.txt 
 python predict_result.py
 ```
 * The result .txt file will be saved in the folder "audio/test/result". It contains the start time, end time and label(show "Chanting" only for convenience because we only want to extract the chanting section from the source file). 
 * Open Audacity and then import the audio and the label file for further editing. We need to adjust the actual start/end time for 
 * Export the edited label file as .segments file and save it to the source wav file. It will be used to extract the chanting section from the original file.
 * To extract chanting from the source audio:

 ```
 python extract_chanting.py source_folder target_folder
 #example
 python extract_chanting.py audio/test/source audio/result_final
 ```
 * All chanting audio will be in the folder /audio/result_final

## Machine Learning
### Data Collection
 * The data are mainly from 「[二十世紀香港粵語吟誦典藏](https://dsprojects.lib.cuhk.edu.hk/en/projects/20th-cantonese-poetry-chanting/home/)」. Save the urls to urls.txt
 * To download the mp3 file:`python getaudio.py`

### Data Preprocessing
 * All .mp3 file have to be convected to .wav file. 
 * Denoised and normalized volume.
 * Audios file are segmented and divided into three classes: Chanting, Speech, Silence.

### Training of the model
 * With the function of pyAudioAnalysis, the segment-based audios are used for feature extraction
   and classifier training
 * SVM method are used as its preformeance are better in our case.
 * The function also include the hyperparameter tuning and evaluation.
 ```
 python train_the_model.py
 ```
For our trained model, 
Confusion Matrix:
|          | Chanting | Speaking | Silence |
| Chanting | 35.39    | 2.39     | 0.00    |
| Speaking | 2.67     | 47.29    | 1.16    |
| Silence  | 0.16     | 0.46     | 10.49   |
(Cha for Chanting, Spe for Speaking, Sil for Silence)

Best macro f1: 92.9
Best macro f1: std 3.9
Selected params: 0.10000

The Classifier with parameter C = 0.1 has the best f1 score of 92.9%

## About pyAudioAnalysis
pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks. Through pyAudioAnalysis you can:
 * Extract audio *features* and representations (e.g. mfccs, spectrogram, chromagram)
 * *Train*, parameter tune and *evaluate* classifiers of audio segments
 * *Classify* unknown sounds
 * *Detect* audio events and exclude silence periods from long recordings
 * Perform *supervised segmentation* (joint segmentation - classification)
 * Perform *unsupervised segmentation* (e.g. speaker diarization) and extract audio *thumbnails*
 * Train and use *audio regression* models (example application: emotion recognition)
 * Apply dimensionality reduction to *visualize* audio data and content similarities

*This is general info. Click [here](https://github.com/tyiannak/pyAudioAnalysis/wiki) for the complete wiki and [here](https://hackernoon.com/audio-handling-basics-how-to-process-audio-files-using-python-cli-jo283u3y) for a more generic intro to audio data handling*

## Further reading

Apart from this README file, to bettern understand how to use this library one should read the following:
  * [Audio Handling Basics: Process Audio Files In Command-Line or Python](https://hackernoon.com/audio-handling-basics-how-to-process-audio-files-using-python-cli-jo283u3y), if you want to learn how to handle audio files from command line, and some basic programming on audio signal processing. Start with that if you don't know anything about audio. 
  * [Intro to Audio Analysis: Recognizing Sounds Using Machine Learning](https://hackernoon.com/intro-to-audio-analysis-recognizing-sounds-using-machine-learning-qy2r3ufl) This goes a bit deeper than the previous article, by providing a complete intro to theory and practice of audio feature extraction, classification and segmentation (includes many Python examples).
 * [The library's wiki](https://github.com/tyiannak/pyAudioAnalysis/wiki)
 * [How to Use Machine Learning to Color Your Lighting Based on Music Mood](https://hackernoon.com/how-to-use-machine-learning-to-color-your-lighting-based-on-music-mood-bi163u8l). An interesting use-case of using this lib to train a real-time music mood estimator.
  * A more general and theoretic description of the adopted methods (along with several experiments on particular use-cases) is presented [in this publication](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610). *Please use the following citation when citing pyAudioAnalysis in your research work*:
```python
@article{giannakopoulos2015pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}
```

For Matlab-related audio analysis material check  [this book](http://www.amazon.com/Introduction-Audio-Analysis-MATLAB%C2%AE-Approach/dp/0080993885).


