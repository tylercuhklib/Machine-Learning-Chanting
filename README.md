*This is general info. Click [here](https://github.com/tyiannak/pyAudioAnalysis/wiki) for the complete wiki and [here](https://hackernoon.com/audio-handling-basics-how-to-process-audio-files-using-python-cli-jo283u3y) for a more generic intro to audio data handling*

# Classification of Cantonese Chanting by Machine Learning.
This project is based on the project「[二十世紀香港粵語吟誦典藏](https://dsprojects.lib.cuhk.edu.hk/en/projects/20th-cantonese-poetry-chanting/home/)」 

The goal of this project is to develop a quick filter tool to identify the chanting activities out from a long time recording.
By using the module pyAudioAnalysis(https://github.com/tyiannak/pyAudioAnalysis.git), it is possible to identify speech and chanting by the machine learning method.

## Data Collection
 * The data are mainly from 「[二十世紀香港粵語吟誦典藏](https://dsprojects.lib.cuhk.edu.hk/en/projects/20th-cantonese-poetry-chanting/home/)」.  
 * To download the mp3 file:
 use getaudio.py and urls.txt

## Data Preprocessing
 * All mp3 convected to wav
 * Sound level to -14 dbFS
 * Denoise is required for better quailty

## Quickstart
 * Clone the source of this library: `git clone https://github.com/tylercuhklib/Machine-Learning-Chanting.git`
 * Open terminal and cd to the project's file path, e.g.
 ```
 cd C:\Users\Users\Documents\Machine-Learning-Chanting
 ```
 * Run the following in terminal:
 ```python
 pip install -r ./requirements.txt 
 python predict_result.py
 ```


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

## Author
<img src="https://tyiannak.github.io/files/3.JPG" align="left" height="100"/>

[Theodoros Giannakopoulos](https://tyiannak.github.io),
Principal Researcher of Multimodal Machine Learning at the [Multimedia Analysis Group of the Computational Intelligence Lab (MagCIL)](https://labs-repos.iit.demokritos.gr/MagCIL/index.html) of the Institute of Informatics and Telecommunications, of the National Center for Scientific Research "Demokritos"

