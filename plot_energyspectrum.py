from pyAudioAnalysis import MidTermFeatures as aF
import os
import numpy as np
import plotly.graph_objs as go 
import plotly

chanting = r'C:\Users\dslab\Documents\Machine-Learning-Chanting\training_audio\training_mix\Chanting'
speech = r'C:\Users\dslab\Documents\Machine-Learning-Chanting\training_audio\training_mix\Speech'
# chanting = r'C:\Users\dslab\Documents\Machine-Learning-Chanting\training_audio\training\Chanting'
# speech = r'C:\Users\dslab\Documents\Machine-Learning-Chanting\training_audio\training\Speech'
# cantonsp = r'C:\Users\dslab\Documents\JohnYeung\lecture-chanting-segmentation\Tyler\004\speech'
# putongsp = r'C:\Users\dslab\Documents\JohnYeung\lecture-chanting-segmentation\Tyler\training_v3 - Copy\Non-Chanting_speech\putongspeech'
dirs = [chanting, speech] 
class_names = [os.path.basename(d) for d in dirs] 
m_win, m_step, s_win, s_step = 4, 1, 0.05, 0.05 

# segment-level feature extraction:
features = [] 
for d in dirs: # get feature matrix for each directory (class) 
    f, files, fn = aF.directory_feature_extraction(d, m_win, m_step, 
                                                   s_win, s_step) 
    features.append(f)
# (each element of the features list contains a 
# (samples x segment features) = (10 x 138) feature matrix)
# print(features[0].shape, features[1].shape)
# print(fn)
# print(features[0])
# print(fn.index('spectral_centroid_mean'))
# select 2 features and create feature matrices for the two classes:
f1 = np.array([features[0][:, fn.index('zcr_mean')],
               features[0][:, fn.index('energy_entropy_mean')]])
f2 = np.array([features[1][:, fn.index('zcr_mean')],
               features[1][:, fn.index('energy_entropy_mean')]])

# plot 2D features
plots = [go.Scatter(x=f1[0, :],  y=f1[1, :], 
                    name=class_names[0], mode='markers'),
         go.Scatter(x=f2[0, :], y=f2[1, :], 
                    name=class_names[1], mode='markers')]
mylayout = go.Layout(xaxis=dict(title="zcr_mean"),
                     yaxis=dict(title="energy_entropy_mean"))
plotly.offline.iplot(go.Figure(data=plots, layout=mylayout))