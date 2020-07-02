# Hateful Memes

### Hosted BY FACEBOOK

Your goal is to create an algorithm that identifies multimodal hate speech in internet memes.

Take an image, add some text: you've got a meme. Internet memes are often harmless and sometimes hilarious. However, by using certain types of images, text, or combinations of each of these data modalities, the seemingly non-hateful meme becomes a multimodal type of hate speech, a hateful meme.

At the massive scale of the internet, the task of detecting multimodal hate is both extremely important and particularly difficult. As the illustrative memes above show, relying on just text or just images to determine whether or not a meme is hateful is insufficient.
Link: https://www.drivendata.org/competitions/64/hateful-memes/

On this notebook, we will use both NLP & image classifcation for classify harmfull Memes.
Memes, could be either: insult text over a valid image or a valid text on harmful image.

## Part:1 - NLP Classification
In this part, we are going to do some NLP work to find inappropriate or insult words.

## Part:2 - Image Classification
In this part, we run an deep learning for for detect an harmful image.

## Part:3 - Put All Together
On the final part, we need to combine both NLP calssification & image calssification for answer the question:
Which Meme is valid and which of then is harmful.

## Part:1 - NLP Classification


```python
#Setp:1 - Libraries
import json
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
en_stop = set(nltk.corpus.stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
import os
import re
import cv2
import numpy as np
import fastai
import shutil
```


```python
#Step:2 - Load JSON train set as Dataframe
#Load train json into pandas
trainDF = pd.read_json('G:/DataScienceProject/HatefulMemes/data/train.jsonl', lines=True)
trainDF['id'] = trainDF['id'].astype('int32')
trainDF['label'] = trainDF['label'].astype('int32')
trainDF.head()
```


```python
#Step:3 - Load dev(CV) & test json into pandas
#Dev 
devDF = pd.read_json('G:/DataScienceProject/HatefulMemes/data/dev.jsonl', lines=True)
devDF['id'] = devDF['id'].astype('int32')
devDF['label'] = devDF['label'].astype('int32')

#Test
testDF = pd.read_json('G:/DataScienceProject/HatefulMemes/data/test.jsonl', lines=True)
testDF['id'] = testDF['id'].astype('int32')
testDF['label'] = 0
testDF['label'] = testDF['label'].astype('int32')
testDF.head()
```


```python
#Step:4 - Clean text
stemmer = WordNetLemmatizer()
#Preprocess func
def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

#TEXT need STR
trainDF['textNew'] = ''
for i, value in enumerate(trainDF['text']):
    text = sent_tokenize(str(value))
    trainDF['textNew'].loc[i] = preprocess_text(text)

devDF['textNew'] = ''
for i, value in enumerate(devDF['text']):
    text = sent_tokenize(str(value))
    devDF['textNew'].loc[i] = preprocess_text(text)

testDF['textNew'] = ''
for i, value in enumerate(testDF['text']):
    text = sent_tokenize(str(value))
    testDF['textNew'].loc[i] = preprocess_text(text)
```


```python
#Step:5 - Check word profanity & count profanity
trainDF['profanity'] = 0
trainDF['profanity'] = trainDF['profanity'].astype('int32')
devDF['profanity'] = 0
devDF['profanity'] = devDF['profanity'].astype('int32')
testDF['profanity'] = 0
testDF['profanity'] = testDF['profanity'].astype('int32')

#Load Google profanity dictionary
profanityPath = 'G:/DataScienceProject/HatefulMemes/Google-profanity.txt'
with open(profanityPath, 'r') as f:
    profanity = f.read()
    f.close()

profanity = profanity.split("\n")

#Need to split text into list per a row
for i, value in enumerate(trainDF['textNew']):
    rowText = trainDF['textNew'].loc[i].split(" ")
    if set(profanity).intersection(list(set(rowText))) != 0:
        trainDF['profanity'].loc[i] = len(set(profanity).intersection(list(set(rowText))))

for i, value in enumerate(devDF['textNew']):
    rowText = devDF['textNew'].loc[i].split(" ")
    if set(profanity).intersection(list(set(rowText))) != 0:
        devDF['profanity'].loc[i] = len(set(profanity).intersection(list(set(rowText))))

for i, value in enumerate(testDF['textNew']):
    rowText = testDF['textNew'].loc[i].split(" ")
    if set(profanity).intersection(list(set(rowText))) != 0:
        testDF['profanity'].loc[i] = len(set(profanity).intersection(list(set(rowText))))

#Let;s check if we got a profanity issue:
print("Profanity %: {:2f}".format((sum(trainDF['profanity'])/len(trainDF['text']))*100))
```


```python
#Step:6 - Reorder col & save
trainDF = trainDF.drop(['text', 'textNew'], axis=1)
devDF = devDF.drop(['text', 'textNew'], axis=1)
testDF = testDF.drop(['text', 'textNew'], axis=1)
colList = list(trainDF.columns)
colList.remove('label')
colList.append('label')
#colList
trainDF = trainDF[colList]
devDF = devDF[colList]
testDF = testDF[colList]
trainDF.to_csv('G:/DataScienceProject/HatefulMemes/trainDF_prof.csv', index=False)
devDF.to_csv('G:/DataScienceProject/HatefulMemes/devDF_prof.csv', index=False)
testDF.to_csv('G:/DataScienceProject/HatefulMemes/testDF_prof.csv', index=False)
```

As for now, we are able to classify Memes text.
Let's continue with image classification.

## Part:2 - Image Classification


```python
#Step:7 - Image resize
def imgResize(file):
    fullImgName = folder + file
    img = cv2.imread(fullImgName)
    width = img.shape[1]
    height = img.shape[0]

    if width > height:
        ratio = int(width / 224)
        if ratio > 1:
            width = 224
            height = int(height / ratio)
        else:
            width = 224
            height = 224

    elif height > width:
        ratio = int(height / 224)
        if ratio > 1:
            height = 224
            width = int(width / ratio)
        else:
            width = 224
            height = 224

    elif width == height:
        width = 224
        height = 224

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(img, dsize)
    cv2.imwrite(fullImgName, output)

    return

for i, value in enumerate(imgList):
    imgResize(value)

```


```python
#Step:8- Create folder & split train
#Create train, cv(dev), test folders
if os.path.exists('G:/DataScienceProject/HatefulMemes/train') is not None:
    os.mkdir('G:/DataScienceProject/HatefulMemes/train')
    os.mkdir('G:/DataScienceProject/HatefulMemes/train/harmful')
    os.mkdir('G:/DataScienceProject/HatefulMemes/train/non-harmful')

if os.path.exists('G:/DataScienceProject/HatefulMemes/cv') is not None:
    os.mkdir('G:/DataScienceProject/HatefulMemes/cv')
    os.mkdir('G:/DataScienceProject/HatefulMemes/cv/harmful')
    os.mkdir('G:/DataScienceProject/HatefulMemes/cv/non-harmful')

if os.path.exists('G:/DataScienceProject/HatefulMemes/test') is not None:
    os.mkdir('G:/DataScienceProject/HatefulMemes/test')

#Split image according to their labels:
folder = 'G:/DataScienceProject/HatefulMemes/data/img/'
trainDF = pd.read_csv("G:/DataScienceProject/HatefulMemes/trainDF_prof.csv")
for i, value in enumerate(trainDF['id']):
    #print(str(value) + '.png')
    if len(str(trainDF['id'][i])) == 5:
        imgSrc = folder + str(value) + '.png'
    else:
        imgSrc = folder + '0' + str(value) + '.png'
    if trainDF['label'][i] == 1:
        imgDst = 'G:/DataScienceProject/HatefulMemes/train/harmful/' + str(value) + '.png'
    else:
        imgDst = 'G:/DataScienceProject/HatefulMemes/train/non-harmful/' + str(value) + '.png'

    shutil.copyfile(imgSrc, imgDst)

devDF = pd.read_csv("G:/DataScienceProject/HatefulMemes/devDF_prof.csv")
for i, value in enumerate(devDF['id']):
    #As image id once dispalyed as str while other dispalyed as int
    #Means that img id: 01224 can be dispalyed 1234 - we nedd to add 0
    if len(str(devDF['id'][i])) == 5:
        imgSrc = folder + str(value) + '.png'
    else:
        imgSrc = folder + '0' + str(value) + '.png'
    if devDF['label'][i] == 1:
        imgDst = 'G:/DataScienceProject/HatefulMemes/cv/harmful/' + str(value) + '.png'
    else:
        imgDst = 'G:/DataScienceProject/HatefulMemes/cv/non-harmful/' + str(value) + '.png'

    shutil.copyfile(imgSrc, imgDst)

testDF = pd.read_csv("G:/DataScienceProject/HatefulMemes/testDF_prof.csv")
for i, value in enumerate(testDF['id']):
    #print(str(value) + '.png')
    if len(str(testDF['id'][i])) == 5:
        imgSrc = folder + str(value) + '.png'
    else:
        imgSrc = folder + '0' + str(value) + '.png'
    imgDst = 'G:/DataScienceProject/HatefulMemes/test/' + str(value) + '.png'
    shutil.copyfile(imgSrc, imgDst)
```


```python
#Step:9- FASTAI
from fastai.vision import *
import warnings
warnings.filterwarnings('ignore')
path = 'G:/DataScienceProject/HatefulMemes/train'
folderList = os.listdir(path)
data = ImageDataBunch.from_folder(path,
                                  train=".",
                                  test="../cv",
                                  valid_pct=0.2,
                                  classes=folderList)
```


```python
#Step:10 - Check accuricy over iteration 
from fastai.metrics import error_rate # 1 - accuracy
learn = create_cnn(data, models.resnet34, metrics=accuracy)
defaults.device = torch.device('cuda') # makes sure the gpu is used
learn.fit_one_cycle(15)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.128977</td>
      <td>0.751048</td>
      <td>0.599412</td>
      <td>01:55</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.934994</td>
      <td>0.768768</td>
      <td>0.619412</td>
      <td>00:58</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.741383</td>
      <td>0.669122</td>
      <td>0.628235</td>
      <td>00:55</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.675674</td>
      <td>0.662907</td>
      <td>0.621765</td>
      <td>00:55</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.654934</td>
      <td>0.689120</td>
      <td>0.621176</td>
      <td>00:55</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.639400</td>
      <td>0.679731</td>
      <td>0.621765</td>
      <td>00:56</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.607651</td>
      <td>0.679025</td>
      <td>0.628235</td>
      <td>00:57</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.582655</td>
      <td>0.714491</td>
      <td>0.622941</td>
      <td>00:56</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.544849</td>
      <td>0.695222</td>
      <td>0.637647</td>
      <td>00:59</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.500382</td>
      <td>0.712179</td>
      <td>0.634706</td>
      <td>01:04</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.458525</td>
      <td>0.738097</td>
      <td>0.635882</td>
      <td>00:55</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.422243</td>
      <td>0.751213</td>
      <td>0.637647</td>
      <td>00:55</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.385850</td>
      <td>0.766543</td>
      <td>0.641176</td>
      <td>00:55</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.362012</td>
      <td>0.766549</td>
      <td>0.630588</td>
      <td>00:56</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.352800</td>
      <td>0.770023</td>
      <td>0.624118</td>
      <td>00:55</td>
    </tr>
  </tbody>
</table>



```python
learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='78' class='' max='106' style='width:300px; height:20px; vertical-align: middle;'></progress>
      73.58% [78/106 00:49<00:17 0.8363]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


![png](output_14_2.png)



```python
learn.fit_one_cycle(15, max_lr=slice(3e-5, 3e-4))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.370912</td>
      <td>0.781442</td>
      <td>0.627059</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.334173</td>
      <td>0.813298</td>
      <td>0.617647</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.322590</td>
      <td>0.927165</td>
      <td>0.620000</td>
      <td>01:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.304326</td>
      <td>0.962594</td>
      <td>0.635882</td>
      <td>01:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.242204</td>
      <td>1.046814</td>
      <td>0.650000</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.180188</td>
      <td>1.101047</td>
      <td>0.629412</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.137428</td>
      <td>1.236673</td>
      <td>0.610588</td>
      <td>01:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.091184</td>
      <td>1.323357</td>
      <td>0.647647</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.065771</td>
      <td>1.314638</td>
      <td>0.640588</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.041713</td>
      <td>1.422832</td>
      <td>0.642941</td>
      <td>01:10</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.021176</td>
      <td>1.458293</td>
      <td>0.641765</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.010701</td>
      <td>1.502716</td>
      <td>0.636471</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.007298</td>
      <td>1.520096</td>
      <td>0.648823</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.004224</td>
      <td>1.520694</td>
      <td>0.648235</td>
      <td>01:08</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.002895</td>
      <td>1.566119</td>
      <td>0.650000</td>
      <td>01:07</td>
    </tr>
  </tbody>
</table>



```python
learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='43' class='' max='106' style='width:300px; height:20px; vertical-align: middle;'></progress>
      40.57% [43/106 00:38<00:56 0.0033]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


![png](output_16_2.png)



```python
#Step:11 - Final model & save
if os.path.exists('G:/DataScienceProject/webcrawler/train/models/tmp.pth'):
    os.remove('G:/DataScienceProject/webcrawler/train/models/tmp.pth')
    os.rmdir('G:/DataScienceProject/webcrawler/train/models')
else:
    {}

learn.fit_one_cycle(15, max_lr=slice(3e-5, 3e-4))
learn.export('G:/DataScienceProject/HatefulMemes/fastai.pkl')
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.004419</td>
      <td>1.592780</td>
      <td>0.638824</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.006852</td>
      <td>1.776606</td>
      <td>0.648823</td>
      <td>01:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.057547</td>
      <td>1.897591</td>
      <td>0.592941</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.108457</td>
      <td>1.822715</td>
      <td>0.643529</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.102978</td>
      <td>1.585258</td>
      <td>0.608824</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.093831</td>
      <td>1.677260</td>
      <td>0.596471</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.068257</td>
      <td>1.621009</td>
      <td>0.634706</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.048783</td>
      <td>1.671408</td>
      <td>0.621176</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.032073</td>
      <td>1.757552</td>
      <td>0.636471</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.024467</td>
      <td>1.739254</td>
      <td>0.620588</td>
      <td>01:08</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.010387</td>
      <td>1.750874</td>
      <td>0.647059</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.005348</td>
      <td>1.814748</td>
      <td>0.637059</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.003191</td>
      <td>1.860233</td>
      <td>0.647647</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.002175</td>
      <td>1.840992</td>
      <td>0.646471</td>
      <td>01:07</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.002233</td>
      <td>1.846701</td>
      <td>0.648823</td>
      <td>01:08</td>
    </tr>
  </tbody>
</table>



```python
#Step12 - Image class interpetation
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(4, figsize=(20,25))
```






![png](output_18_1.png)



```python
#Step13 - Image classification
import fastai
from fastai.metrics import error_rate
from fastai.vision import *
import warnings
import pandas as pd
import os

learn = load_learner('G:/DataScienceProject/HatefulMemes/', 'fastai.pkl')

devDF = pd.read_csv('G:/DataScienceProject/HatefulMemes/devDF_prof.csv')
testDF = pd.read_csv('G:/DataScienceProject/HatefulMemes/testDF_prof.csv')
devDF['imgClass'] = 0
testDF['imgClass'] = 0
devDF['id'] = devDF['id'].astype('int32')
testDF['id'] = testDF['id'].astype('int32')

cvPath = 'G:/DataScienceProject/HatefulMemes/data/img/'
for i in range(0, len(devDF)):
    if devDF['id'].loc[i] < 10000:
        file = cvPath + '0' + str(devDF['id'].loc[i]) + '.png'
    else:
        file = cvPath + str(devDF['id'].loc[i]) + '.png'
    img = open_image(file)
    pred_class, pred_idx, output = learn.predict(img)
    if str(pred_class) == 'harmful':
        devDF['imgClass'].loc[i] == 1



testPath = cvPath
for i in range(0, len(testDF)):
    if testDF['id'].loc[i] < 10000:
        file = cvPath + '0' + str(testDF['id'].loc[i]) + '.png'
    else:
        file = cvPath + str(testDF['id'].loc[i]) + '.png'
    img = open_image(file)
    pred_class, pred_idx, output = learn.predict(img)
    if str(pred_class) == 'harmful':
        testDF['imgClass'].loc[i] == 1

devDF = devDF.drop(['img'], axis=1)
testDF = testDF.drop(['img'], axis=1)

colList = list(devDF.columns)
colList.remove('label')
colList.append('label')
devDF = devDF[colList]

colList = list(testDF.columns)
colList.remove('label')
colList.append('label')
testDF = testDF[colList]

devDF.to_csv('G:/DataScienceProject/HatefulMemes/devDF_forPhase2.csv', index=False)
testDF.to_csv('G:/DataScienceProject/HatefulMemes/testDF_forPhase2.csv', index=False)
'''
id,proba,label
39420,0.4,0
'''
```




    '\nid,proba,label\n39420,0.4,0\n'



# Part:3 - Put All Together

Now we can run the overlay ML agorithm.
We got the profanity check as count of harmful words.
Also, we have the image classification.
The above will give us 2 new classification features
So, what keeping us from run the final do the final ML?


```python
#Step14 - Build the final ML that overlay on both NLP & image classification
import nbconvert
import pandas as pd
from pycaret.classification import *

devDF = pd.read_csv('G:/DataScienceProject/HatefulMemes/devDF_forPhase2.csv')
testDF = pd.read_csv('G:/DataScienceProject/HatefulMemes/testDF_forPhase2.csv')
exp1 = setup(devDF, target = 'label')
```

     
    Setup Succesfully Completed!
    


<style  type="text/css" >
</style><table id="T_64fc6983_baf7_11ea_aae8_94de8078c78e" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow0_col1" class="data row0 col1" >4952</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow1_col0" class="data row1 col0" >Target Type</td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow1_col1" class="data row1 col1" >Binary</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow2_col0" class="data row2 col0" >Label Encoded</td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow2_col1" class="data row2 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow3_col0" class="data row3 col0" >Original Data</td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow3_col1" class="data row3 col1" >(500, 4)</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow4_col0" class="data row4 col0" >Missing Values </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow4_col1" class="data row4 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow5_col0" class="data row5 col0" >Numeric Features </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow5_col1" class="data row5 col1" >1</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow6_col0" class="data row6 col0" >Categorical Features </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow6_col1" class="data row6 col1" >2</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow7_col0" class="data row7 col0" >Ordinal Features </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow7_col1" class="data row7 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow8_col0" class="data row8 col0" >High Cardinality Features </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow8_col1" class="data row8 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow9_col0" class="data row9 col0" >High Cardinality Method </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow9_col1" class="data row9 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow10_col0" class="data row10 col0" >Sampled Data</td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow10_col1" class="data row10 col1" >(500, 4)</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow11_col0" class="data row11 col0" >Transformed Train Set</td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow11_col1" class="data row11 col1" >(349, 5)</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow12_col0" class="data row12 col0" >Transformed Test Set</td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow12_col1" class="data row12 col1" >(151, 5)</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow13_col0" class="data row13 col0" >Numeric Imputer </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow13_col1" class="data row13 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow14_col0" class="data row14 col0" >Categorical Imputer </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow14_col1" class="data row14 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow15_col0" class="data row15 col0" >Normalize </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow15_col1" class="data row15 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow16_col0" class="data row16 col0" >Normalize Method </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow16_col1" class="data row16 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow17_col0" class="data row17 col0" >Transformation </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow17_col1" class="data row17 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow18_col0" class="data row18 col0" >Transformation Method </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow18_col1" class="data row18 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow19_col0" class="data row19 col0" >PCA </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow19_col1" class="data row19 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow20_col0" class="data row20 col0" >PCA Method </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow20_col1" class="data row20 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow21_col0" class="data row21 col0" >PCA Components </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow21_col1" class="data row21 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow22_col0" class="data row22 col0" >Ignore Low Variance </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow22_col1" class="data row22 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow23_col0" class="data row23 col0" >Combine Rare Levels </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow23_col1" class="data row23 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow24_col0" class="data row24 col0" >Rare Level Threshold </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow24_col1" class="data row24 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow25_col0" class="data row25 col0" >Numeric Binning </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow25_col1" class="data row25 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow26_col0" class="data row26 col0" >Remove Outliers </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow26_col1" class="data row26 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow27_col0" class="data row27 col0" >Outliers Threshold </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow27_col1" class="data row27 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow28_col0" class="data row28 col0" >Remove Multicollinearity </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow28_col1" class="data row28 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow29_col0" class="data row29 col0" >Multicollinearity Threshold </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow29_col1" class="data row29 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow30_col0" class="data row30 col0" >Clustering </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow30_col1" class="data row30 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow31_col0" class="data row31 col0" >Clustering Iteration </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow31_col1" class="data row31 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow32_col0" class="data row32 col0" >Polynomial Features </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow32_col1" class="data row32 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow33_col0" class="data row33 col0" >Polynomial Degree </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow33_col1" class="data row33 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow34_col0" class="data row34 col0" >Trignometry Features </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow34_col1" class="data row34 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow35_col0" class="data row35 col0" >Polynomial Threshold </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow35_col1" class="data row35 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow36_col0" class="data row36 col0" >Group Features </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow36_col1" class="data row36 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow37_col0" class="data row37 col0" >Feature Selection </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow38_col0" class="data row38 col0" >Features Selection Threshold </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow38_col1" class="data row38 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow39_col0" class="data row39 col0" >Feature Interaction </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow39_col1" class="data row39 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow40_col0" class="data row40 col0" >Feature Ratio </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_64fc6983_baf7_11ea_aae8_94de8078c78elevel0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow41_col0" class="data row41 col0" >Interaction Threshold </td>
                        <td id="T_64fc6983_baf7_11ea_aae8_94de8078c78erow41_col1" class="data row41 col1" >None</td>
            </tr>
    </tbody></table>



```python
#Step:15 - Compare modules
compare_models()
```




<style  type="text/css" >
    #T_763def98_baf7_11ea_b9d2_94de8078c78e th {
          text-align: left;
    }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col1 {
            background-color:  yellow;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col2 {
            background-color:  yellow;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col6 {
            background-color:  yellow;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col4 {
            background-color:  yellow;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col3 {
            background-color:  yellow;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col5 {
            background-color:  yellow;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col6 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col0 {
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col1 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col2 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col3 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col4 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col5 {
            : ;
            text-align:  left;
        }    #T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col6 {
            : ;
            text-align:  left;
        }</style><table id="T_763def98_baf7_11ea_b9d2_94de8078c78e" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col0" class="data row0 col0" >Light Gradient Boosting Machine</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col1" class="data row0 col1" >0.561600</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col2" class="data row0 col2" >0.572000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col3" class="data row0 col3" >0.596700</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col4" class="data row0 col4" >0.557700</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col5" class="data row0 col5" >0.573600</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow0_col6" class="data row0 col6" >0.124800</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col0" class="data row1 col0" >Random Forest Classifier</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col1" class="data row1 col1" >0.558600</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col2" class="data row1 col2" >0.565900</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col3" class="data row1 col3" >0.497100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col4" class="data row1 col4" >0.574400</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col5" class="data row1 col5" >0.531000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow1_col6" class="data row1 col6" >0.117600</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col0" class="data row2 col0" >Gradient Boosting Classifier</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col1" class="data row2 col1" >0.549900</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col2" class="data row2 col2" >0.526300</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col3" class="data row2 col3" >0.515000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col4" class="data row2 col4" >0.552500</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col5" class="data row2 col5" >0.529500</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow2_col6" class="data row2 col6" >0.100700</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col0" class="data row3 col0" >Decision Tree Classifier</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col1" class="data row3 col1" >0.547100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col2" class="data row3 col2" >0.547400</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col3" class="data row3 col3" >0.503300</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col4" class="data row3 col4" >0.557000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col5" class="data row3 col5" >0.524900</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow3_col6" class="data row3 col6" >0.094800</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col0" class="data row4 col0" >Extra Trees Classifier</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col1" class="data row4 col1" >0.544200</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col2" class="data row4 col2" >0.536100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col3" class="data row4 col3" >0.502900</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col4" class="data row4 col4" >0.552800</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col5" class="data row4 col5" >0.525400</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow4_col6" class="data row4 col6" >0.088800</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col0" class="data row5 col0" >CatBoost Classifier</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col1" class="data row5 col1" >0.544100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col2" class="data row5 col2" >0.553800</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col3" class="data row5 col3" >0.532000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col4" class="data row5 col4" >0.551000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col5" class="data row5 col5" >0.537100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow5_col6" class="data row5 col6" >0.089100</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col0" class="data row6 col0" >Ada Boost Classifier</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col1" class="data row6 col1" >0.532900</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col2" class="data row6 col2" >0.531300</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col3" class="data row6 col3" >0.476100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col4" class="data row6 col4" >0.545800</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col5" class="data row6 col5" >0.495800</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow6_col6" class="data row6 col6" >0.067800</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col0" class="data row7 col0" >Naive Bayes</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col1" class="data row7 col1" >0.527200</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col2" class="data row7 col2" >0.530300</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col3" class="data row7 col3" >0.396100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col4" class="data row7 col4" >0.541400</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col5" class="data row7 col5" >0.450500</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow7_col6" class="data row7 col6" >0.057600</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col0" class="data row8 col0" >Extreme Gradient Boosting</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col1" class="data row8 col1" >0.521300</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col2" class="data row8 col2" >0.528300</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col3" class="data row8 col3" >0.521600</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col4" class="data row8 col4" >0.525100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col5" class="data row8 col5" >0.515500</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow8_col6" class="data row8 col6" >0.044800</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col0" class="data row9 col0" >Logistic Regression</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col1" class="data row9 col1" >0.498600</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col2" class="data row9 col2" >0.540700</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col3" class="data row9 col3" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col4" class="data row9 col4" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col5" class="data row9 col5" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow9_col6" class="data row9 col6" >0.000000</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col0" class="data row10 col0" >SVM - Linear Kernel</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col1" class="data row10 col1" >0.498600</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col2" class="data row10 col2" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col3" class="data row10 col3" >0.900000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col4" class="data row10 col4" >0.450000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col5" class="data row10 col5" >0.599900</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow10_col6" class="data row10 col6" >0.000000</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col0" class="data row11 col0" >Quadratic Discriminant Analysis</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col1" class="data row11 col1" >0.498600</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col2" class="data row11 col2" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col3" class="data row11 col3" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col4" class="data row11 col4" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col5" class="data row11 col5" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow11_col6" class="data row11 col6" >0.000000</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col0" class="data row12 col0" >Ridge Classifier</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col1" class="data row12 col1" >0.489900</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col2" class="data row12 col2" >0.000000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col3" class="data row12 col3" >0.514700</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col4" class="data row12 col4" >0.492700</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col5" class="data row12 col5" >0.500100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow12_col6" class="data row12 col6" >-0.018700</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col0" class="data row13 col0" >Linear Discriminant Analysis</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col1" class="data row13 col1" >0.489900</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col2" class="data row13 col2" >0.510300</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col3" class="data row13 col3" >0.514700</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col4" class="data row13 col4" >0.492700</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col5" class="data row13 col5" >0.500100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow13_col6" class="data row13 col6" >-0.018700</td>
            </tr>
            <tr>
                        <th id="T_763def98_baf7_11ea_b9d2_94de8078c78elevel0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col0" class="data row14 col0" >K Neighbors Classifier</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col1" class="data row14 col1" >0.487000</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col2" class="data row14 col2" >0.500200</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col3" class="data row14 col3" >0.493100</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col4" class="data row14 col4" >0.487300</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col5" class="data row14 col5" >0.486800</td>
                        <td id="T_763def98_baf7_11ea_b9d2_94de8078c78erow14_col6" class="data row14 col6" >-0.025100</td>
            </tr>
    </tbody></table>



This above comparison, give us very low accuracy.
So, let's stack few MLs into a single model.


```python
#Step:16 - Stacking model for improve ML
ridge = create_model('ridge')
lda = create_model('lda')
gbc = create_model('gbc')
xgboost = create_model('xgboost')

# stacking models
stacker = stack_models(estimator_list = [ridge,lda,gbc], meta_model = xgboost)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5429</td>
      <td>0.5621</td>
      <td>0.5556</td>
      <td>0.5556</td>
      <td>0.5556</td>
      <td>0.0850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4857</td>
      <td>0.5245</td>
      <td>0.2778</td>
      <td>0.5000</td>
      <td>0.3571</td>
      <td>-0.0161</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6571</td>
      <td>0.7435</td>
      <td>0.5556</td>
      <td>0.7143</td>
      <td>0.6250</td>
      <td>0.3182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6857</td>
      <td>0.6732</td>
      <td>0.7778</td>
      <td>0.6667</td>
      <td>0.7179</td>
      <td>0.3678</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.5714</td>
      <td>0.5882</td>
      <td>0.6111</td>
      <td>0.5789</td>
      <td>0.5946</td>
      <td>0.1408</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.5714</td>
      <td>0.6111</td>
      <td>0.4706</td>
      <td>0.5714</td>
      <td>0.5161</td>
      <td>0.1379</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.7714</td>
      <td>0.7582</td>
      <td>0.7647</td>
      <td>0.7647</td>
      <td>0.7647</td>
      <td>0.5425</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.7714</td>
      <td>0.8088</td>
      <td>0.8235</td>
      <td>0.7368</td>
      <td>0.7778</td>
      <td>0.5440</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.6857</td>
      <td>0.7761</td>
      <td>0.5294</td>
      <td>0.7500</td>
      <td>0.6207</td>
      <td>0.3657</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.6176</td>
      <td>0.5865</td>
      <td>0.5294</td>
      <td>0.6429</td>
      <td>0.5806</td>
      <td>0.2353</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.6361</td>
      <td>0.6632</td>
      <td>0.5895</td>
      <td>0.6481</td>
      <td>0.6110</td>
      <td>0.2721</td>
    </tr>
    <tr>
      <th>SD</th>
      <td>0.0906</td>
      <td>0.0965</td>
      <td>0.1556</td>
      <td>0.0882</td>
      <td>0.1187</td>
      <td>0.1794</td>
    </tr>
  </tbody>
</table>
</div>



```python
save_experiment(experiment_name = 'G:/DataScienceProject/HatefulMemes/Exp1')
```

    Experiment Succesfully Saved
    


```python
#Step17 - Prediction
prediction = predict_model(stacker, data = testDF)
```


```python
print(prediction)
```

            id  profanity  imgClass  label  Label   Score
    0    16395          0         0      0      1  0.9847
    1    37405          0         0      0      1  0.9847
    2    94180          0         0      0      1  0.9847
    3    54321          0         0      0      1  0.9847
    4    97015          0         0      0      1  0.9847
    ..     ...        ...       ...    ...    ...     ...
    995   3869          0         0      0      1  0.9847
    996  23817          0         0      0      1  0.9847
    997  56280          0         0      0      1  0.9847
    998  29384          0         0      0      1  0.9847
    999  34127          0         0      0      1  0.9847
    
    [1000 rows x 6 columns]
    


```python
#Step18 - Submission
submission = pd.DataFrame(columns=['id', 'proba', 'label'])
submission['id'] = prediction['id']
submission['proba'] = prediction['Score']
submission['label'] = prediction['Label']
submission.to_csv("G:/DataScienceProject/HatefulMemes/submit1.csv", index=False)
```
