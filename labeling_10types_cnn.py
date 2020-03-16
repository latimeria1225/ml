#!/usr/bin/env python
# coding: utf-8

# In[138]:


# Keras以外のライブラリ
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image
from sklearn.model_selection import train_test_split


# In[139]:


# Kerasのライブラリ
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model
import keras


# In[140]:


# 画像データと正解ラベルの読み込み
path = 'C:\\SIGNATE\\picture_labeling\\'
train_label = pd.read_csv(path + 'train_master.tsv', delimiter='\t')
train_image_path = path + 'train_images\\train_images\\'

X=[]
Y=[]

for i in range(len(train_label)):
    image = load_img(train_image_path + train_label.iloc[i,0], target_size=(96,96))
    data = np.asarray(image)
    X.append(data)
    Y.append(train_label.iloc[i,1])
    
X = np.array(X)
Y = np.array(Y)

X = X.astype('float32')
X = X / 255.0


# In[141]:


# 正解ラベルの形式を変換
Y = np_utils.to_categorical(Y, 10)


# In[142]:


# 学習用データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[143]:


# データの確認
print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)


# In[144]:


# kerasでCNNモデル構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))      
model.add(Activation('softmax'))


# In[145]:


# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# 実行
# 出力あり(verbose=1)。
history = model.fit(X_train, y_train, batch_size=100, epochs=100,
                   validation_data = (X_test, y_test), verbose = 1)


# In[146]:


# モデルの保存
model.save(path + 'model\\' + 'cnn_moodel_for_10_labeling.h5') 


# In[147]:


# 提出用データの読み込み
path = 'C:\\SIGNATE\\picture_labeling\\'
submit_label = pd.read_csv(path + 'sample_submit.tsv', delimiter='\t', skiprows=0)
test_image_path = path + 'test_images\\test_images\\'

X_submit=[]

for i in range(len(submit_label)):
    image = load_img(test_image_path + submit_label.iloc[i,0], target_size=(96,96))
    data = np.asarray(image)
    X_submit.append(data)
    
X_submit = np.array(X_submit)

X_submit = X_submit.astype('float32')
X_submit = X_submit / 255.0


# In[148]:


len(submit_label)


# In[149]:


# モデルの適用
output = model.predict(X_submit, batch_size=50, verbose=1, steps=None)
Y = []
Y = output


# In[150]:


print(Y)


# In[151]:


Y = np.argmax(Y, axis=1)


# In[152]:


print(Y[0:20])


# In[153]:


# 結果の出力
submit_data = pd.read_csv(path + 'sample_submit.tsv', sep='\t')

for index, row in submit_data.iterrows():
    submit_data.loc[index,'label_id'] = Y[index]

submit_data.to_csv(path + 'sample_submit.tsv', index=False, sep='\t')


# In[154]:


# グラフの表示

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(history):
    # Plot the loss in the history
    axL.plot(history.history['loss'],label="loss for training")
    axL.plot(history.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(history):
    # Plot the loss in the history
    axR.plot(history.history['accuracy'],label="loss for training")
    axR.plot(history.history['val_accuracy'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(history)
plot_history_acc(history)
fig.savefig(path + '\\fig\\')

