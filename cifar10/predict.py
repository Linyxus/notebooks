#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
from scipy import misc
import numpy

_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck']

def _display_pred(pred):
    print('### Possibilities:')
    for i, prob in enumerate(pred):
        print('%s: %.2f%%' % (_labels[i], prob * 100))

def _to_channel_first(img):
    res = numpy.zeros((3, 32, 32))
    for i in range(3):
        for j in range(32):
            for k in range(32):
                res[i][j][k] = img[j][k][i]
    return res

if __name__ == '__main__':
    print('# Welcome to Predictor!')
    print('# Loading model data...')
    model = keras.models.load_model('trained_model.h5')
    print('# Model loaded.')
    print('### Details:')
    print(model.summary())
    while True:
        filename = input('# Input filename of a image: ')
        im = misc.imread(filename)
        im = misc.imresize(im, (32, 32))
        im = _to_channel_first(im)
        x = numpy.array([im])
        pred = model.predict(x)[0]
        _display_pred(pred)
