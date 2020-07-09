# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:01:07 2019

@author: dorot
"""

import numpy.fft as ft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from scipy import signal

D=plt.imread('domek_r300.pgm')
W=plt.imread('wzor.pgm')
plt.gray()

wzorzec_zera=np.zeros(D.shape)
wzorzec_zera[0:W.shape[0],0:W.shape[1]]=W

Df=ft.fft2(D)
Dfs=ft.fftshift(Df)
Dan=np.angle(Dfs)
Wf=ft.fft2(wzorzec_zera)
Wfs=ft.fftshift(Wf)
Wan=np.angle(Wfs)

cor = signal.correlate2d (Wan,Dan )
corishift=ft.ifftshift(cor)
corinv=ft.ifft2(corishift)
plt.figure('korelacja')
plt.imshow(abs(corinv))

def phase_correlation(a, b):
    G_a = ft.fft2(a)
    G_b = ft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r
ph_cor=phase_correlation(D,wzorzec_zera)
plt.figure('korelacja faz')
plt.imshow(ph_cor)

plt.figure('wzor')
plt.imshow(wzorzec_zera)
plt.figure('ref')
plt.imshow(D)
y,x=np.unravel_index(np.argmax(ph_cor),ph_cor.shape)

print(x,y)
macierz_translacji = np.float32([[1,0,x],[0,1,y]]) # gdzie dx, dy -wektor przesuniecia, czyli wspolrzedne maksimum
obraz_przesuniety = cv2.warpAffine(wzorzec_zera, macierz_translacji,wzorzec_zera.shape)
plt.figure('wynik')
plt.imshow(obraz_przesuniety)
plt.show()