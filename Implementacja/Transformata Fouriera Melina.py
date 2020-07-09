
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
D=plt.imread('Obrazy_Mellin_z_zajec\domek_r300.pgm')
#W=plt.imread('wzor.pgm')
W = plt.imread('Obrazy_Mellin_z_zajec\domek_r0_64.pgm')
plt.gray()

#Hanning window + zeros
def hanning2D(n):
    h=np.hanning(n)
    return np.sqrt(np.outer(h,h))
Wh=W*hanning2D(64)

wzorzec_zera=np.zeros(D.shape)
wzorzec_zera[0:Wh.shape[0],0:Wh.shape[1]]=Wh

#Fourier transorm
Df=ft.fft2(D)
Df=ft.fftshift(Df)
Wf=ft.fft2(wzorzec_zera)
Wf=ft.fftshift(Wf)

#HighPass Filter
def highpassFilter(size):
    rows=np.cos(np.pi*np.matrix([-0.5+x/(size[0]-1) for x in range (size[0])]))
    cols=np.cos(np.pi*np.matrix([-0.5+x/(size[1]-1) for x in range (size[1])]))    
    X=np.outer(rows,cols)
    return (1.0-X)*(2.0-X)

Dfil=abs(Df)*highpassFilter(D.shape)
Wfil=abs(Wf)*highpassFilter(D.shape)
plt.figure('Dfil')
plt.imshow(Dfil)
plt.figure('Wfil')
plt.imshow(Wfil)

#Log polar
R=D.shape[0]//2
m=D.shape[0]/np.log(R)
center=(D.shape[0]//2,D.shape[1]//2)
Dlog=cv2.logPolar(Dfil,center,m,cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
Wlog=cv2.logPolar(Wfil,center,m,cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)


Wlogf=ft.fft2(Wlog)
Wlogf=ft.fftshift(Wlogf)
Dlogf=ft.fft2(Dlog)
Dlogf=ft.fftshift(Dlogf)

#Phase correlation
def phase_cor(w,d):
    con_W=w.conj()
    R=d*con_W/abs(d*con_W)
    r=abs(ft.ifft2(R))
    return r
rr=phase_cor(Wlogf,Dlogf)

plt.figure('corr')
plt.imshow(rr)

f_mul = Wlogf.conj() * Dlogf
rev_mul = np.fft.ifft2(f_mul/abs(f_mul))
plt.figure("res1")
plt.imshow(abs(rev_mul))

#x-skala,y- obrot
y,x=np.unravel_index(np.argmax(rr),rr.shape)
c_logr=x
c_ang=y
size_logr=Dlog.shape[0]
if c_logr>size_logr//2:
    wykl=size_logr-c_logr
else:
    wykl=-c_logr
scale=np.exp(wykl/m)
A=c_ang*360.0/size_logr
angle1=-A
angle2=180-A

#translation(angle, rotation)
trans_center=(wzorzec_zera.shape[0]/2-0.5,wzorzec_zera.shape[1]/2-0.5)
W_zeros_center=np.zeros(D.shape)
W_zeros_center[D.shape[0]//2-(W.shape[0]//2):D.shape[1]//2+(W.shape[1]//2),D.shape[1]//2-(W.shape[1]//2):D.shape[1]//2+(W.shape[1]//2)]=Wh

#angle1
translation1=cv2.getRotationMatrix2D(trans_center,angle1,scale)
final_image1=cv2.warpAffine(W_zeros_center,translation1,W_zeros_center.shape)
final_ft1=ft.fft2(final_image1)
final_ft1=ft.fftshift(final_ft1)
r1=phase_cor(final_ft1,Df)
#angle2
translation2=cv2.getRotationMatrix2D(trans_center,angle2,scale)
final_image2=cv2.warpAffine(W_zeros_center,translation2,W_zeros_center.shape)
final_ft2=ft.fft2(final_image2)
final_ft2=ft.fftshift(final_ft2)
r2=phase_cor(final_ft2,Df)

plt.figure('Wlog')
plt.imshow(r1)
plt.figure('Dlog')
plt.imshow(r2)
#maximum arg
r12=[np.argmax(r1),np.argmax(r2)]
arg_max=r12[0]
r=r1
wza = final_image1
if np.max(r1)<np.max(r2):
    r=r2
    wza = final_image2
    arg_max=r12[1]


#plt.imshow(W_zeros_center)
y,x=np.unravel_index(arg_max,r.shape)
macierz_translacji = np.float32([[1,0,x],[0,1,y]]) 
obraz_przesuniety = cv2.warpAffine(wza, macierz_translacji,(r.shape[0],r.shape[1]))

#plt.subplot(211)
plt.figure('D')
plt.imshow(D)
plt.figure('W')
plt.imshow(W)
#plt.subplot(212)
plt.figure('res')
plt.imshow(obraz_przesuniety)
##plt.plot(x,y,'bo')
#plt.show()
#cv2.imwrite('Badania/Jakosc/Zima/93pro.jpg', obraz_przesuniety)
#cv2.imwrite('Badania/Jakosc/Zima/93pro_mapa.jpg', D)
