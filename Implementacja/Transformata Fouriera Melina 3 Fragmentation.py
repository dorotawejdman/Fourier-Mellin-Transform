
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:01:07 2019

@author: dorota

Wersja z calym programem realizujacym transformacje F-M jako def, wykonywanym 
w petli na 9 fragmentach obrazu.

"""
import copy
import numpy.fft as ft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
#-----------------------------------------------------------------------------
def FM(D,W):

    
    #D=plt.imread('Obrazy\mapa_zima\mapa_zima_kwadrat.pgm')
    #W=plt.imread('Obrazy\mapa_zima\skala\wzor_1_200_40%.pgm')
    
    d=6
    #d - ilosc znajdywanych maximow w korelacji logpolarow
    plt.gray()
#    plt.figure('D&W')
#    plt.subplot(121)
#    #plt.figure('D')
#    plt.imshow(D)
#    plt.subplot(122)
#    #plt.figure('W')
#    plt.imshow(W)
    
    #Hanning window + zeros--------------------------------------------------------
    def hanning2D(n):
        h=np.hanning(n)
        return np.sqrt(np.outer(h,h))
    Wh=W*hanning2D(W.shape[0])
    
    wzorzec_zera=np.zeros(D.shape)
    wzorzec_zera[0:Wh.shape[0],0:Wh.shape[1]]=Wh
    
    #First Fourier transformation--------------------------------------------------
    Df=ft.fft2(D)
    Df=ft.fftshift(Df)
    Wf=ft.fft2(wzorzec_zera)
    Wf=ft.fftshift(Wf)
    
    #HighPass Filter---------------------------------------------------------------
    def highpassFilter(size):
        rows=np.cos(np.pi*np.matrix([-0.5+x/(size[0]-1) for x in range (size[0])]))
        cols=np.cos(np.pi*np.matrix([-0.5+x/(size[1]-1) for x in range (size[1])]))    
        X=np.outer(rows,cols)
        return (1.0-X)*(2.0-X)
    
    Dfil=abs(Df)*highpassFilter(D.shape)
    Wfil=abs(Wf)*highpassFilter(D.shape)
    plt.figure('Dfil')
    plt.imshow(Dfil)
    #plt.figure('Wfil')
    #plt.imshow(Wfil)
    
    #Log polar---------------------------------------------------------------------
    R=D.shape[0]//2
    m=D.shape[0]/np.log(R)
    center=(D.shape[0]//2,D.shape[1]//2)
    Dlog=cv2.warpPolar(Dfil,center,center, m, cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
    Wlog=cv2.warpPolar(Wfil,center,center, m, cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
    # zrodlo, rozmiar, srodek
    #Fill_outliers -fills all of the destination image pixels. If some of them correspond to outliers in the source image, they are set to zero
    #Dlog=cv2.logPolar(Dfil,center,m,cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
    #Wlog=cv2.logPolar(Wfil,center,m,cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
    
    plt.figure('logpolar')
    plt.subplot(211)
    #plt.figure('dlog')
    plt.imshow(Dlog)
    plt.subplot(212)
    #plt.figure('wlog')
    plt.imshow(Wlog)
    
    Wlogf=ft.fft2(Wlog)
    Wlogf=ft.fftshift(Wlogf)
    Dlogf=ft.fft2(Dlog)
    Dlogf=ft.fftshift(Dlogf)
    
    #Second Phase correlation------------------------------------------------------
    def phase_cor(w,d):
        con_W=w.conj()
        R=d*con_W/abs(d*con_W)
        r=abs(ft.ifft2(R))
        return r
    rr=phase_cor(Wlogf,Dlogf)
    
    plt.figure('second correlation of log')
    plt.imshow(rr)
    
    trans_center=(wzorzec_zera.shape[0]/2-0.5,wzorzec_zera.shape[1]/2-0.5)
    W_zeros_center=np.zeros(D.shape)
    W_zeros_center[D.shape[0]//2-(W.shape[0]//2):D.shape[1]//2+(W.shape[1]//2),D.shape[1]//2-(W.shape[1]//2):D.shape[1]//2+(W.shape[1]//2)]=Wh
     
    #few max-------------------------------------------------------------------
    
    rr_copy=copy.copy(rr)
    #d=15 #ilosc it w liczeniu wspolrzednych
    yc= d* [0]
    xc= d* [0]
    iteration=d*[0]
    scale = d*[0]
    angle1 = d*[0]
    angle2 = d*[0]
    
    for i in range(0, d):
        yy,xx=np.unravel_index(np.argmax(rr_copy),rr_copy.shape)
        #yt.astype(int)
        #xt.astype(int)
        yc[i]=yy
        xc[i]=xx
        rr_copy[yy,xx]=0
    
    #x-scale, y-rotation-----------------------------------------------------------  
    
        c_logr=xx
        c_ang=yy
        size_logr=Dlog.shape[0]
        if c_logr>size_logr//2:
            wykl=size_logr-c_logr
        else:
            wykl=-c_logr 
        scale[i]=np.exp(wykl/m)
        A=c_ang*360.0/size_logr
        angle1[i]=-A
        angle2[i]=180-A
        
        #translation(angle, scale)--------------------------------------------------
        #angle1
        translation1=cv2.getRotationMatrix2D(trans_center,angle1[i],scale[i])
        final_image1=cv2.warpAffine(W_zeros_center,translation1,W_zeros_center.shape)
        final_ft1=ft.fft2(final_image1)
        final_ft1=ft.fftshift(final_ft1)
        r1=phase_cor(final_ft1,Df)
        
        #angle2
        translation2=cv2.getRotationMatrix2D(trans_center,angle2[i],scale[i])
        final_image2=cv2.warpAffine(W_zeros_center,translation2,W_zeros_center.shape)
        final_ft2=ft.fft2(final_image2)
        final_ft2=ft.fftshift(final_ft2) #tu musi byc shift
        r2=phase_cor(final_ft2,Df)
        
    #maximum arg-------------------------------------------------------------------
        r12=[np.argmax(r1),np.argmax(r2)]
    
        if np.max(r1)<np.max(r2):
            arg_max=r12[1]
            r=r2
            wza = final_image2
        else:
            arg_max=r12[0]
            r=r1
            wza = final_image1
        if i > 0:
            if np.max(rprev)> np.max(r):
                arg_max=np.argmax(rprev)
                r=rprev
                wza= finalprev
                iteration[i]=i #tam gdzie ostatnie zero tamte wspolrzedne zostaly wybrane
                
        finalprev=copy.copy(wza)
        rprev=copy.copy(r)
    
    #plt.figure('Wlog')
    #plt.imshow(r1)
    #plt.figure('Dlog')
    #plt.imshow(r2)
     
    #plt.figure('final image - Scale + rotation')
    #plt.imshow(wza)
    #plt.figure('correlation of final image')
    #plt.imshow(r)
    
    #Last translation--------------------------------------------------------------
    #plt.imshow(W_zeros_center)
    y1,x1=np.unravel_index(arg_max,r.shape)
    y=y1
    x=x1
    if x>D.shape[0]/2:
        x=-D.shape[0]+x
    if y>D.shape[0]/2:
        y=-D.shape[0]+y
        
    macierz_translacji = np.float32([[1,0,x],[0,1,y]]) 
    obraz_przesuniety = cv2.warpAffine(wza, macierz_translacji,(r.shape[0],r.shape[1]))

    #plt.plot(x,y,'bo')
    #plt.show()
#    cv2.imwrite('Badania/2.jpg', obraz_przesuniety)
#    cv2.imwrite('Badania/1.jpg', D)
    return obraz_przesuniety,r


#D=plt.imread('obrazy_Mellin\domek_r0.pgm')
#W = plt.imread('obrazy_Mellin\domek_r0_64.pgm')
#D=plt.imread('Obrazy\mapa_zima\przesuniecie\s0.pgm')
#W=plt.imread('Obrazy\mapa_zima\przesuniecie\s30.pgm')
    
wzor=plt.imread('Obrazy\mapa_zima\wzor11.pgm')
obraz=plt.imread('Obrazy\mapa_zima\skala\mapa_zima_90%.pgm')

obraz=plt.imread('E:\Praca inz\Fourier Melin\Obrazy\original\zima\zima1\zima1_1600.pgm')
wzor=plt.imread('E:\Praca inz\Fourier Melin\Obrazy\original\zima\zima1\wzor_2000_1.pgm')

psize=int(obraz.shape[0]/2)
#part=np.zeros([psize,psize])

maxima=9*[0]
it=9*[0]
i=0
for k in range(0, 3):
    for l in range(0, 3):
        part=obraz[int(0+k*psize/2):psize+int(k*psize/2),int(0+l*psize/2):int(psize+l*psize/2)]
        result,cor=FM(part,wzor)        #obraz przesuniety, ostatnią korelację
        cv2.imwrite("Badania/Fragmentacja/fragment"+str(k)+str(l)+".jpg", result)
        cv2.imwrite("Badania/Fragmentacja/fragment"+str(k)+str(l)+"o.jpg", part)   
        
        maxima[i]=np.max(cor)
        
        if k > 0 or l>0:
            if np.max(cor_prev)>np.max(cor):
                
                cor=cor_prev
                result=result_prev
                part=part_prev
                it[i]=1
                
        result_prev=result
        cor_prev=cor
        part_prev=part
        i=i+1

#result=FM(D)
#
plt.figure('Result')
plt.subplot(121)
plt.imshow(part)
plt.subplot(122)
plt.imshow(result)