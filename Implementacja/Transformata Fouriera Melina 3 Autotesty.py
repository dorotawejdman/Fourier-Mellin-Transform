
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:01:07 2019

@author: dorota

Wersja z calym programem realizujacym transformacje F-M jako def, wykonywanym 
w petli na 9 fragmentach obrazu.
Dodane autotesty

"""
import copy
import numpy.fft as ft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import skimage
import csv
import time
start_time = time.time()

#Hanning window + zeros--------------------------------------------------------
def hanning2D(n):
    h=np.hanning(n)
    return np.sqrt(np.outer(h,h))

#HighPass Filter---------------------------------------------------------------
def highpassFilter(size):
    rows=np.cos(np.pi*np.matrix([-0.5+x/(size[0]-1) for x in range (size[0])]))
    cols=np.cos(np.pi*np.matrix([-0.5+x/(size[1]-1) for x in range (size[1])]))    
    X=np.outer(rows,cols)
    return (1.0-X)*(2.0-X)
#Phase correlation------------------------------------------------------
def phase_cor(w,d):
    con_W=w.conj()
    R=d*con_W/abs(d*con_W)
    r=abs(ft.ifft2(R))
    return r

#wzorce w obróconych obrazach zima_2000
#numery wzorcow zawierajacych sie w lato 1/2 (4000px) obrocone o 30/135 st 
nr_30_1 = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14] #4?
nr_30_2 = [1, 2, 6, 7, 9, 10, 11, 12] #10????

nr_135_1 = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
nr_135_2 = [1, 2, 7, 9, 11, 12]

nr_20 = [1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
nr_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
nr_15 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

#-----------------------------------------------------------------------------
def FM(D,W):
    
    d=10
    #d - ilosc znajdywanych maximow w korelacji logpolarow
    plt.gray()
    
    #Hanning window + zeros-
    Wh=W*hanning2D(W.shape[0])
    
    wzorzec_zera=np.zeros(D.shape)
    wzorzec_zera[0:Wh.shape[0],0:Wh.shape[1]]=Wh
    
    #First Fourier transformation--------------------------------------------------
    Df=ft.fft2(D)
    Df=ft.fftshift(Df)
    Wf=ft.fft2(wzorzec_zera)
    Wf=ft.fftshift(Wf)
    
    #HighPass Filter--------------------------------------------------------------- 
    Dfil=abs(Df)*highpassFilter(D.shape)
    Wfil=abs(Wf)*highpassFilter(D.shape)
    
    #Log polar---------------------------------------------------------------------
    R=D.shape[0]//2
    m=D.shape[0]/np.log(R)
    center=(D.shape[0]//2,D.shape[1]//2)
    Dlog=cv2.warpPolar(Dfil,center,center, R, cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
    Wlog=cv2.warpPolar(Wfil,center,center, R, cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
    # zrodlo, rozmiar, srodek
        
    Wlogf=ft.fft2(Wlog)
    Wlogf=ft.fftshift(Wlogf)
    Dlogf=ft.fft2(Dlog)
    Dlogf=ft.fftshift(Dlogf)
    
    #First Phase correlation------------------------------------------------------
    rr=phase_cor(Wlogf,Dlogf)
    
    trans_center=(wzorzec_zera.shape[0]/2-0.5,wzorzec_zera.shape[1]/2-0.5)
    W_zeros_center=np.zeros(D.shape)
    W_zeros_center[D.shape[0]//2-(W.shape[0]//2):D.shape[1]//2+(W.shape[1]//2),D.shape[1]//2-(W.shape[1]//2):D.shape[1]//2+(W.shape[1]//2)]=Wh
     
    #few max------------------------------------------------------------------- 
    rr_copy=copy.copy(rr)
    yc= d* [0]
    xc= d* [0]
    iteration=d*[0]
    scale = d*[0]
    angle1 = d*[0]
    angle2 = d*[0]
    rangle = d*[0]
    
    for i in range(0, d):
        yy,xx=np.unravel_index(np.argmax(rr_copy),rr_copy.shape)
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
            rangle[i]=angle2[i]
        else:
            arg_max=r12[0]
            r=r1
            wza = final_image1
            rangle[i]=angle1[i]
        if i > 0:
            if np.max(rprev)> np.max(r):
                arg_max=np.argmax(rprev)
                r=rprev
                wza= finalprev
                iteration[i]=i #tam gdzie ostatnie zero tamte wspolrzedne zostaly wybrane
                
        finalprev=copy.copy(wza)
        rprev=copy.copy(r)
    
    #Last translation--------------------------------------------------------------
    y1,x1=np.unravel_index(arg_max,r.shape)
    y=y1
    x=x1
    if x>D.shape[0]/2:
        x=-D.shape[0]+x
    if y>D.shape[0]/2:
        y=-D.shape[0]+y
        
    macierz_translacji = np.float32([[1,0,x],[0,1,y]]) 
    obraz_przesuniety = cv2.warpAffine(wza, macierz_translacji,(r.shape[0],r.shape[1]))
    
    for q in range(0, d):
        if iteration[q]==0:
            ktora=q
            
    return obraz_przesuniety,r,scale[ktora],rangle[ktora]

def program6(path,ref,numery_wzorcow1,numery_wzorcow2):
    
    start_time = time.time()   
    plik=path+"\\testy.csv"
    csvfile= open(plik, "w+")
    csvfile.truncate()
    
    ilosc_wzorcow = len(numery_wzorcow1) + len(numery_wzorcow2)
    
    for t in range (0,ilosc_wzorcow):
        
        if t<len(numery_wzorcow1):
            obraz = plt.imread('E:\Praca inz\Fourier Melin\Obrazy\original\zima\zima1\zima1'+ref+'.pgm')
            wzor = skimage.io.imread('E:\Praca inz\Fourier Melin\Obrazy\original\zima\zima1\wzor_4000_'+str(numery_wzorcow1[t])+'.pgm')
        else:
            tmp = t-len(numery_wzorcow1)
            obraz = plt.imread('E:\Praca inz\Fourier Melin\Obrazy\original\zima\zima2\lato2'+ref+'.pgm')
            wzor = skimage.io.imread('E:\Praca inz\Fourier Melin\Obrazy\original\zima\zima2\wzor_4000_'+str(numery_wzorcow2[tmp])+'.pgm')
        
        psize=int(obraz.shape[0]/2)
        
        maxima=9*[0]
        it=9*[0]
        i=0
        for k in range(0, 3):
            for l in range(0, 3):
                part=obraz[int(0+k*psize/2):psize+int(k*psize/2),int(0+l*psize/2):int(psize+l*psize/2)]
                result,cor,scale,angle=FM(part,wzor)        #obraz przesuniety, ostatnią korelację
                
                maxima[i]=np.max(cor)
                
                if k > 0 or l>0:
                    if np.max(cor_prev)>np.max(cor):
                        
                        cor=cor_prev
                        result=result_prev
                        part=part_prev
                        scale=scale_prev
                        angle=angle_prev
                        it[i]=1
                        
                result_prev=result
                cor_prev=cor
                part_prev=part
                scale_prev=scale
                angle_prev=angle
                i=i+1
                
        cv2.imwrite(path+"\wzorzec_4000_"+str(t+1)+".jpg", result)
        cv2.imwrite(path+"\wzorzec_4000_"+str(t+1)+"o.jpg", part)
        with open(plik, 'a', encoding='utf-8', newline='') as csvfile:
            csvwriter=csv.writer(csvfile,delimiter=";")
            csvwriter.writerow([str(scale),str(angle)])
        
    print("--- %s seconds ---" % (time.time() - start_time))


#program6("lato\\fragmentacja\\brak",'_4000',nr_15,nr_15)
program6("lato\\fragmentacja\\obrot_90",'_4000_90',nr_15,nr_15)
program6("lato\\fragmentacja\\obrot_30",'_4000_30',nr_30_1,nr_30_2)
program6("lato\\fragmentacja\\obrot_135",'_4000_135',nr_135_1,nr_135_2)
program6("lato\\fragmentacja\\skala_08",'_4000_08',nr_15,nr_15)
program6("lato\\fragmentacja\\skala_09",'_4000_09',nr_15,nr_15)
program6("lato\\fragmentacja\\skala_11",'_4000_11',nr_15,nr_15)
program6("lato\\fragmentacja\\skala_12",'_4000_12',nr_15,nr_15)