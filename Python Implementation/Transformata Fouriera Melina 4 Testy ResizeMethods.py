
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


#lato 2000
#wzorce w obróconych obrazach
#st_30 = [1, 2, 3, 4, 5, 7, 9, 10, 13 ,15, 16, 17, 18, 19, 20] 
#numery wzorcow zawierajacych sie w zima 1 (2000px) obrocone o 30 st w prawo
#st_135 = [1, 2, 3, 4, 5, 7, 9, 10, 13, 15, 16, 17, 18, 19, 20]

#lato 4000
st_30_1 = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14] #4?
st_30_2 = [1, 2, 6, 7, 9, 10, 11, 12] #10????

st_135_1 = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
st_135_2 = [1, 2, 7, 9, 11, 12]
st_1=[1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15]
st_2=st_1

###4000 zima 1/2 bez obrotu
#st_1 = [1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
#st_2 = [1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15]
##4000 zima 1/2 30 stopni
#st_30_1=[1, 2, 3, 4, 5, 7, 8, 9, 12, 14, 15, 16, 17, 19, 20, 21, 25]
#st_30_2=[1, 2, 3, 5, 6, 8, 9, 10, 11, 14, 15]
##4000 zima 1/2 135 stopni
#st_135_1=[1, 2, 3, 5, 7, 9, 12, 14, 15, 16, 17, 19, 20, 21, 25]
#st_135_2=[1, 2, 3, 6, 8, 10, 11, 12, 14]

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
    ra=d*con_W/abs(d*con_W)
    r=abs(ft.ifft2(ra))
    return r
#-----------------------------------------------------------------------------
def FM(D,Wz,metoda):
    
    powiekszenie=D.shape[0]/Wz.shape[0]

    W = cv2.resize(Wz,(int(D.shape[0]),int(D.shape[0])),interpolation = metoda)
    
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
    G=(D.shape[0]*np.sqrt(2))//2
    m=D.shape[0]/np.log(R)
    center=(D.shape[0]//2,D.shape[1]//2)
    Dlog=cv2.warpPolar(Dfil,center,center,R, cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
    Wlog=cv2.warpPolar(Wfil,center,center,R, cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
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
        scale[i]=np.exp(wykl/m)/powiekszenie
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
            
    return obraz_przesuniety,r,scale[ktora],rangle[ktora],powiekszenie

#main--------------------------------------------------------------------------

def program_7(path,ref,metoda,numery_wzorcow1,numery_wzorcow2):
    
    start_time = time.time()
    ilosc_wzorcow = len(numery_wzorcow1)+len(numery_wzorcow2)
    plik=path+"\\testy.csv"
    csvfile = open(plik, "w+")
    csvfile.truncate()
    for t in range (0,ilosc_wzorcow):
        if t<len(numery_wzorcow1):
           obraz=plt.imread('E:\Praca inz\Fourier Melin\Obrazy\original\lato\lato1\lato1'+ref+'.pgm')
           wzor=skimage.io.imread('E:\Praca inz\Fourier Melin\Obrazy\original\lato\lato1\wzor_4000_'+str(numery_wzorcow1[t])+'.pgm')
        else:
           tmp=t-len(numery_wzorcow1)
           obraz=plt.imread('E:\Praca inz\Fourier Melin\Obrazy\original\lato\lato2\lato2'+ref+'.pgm')
           wzor=skimage.io.imread('E:\Praca inz\Fourier Melin\Obrazy\original\lato\lato2\wzor_4000_'+str(numery_wzorcow2[tmp])+'.pgm')
        #obraz=plt.imread('E:\Praca inz\Fourier Melin\Obrazy\mapa_zima\skala\mapa_zima_90%.pgm')
        #obraz=skimage.io.imread('E:\Praca inz\Fourier Melin\Obrazy\original\zima\zima1\zima1'+ref+'.pgm')
        #wzor=skimage.io.imread('E:\Praca inz\Fourier Melin\Obrazy\original\zima\zima1\wzor_2000_'+str(numery_wzorcow[(t)])+'.pgm')
        #wzor=skimage.io.imread('E:\Praca inz\Fourier Melin\Obrazy\mapa_zima\wzor'+str(t+1)+'.pgm')
        psize=int(obraz.shape[0]/2)
        
        maxima=9*[0]
        it=9*[0]
        i=0
        for k in range(0, 3):
            for l in range(0, 3):
                part=obraz[int(0+k*psize/2):psize+int(k*psize/2),int(0+l*psize/2):int(psize+l*psize/2)]
                result,cor,scale,angle,powiekszenie=FM(part,wzor,metoda)        #obraz przesuniety, ostatnią korelację
                
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
                
        #cv2.imwrite(path+"\\wzorzec_4000_"+str(t+1)+".jpg", result)
        #cv2.imwrite(path+"\\wzorzec_4000_"+str(t+1)+"o.jpg", part)
        with open(plik, 'a', encoding='utf-8', newline='') as csvfile:
            csvwriter=csv.writer(csvfile,delimiter=";")
            csvwriter.writerow([str(scale),str(angle)])
        
    print("--- %s seconds ---" % (time.time() - start_time))
    print(ref)
    
    
#numery_wzorcow=st_30        #st/st_30/st_135
#plik="testy_7_30.csv"      #6_34/ 6_09/ 6_11/ 6_30/ 6_135
#path="7\obrot_30"          #obrot/skala_jw 
#ref='_2000_30'

#program_7("testy_7_135.csv","7\obrot_135",'_2000_135',st_135)

#numery_wzorcow=st        #st/st_30/st_135
#plik="testy_7_08.csv"      #6_34/ 6_09/ 6_11/ 6_30/ 6_135
#path="7\skala_08"          #obrot/skala_jw 
#ref='_1600'
#
#program_7(plik,path,ref,numery_wzorcow)
#
#numery_wzorcow=st        #st/st_30/st_135
#plik="testy_7_12.csv"      #6_34/ 6_09/ 6_11/ 6_30/ 6_135
#path="7\skala_12"          #obrot/skala_jw 
#ref='_2400'
    
#metoda=cv2.INTER_LINEAR
##program_7("lato\\Wersja_7_Lin\\brak", '_4000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin\\obrot_30", '_4000_30_stopni', metoda, st_30_1, st_30_2)
#program_7("lato\\Wersja_7_Lin\\obrot_135", '_4000_135_stopni', metoda, st_135_1, st_135_2)
#program_7("lato\\Wersja_7_Lin\\skala_09", '_3600', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin\\skala_08", '_3200', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin\\skala_11", '_4400', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin\\skala_12", '_4800', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin\\skala_34", '_3000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin\\skala_54", '_5000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin\\obrot_90", '_4000_90_stopni', metoda, st_1, st_2)

#metoda=cv2.INTER_CUBIC
#program_7("lato\\Wersja_7_Cub\\brak", '_4000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Cub\\obrot_30", '_4000_30_stopni', metoda, st_30_1, st_30_2)
#program_7("lato\\Wersja_7_Cub\\obrot_135", '_4000_135_stopni', metoda, st_135_1, st_135_2)
#program_7("lato\\Wersja_7_Cub\\obrot_90", '_4000_90_stopni', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Cub\\skala_09", '_3600', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Cub\\skala_08", '_3200', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Cub\\skala_11", '_4400', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Cub\\skala_12", '_4800', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Cub\\skala_34", '_3000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Cub\\skala_54", '_5000', metoda, st_1, st_2)

#
#metoda=cv2.INTER_LANCZOS4
#program_7("lato\\Wersja_7_Lanczos\\brak", '_4000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lanczos\\obrot_30", '_4000_30_stopni', metoda, st_30_1, st_30_2)
#program_7("lato\\Wersja_7_Lanczos\\obrot_135", '_4000_135_stopni', metoda, st_135_1, st_135_2)
#program_7("lato\\Wersja_7_Lanczos\\obrot_90", '_4000_90_stopni', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lanczos\\skala_09", '_3600', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lanczos\\skala_08", '_3200', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lanczos\\skala_11", '_4400', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lanczos\\skala_12", '_4800', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lanczos\\skala_34", '_3000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lanczos\\skala_54", '_5000', metoda, st_1, st_2)
#
##bo brakowalo tu pliku z danymi
#program_7("lato\\Wersja_7_Lin\\brak", '_4000', cv2.INTER_LINEAR, st_1, st_2)
metoda=cv2.INTER_LINEAR
#program_7("lato\\Wersja_7_Lin_G\\brak", '_4000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin_G\\obrot_30", '_4000_30_stopni', metoda, st_30_1, st_30_2)
#program_7("lato\\Wersja_7_Lin_G\\obrot_135", '_4000_135_stopni', metoda, st_135_1, st_135_2)
#program_7("lato\\Wersja_7_Lin_G\\obrot_90", '_4000_90_stopni', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin_G\\skala_09", '_3600', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin_G\\skala_08", '_3200', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin_G\\skala_11", '_4400', metoda, st_1, st_2)
program_7("lato\\Wersja_7_Lin\\skala_12", '_4800', metoda, st_1, st_2)
#program_7("zima\\Wersja_7_Lin_G\\skala_34", '_3000', metoda, st_1, st_2)
#program_7("lato\\Wersja_7_Lin_G\\skala_54", '_5000', metoda, st_1, st_2)