# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 18:23:30 2016
@author: claire
"""

""" S and P phases picking following (Baillard 2014)
Functions : 
    
    - CFKurtosis :Characteristic function (CF) of the trace :  Kurtosis
    - F2 : Clean Cf of negative gradients
    - F3 :Remove linear trend of F2
    - MeanF3: Smooth F3
    - F4 : Greatest minima correspond to the greatest onset strenghts
    - F4prim : slim F4
    - DR :  Dip-rectinarity function
    - PSswavefilter :  P and S wave filter ( Ross et al 2016 )
    - WavePicking : P and S wave picking 
polarisation: calculate the degreee of rectinearility and the dip of maximum polarization
"""
import math
import numpy as np
import scipy
from obspy.signal.polarization import polarization_analysis
import matplotlib.pylab as plt

def CFkurtosis(tr,T, df):
    """return the caracteristic function based on the cumulative kurtosis calculate over a 5 second time window for the S wave picking 
    after trying I found that P picking is better when the kurtosis[i] is calculated on a center windows tr.data[i-T*100+1:i+1]
    input : 
        - tr: trace 
        - T : int, windows size in second
        - df : int,  samplin rate 
    output :
        - Cf Characteristic function which is the cumulative kurtosis calculated over T second time windows 
    exemple :
        for a period T = 5 seconds 
        Cf = CFkurtosisS(tr.data,5,100) """
        
    N = T*df
    Cf =np.zeros((1,N)) 
    cf = []
    for i in range(N, len(tr.data)):
        # the Kusrtosis is calculated on a T second moving windows 
        data = tr.data[i-N+1:i+1]  
        #Kurtosis calculation as follow the Pearson definition
        Kurtosis =  scipy.stats.kurtosis(data, axis = 0, bias=False, fisher=False) 
        cf.append(Kurtosis)
    Cf = np.append(Cf, np.array(cf))

    return Cf    

def F2(Cf):
    """ remove the negative slope of the characteristic function calculated as the sliding kurtosis fonction
    input :
        Cf: 1D array, Kurtosis characteristic funtion calculated with CFkurtosis 
    output:
        F2: 1D array,  Cf cleaned of negative gradients
    exemple :
        Cf = CFkurtosis(tr.data,5,100)
        F2 = F2(Cf)
    """
    F2=np.zeros(Cf.shape)
    F2[0]=Cf[0]
    delta = Cf[1:] - Cf[:-1]
    delta[delta < 0.] = 0.
    F2[1:] = np.cumsum(delta)
    return F2
        
def F3(F2):
    """ remove the linear trend of F2
    input:
        - F2: 1D array, Cf cleaned of negative gradients
    output:
        - F3: 1D array, F2 without linear trend
    exemple:
        Cf = CFkurtosis(tr.data,5,100)
        F2 = F2(Cf)
        F3 = F3(F2)
        """
    
    F3 = np.zeros(F2.shape, dtype=np.float32)
    a = (F2[-1] - F2[0]) / F2.size
    b = F2[0]
    F3 = F2 - (np.arange(F2.size, dtype=np.float32)*a + b)
    F3 = np.hstack( (np.array([0.]), F3[:-1]) )
    return F3
    
def MeanF3(F3,WT,SR=50.):
    """ Smooth F3 
    imput:
        -F3: 1D array, removal of the linear trend of F2
        -WT: int, smoothing windowss size in second 
    output :
        -F3mean: 1D array, F3 smoothed
        
    exemple:
        f3mean = MeanF3(F3,1)
"""    
    Wt = int(SR*WT)
    window = np.ones(Wt, dtype=np.float32)
    window /= window.size
    return scipy.ndimage.filters.gaussian_filter(F3, Wt/2.)
   
    
def F4(F3):
    """Greatest minima correspond to the greatest onset strenghts
    input : 
        -F3: 1D array, F2 cleaned of a linear trend and smooth
    output :
        -F4: 1D array,picking minima on F4 give a good estimate on phase onset 
        
    exemple :
        f4 = F4(F3)
    """
    F4 = np.zeros(F3.shape, dtype=np.float32)
    T = np.zeros(F3.shape, dtype=np.float32)
    for i in range((len(F3)-1)):
        T[i]=F3[i]-max(F3[i],F3[i+1])
        if T[i]<0:
            F4[i]=T[i]
        else : 
            F4[i]=0
    return F4
    
def F4prim(F4):
    """Slim the minimum of F4
    input : 
        -F4: 1D array, function where the greatest minima correspond to the greatest onset strenghts
    output :
        -F4prim: 1D array, Slim F4
    exemple :
        f4prim = F4prim(F3)
    """
    F4prim = np.zeros(F4.shape, dtype=np.float32)
    for i in range((len(F4prim)-1)):
        # f4 IS NEGATIVE
        F4prim[i] = F4[i] - max(F4[i],F4[i+1])       
    return F4prim
    
def DR(rectilinearity, dip,alpha):
    """calculate the dip-rectilinearity function defined in Baillard et al 2014
        Above TP ,the pick is declared as P; below Ts , the pick is declared as S.Ts=-Tp =-0.4 (but have to be tested)
    input 
        -rectilineariuty :type array rectilinearity calculated from flinn
        -dip : type array : dip calculated from flinnn polarisation analysis     
        -alpha: float,  weightweighting factor between 1 and 2 that depends on the clarity
                of the dip and rectilinearity: a value of 2 would be appropri-
                ate for perfectly polarized data, and 1 corresponds to poorly
                polarized data.
    output:
         -DR: array dip-rectinearity (help to determine if P or S wave)
    """
    DR = np.array([rectilinearity(i)*np.sign(alpha * math.sin(dip(i))-rectilinearity(i)) for i in range(len(rectilinearity))])
    return DR
    
def PSswavefilter(rectilinearity,incidence):
    """ p and s wave filter arrays based on Ross et al 2016 
    input : 
        - rectilinearity: type array, rectilinearity calculated from polarisation analysis
        -incidence : type rray calculated from polarisation analysis (flinn 1988 )
    output : 
        - pfilter: type np array; filter of p wave (the convolution with the row signal will remove the s phase)
        - sfilter :type np array s filter (the convolution with the row signal will remove the p phase)
    exemple:
        sfilter, pfilter = PSswavefilter(rectilinearity,incidence)
        Ssignal = sfilter * tr.data
    """
    sfilter = rectilinearity*(np.ones(len(incidence))-np.cos(incidence*math.pi/180))
    pfilter = rectilinearity*(np.cos(incidence*math.pi/180))
    return sfilter, pfilter
 
def check_best_comp(st, T, sr):
    comp = ['N', 'E']
    best_idx = 0
    cf_max = 0.
    for c in range(len(comp)):
        tr = st.select(component=comp[c])[0]
        Cf = CFkurtosis(tr, int(T), sr)
        if Cf.max() > cf_max:
            best_idx = c
            cf_max = Cf.max()
            best_kurto = Cf
    return st.select(component=comp[best_idx])[0], best_kurto

def WavePicking(st,T, SecondP,SecondS,plot):
    """  P and S Phase picking
    -input:
        -st : stream contain the 3 component of the signal
        -T : float, 5s time of the sliding windows see baillard but could be change trial shoaw goiod result with 5s. 
        -plot: bool, if true plot will be draw
        -SecondP, float theorical P arrival in second 
        -Seconds,float theorical S arrival in second 
    -output
        - minF4p: float, coordiantes of the p start in second
        - minF4s: float, coordiantes of the s start in second calculating using the kitosis derived function and rectilinearity
    -exemple:
        minF4p, minF4s = WavePicking3(st_copy,5, 0,10,True)
        """
    # initialisation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st = st.copy()
    trZ = st.select(component = 'Z')[0]
    sr = int(trZ.stats.sampling_rate)
    #trH = st.select(component = 'N')[0]
    trH, CfH = check_best_comp(st, T, sr)
    df = trZ.stats.sampling_rate
    #P_Sdelay = (SecondS-SecondP)
    #DT = max(np.int32(0.5*P_Sdelay*sr), np.int32(0.25*sr))
    DT = np.int32(0.25*sr)
    T_samples = int(T*sr)
    W_sm = 0.3 # width (s) of the gaussian kernel used for smoothing
    buf = T_samples + int(2 * W_sm * sr)
    
    ## Trim signal  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #if SecondP > 60*5: 
    #    a=trZ.stats.starttime 
    #    tr_copy.trim(starttime =a + SecondP-60)
    #    trH_copy.trim(starttime =a + SecondP-60)
    #    SecondP =SecondP-60
    #    if tr_copy.stats.endtime - tr_copy.stats.starttime> 10*60 : 
    #        trH_copy.trim(endtime = a + SecondP+60*10)
    #        tr_copy.trim(endtime = a + SecondP+60*10)
    #        st.trim(endtime = a + SecondP+60*10)
    #elif  tr_copy.stats.endtime - tr_copy.stats.starttime> 10*60 :
    #    a=trZ.stats.starttime 
    #    trH_copy.trim(endtime = a + SecondP+60*10)
    #    tr_copy.trim(endtime = a + SecondP+60*10)
    #    st.trim(endtime = a + SecondP+60*10)
        
    # P phase arrival picking~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Cf = CFkurtosis(trZ, int(T), sr) 
    f2 = F2(Cf)
    f3 = F3(f2) 
    #f3mean = MeanF3(f3, 0.3, SR=sr) #sliding windows of 0.5 s show good results 
    #f4 = F4(f3mean)   
    f4 = F4(f3)
    f4prim = F4prim(f4)

    
    # S phase arrival picking~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #CfH = CFkurtosis(trH, int(T), sr) 
    f2H = F2(CfH)
    f3H = F3(f2H)
    #f3meanN = MeanF3(f3N, 0.3, SR=sr) #sliding windows of 0.5 s show good results 
    #f4N = F4(f3meanN)
    f4H = F4(f3H)
    f4primH = F4prim(f4H)

    #polarisation analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start1 = trH.stats['starttime']
    for tr in st : 
        start = tr.stats['starttime']
        if start>start1 :
            start1 = start 
            
    pol = polarization_analysis(st,T,0.1,1.5,15,start1,trH.stats['endtime'], verbose=False, method='flinn' , var_noise=0.0)
    rectilinearity = pol['rectilinearity']
    incidence = pol['incidence']
    rectilinearity2 = scipy.signal.resample(rectilinearity, len(f4prim),axis=0)
    incidence2= scipy.signal.resample(incidence, len(f4prim),axis=0)

    
    #P S filter cf Ross 2016 
    sfilter, pfilter =  PSswavefilter(rectilinearity2,incidence2)
    f4prim *= pfilter
    f4primH *= sfilter
    sfilter_thrs = sfilter.max()/2.
    #tr_copyS = trH_copy.copy() 
    #tr_copyP = tr_copy.copy()
    #tr_copyS = tr_copyS.data * sfilter
    #tr_copyP = tr_copyP.data * pfilter
    
    # P and S wave picking  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mask = np.arange(buf,len(f4prim)-sr*2)
    minF4p = buf + np.argmin(f4prim[mask]) # prevent from pointing the end or the beginning of the trace !
    #if sfilter[minF4p]>sfilter_thrs :  #NOT A p WAVE HD/Working/WavePicking_%s.png'%component,dpi=300)       
    #    LminF4p2 = np.argwhere( (f4prim[buf:len(f4prim)-sr*2] < 0.1*f4prim[minF4p]) &\
    #                            (sfilter[buf:len(f4prim)-sr*2] < sfilter_thrs) )
    #    if len(LminF4p2)>0:
    #        Minp = LminF4p2[f4prim[buf+LminF4p2].argmin()]
    #        Minp1 = 0
    #        minimum = 0.
    #        for idx in LminF4p2:
    #            if f4prim[buf+idx] < minimum:
    #                Minp1 = idx
    #                minimum = f4prim[idx]
    #        if Minp != Minp1:
    #            print Minp, Minp1
    #        minF4p = buf + Minp  # P picking

    if isinstance(minF4p,(list, tuple,np.ndarray))==True :
        minF4p = minF4p[0]
  
    start_search = min(minF4p + DT, sfilter.size-2)
    mask = np.arange(start_search,sfilter.size)
    min2F4s2 = np.argmin(f4primH[mask])
    minF4s2 = start_search + min2F4s2
    #if sfilter[minF4s2] < sfilter_thrs :  #NOT A S WAVE S limite 
    #    LminF4s2 = np.argwhere((f4primH[start_search:] < 0.2*f4primH[minF4s2]))
    #    if len(LminF4s2)>0:        
    #        Mins = LminF4s2[f4prim[start_search + LminF4s2].argmin()]
    #        minF4s2 = start_search + Mins
    #if isinstance(minF4s2, (list, tuple,np.ndarray)) ==True :
    #    minF4s2 = minF4s2[0]

    # plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if plot == True:
        lw=2
        SR = 50.
        component = trZ.stats.channel
        #f, (ax0,ax1, ax2, ax3, ax4,ax5,ax6,ax7)=plt.subplots(8, sharex=True)
        f, axes = plt.subplots(5, sharex=True)
        (ax0, ax1, ax2, ax3, ax4) = axes
        x = np.arange(0., st[0].data.size/SR, 1./SR)
        ax0.plot(x,trH.data)
        ax0.set_ylabel('Waveform')
        #----------------------------------------
        ax1.plot(x, Cf, label='Sliding kurtosis Z-component', lw=lw)
        ax1.plot(x,CfH, label='Sliding kurtosis H-component', lw=lw)
        ax1.set_ylabel('Sliding Kurtosis')
        #----------------------------------------
        ax2.plot(x, f2 , label='First transform Z-component', lw=lw)
        ax2.plot(x, f2H, label='First transform H-component', lw=lw)
        ax2.set_ylabel('First Transform')
        #----------------------------------------
        ax3.plot(x,f3,  label='Second transform Z-component', lw=lw)
        ax3.plot(x,f3H, label='Second transform H-component', lw=lw)
        ax3.set_ylabel('Second Transform')
        #----------------------------------------
        ax4.plot(x, f4prim, label='Third transform Z-component', lw=lw)
        ax4.plot(x,f4primH, label='Third transform H-component', lw=lw)
        ax4.set_ylabel('Third Transform') 
        #----------------------------------------
        #ax5.plot(x,sfilter)
        #ax5.axhline(sfilter_thrs, ls='--', color='k')
        #ax5.axvline(x=minF4p,color= 'C3')
        #ax5.axvline(x=minF4s2,color= 'C2')
        #ax5.set_ylabel('$sfilter$')
        #ax6.plot(x,pfilter)
        #ax6.axvline(x=minF4p,color= 'C3')
        #ax6.axvline(x=minF4s2,color= 'C2')
        #ax6.set_ylabel('$pfilter$')
        ##ax7.plot(trH_copy, 'k')
        ##ax7.plot(tr_copyS, 'b')
        ##ax7.plot(tr_copyP, 'C3')
        #ax7.axvline(x=minF4p,color= 'C3')
        #ax7.axvline(x=minF4s2,color= 'C2')
        for i, ax in enumerate(axes):
            if i == 0:
                ax.axvline(x=np.float32(minF4p)/SR , color= 'C3', lw=lw, label='Picked P wave')
                ax.axvline(x=np.float32(minF4s2)/SR, color= 'C2', lw=lw, label='Picked S wave')
            else:
                ax.axvline(x=np.float32(minF4p)/SR , color= 'C3', ls='--', lw=0.75)
                ax.axvline(x=np.float32(minF4s2)/SR ,color= 'C2', ls='--', lw=0.75)
            ax.legend(loc='lower right', fancybox=True, framealpha=1.)
            if i == len(axes)-1:
                ax.set_xlabel('Time (s)')
            #ax.set_xlim(x.min(), x.max())
            ax.set_xlim(x.min(), 40.)
        ax0.set_title('%s.%s (sliding window\'s duration=%.1fs)' %(trZ.stats.station, trZ.stats.channel, T))    
    return minF4p/df, minF4s2/df
