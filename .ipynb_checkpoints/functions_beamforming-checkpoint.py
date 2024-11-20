
import datetime as dt
from numpy import (abs, real, cumsum, concatenate, mean, sign, ones, array,  pi,
                   arange, zeros, interp, square, sqrt, argmin, floor, sum, fliplr,sin,cos,radians,exp,linspace)
from scipy.fft import fft, ifft
from scipy.signal import correlate
from obspy.core import read
from obspy import UTCDateTime


def latlon_to_enu(lat, lon, ref_lat, ref_lon, radius= 6371.0):
        lat, lon, ref_lat, ref_lon = map(radians, [lat, lon, ref_lat, ref_lon])
        d_lat = lat - ref_lat
        d_lon = lon - ref_lon
        x = radius * d_lon * cos(ref_lat)
        y = radius * d_lat
        return x, y

def calculate_beam_power(stations,azimuth, velocity,all_ft_data,cartesian_coords,nfreq,frequency_band,data_gap):
        complex_sum = 0
        freq_int=linspace(frequency_band[0],frequency_band[1],nfreq) #frequency array for integration
        # Loop over each station
        k=0
        for ft_data in all_ft_data:
                if stations[k] in data_gap:
                        k+=1
                x, y = cartesian_coords[k]
                delay = (x * sin(radians(azimuth)) + y * cos(radians(azimuth))) / velocity
                phase_shift = exp(1j * 2 * pi * freq_int * delay)
                complex_sum += ft_data * phase_shift
                k+=1
        # Compute the integral over freq
        df=(frequency_band[1]-frequency_band[0])/nfreq
        beam_power = (sum((abs(complex_sum)**2)*(df)))/(frequency_band[1]-frequency_band[0])
        return beam_power

def day_process(project_folder, station, channel, year, day, fmin, fmax, 
                hmin, hmax, whiten, sr):
    #Load, filter, clip, and, if selected, whiten the data for a single channel for a single day (or part thereof)

    data_folder = project_folder + station+'/'+channel+'.D/'
    file_name ='EC.'+station+'..'+channel+'.D.'+str(year)+'.'+str(day).zfill(3)
    try:
        str1 = read(data_folder+file_name)
    except Exception as e:
        print (f'{data_folder+file_name} not available')
        return None

    str1.merge(fill_value='interpolate')
    str1.detrend('demean')
    
    #VCH1 seems to start about 1 second late each day, so trim start and end of any stream to avoid issues
    buf=10
    st = UTCDateTime(dt.datetime.strptime(str(year)+ " " +str(day).zfill(3), '%Y %j')) + hmin*60*60 + buf
    if hmax==24:
        buf=-10
    et = UTCDateTime(dt.datetime.strptime(str(year)+ " " +str(day).zfill(3), '%Y %j')) + hmax*60*60 + buf

    str1 = str1.trim(st, et)
    ds = str1[0].stats.sampling_rate
    str1 = str1.filter('bandpass', freqmin=0.01, freqmax=8.0, corners=2, zerophase=True)
    str1.decimate(int(ds/sr))
    #Filter to specified bandwidth
    str1 = str1.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    
    #Make sure length is even number - makes IFFT much faster
    N = len(str1[0].data[:-1])
    yf = fft(str1[0].data[:-1])

    try:
        yf_match = concatenate((yf[int(N/2):],yf[:int(N/2)]))
        
        if whiten == True:
            S = 400
            y_smooth = cumsum(abs(yf_match))
            y_smooth[S:] = y_smooth[S:] - y_smooth[:-S]
            y_smooth = y_smooth[S-1:]/S
            y_smooth = concatenate((y_smooth[0]*ones(int(S/2)-1),y_smooth,y_smooth[0]*ones(int(S/2))))       
            y_white = yf_match/y_smooth
            
        else:
            y_white = yf_match.copy()
        
        y_filt_white_split = concatenate((y_white[int(N/2):],y_white[:int(N/2)]))
        
        data_white_filt = real(ifft(y_filt_white_split))
        data_white_filt = data_white_filt - mean(data_white_filt)
        
        #Clip amplitude at 3*RMS?
        #RMS = sqrt(mean(square(data_white_filt)))
        #data_white_filt[abs(data_white_filt)>3*RMS] = 3*RMS*sign(data_white_filt[abs(data_white_filt)>3*RMS])
        
        #or 1-bit normalization
        data_white_filt = sign(data_white_filt)
        
    except:
        print('processing failed')
        return None
    
    return data_white_filt