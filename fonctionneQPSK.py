# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as scy
import threading,time
import multiprocessing
import sys
import bitarray
import cmath
import struct

from scipy.fftpack import fft
from numpy import pi
from numpy import sqrt
from numpy import sin
from numpy import cos
from numpy import zeros
from numpy import r_
from  scipy.io.wavfile import read as wavread

# Used for symbol creation. Returns a decimal number from a 2 bit input
def GetQpskSymbol(bit1, bit2):
    if bit1 == 0 and bit2 == 0:
        return 0
    elif bit1 == 0 and bit2 == 1:
        return 1
    elif bit1 == 1 and bit2 == 0:
        return 2
    elif bit1 == 1 and bit2 == 1:
        return 3
    else:
        return -1

# Maps a given symbol to a complex signal. Optionally, noise and phase offset can be added.
def QpskSymbolMapper(symbols:int,amplitude_I, amplitude_Q,noise1=0, noise2=0,  phaseOffset1 = 0, phaseOffset2 = 0):
    if(symbols == 0):
        return sqrt(amplitude_I**2 + amplitude_Q**2)*(cos(np.deg2rad(45) + phaseOffset1)+ 1j *sin(np.deg2rad(45) + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 1):
        return sqrt(amplitude_I**2 + amplitude_Q**2)*(cos(np.deg2rad(135) + phaseOffset1) + 1j * sin(np.deg2rad(135)  + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 2):
        return sqrt(amplitude_I**2 + amplitude_Q**2)*(cos(np.deg2rad(225)  + phaseOffset1) + 1j * sin(np.deg2rad(225) + phaseOffset2)) + (noise1 + 1j*noise2)
    elif(symbols == 3):
        return sqrt(amplitude_I**2 + amplitude_Q**2)*(cos(np.deg2rad(315)  + phaseOffset1) + 1j *sin(np.deg2rad(315)  + phaseOffset2)) + (noise1 + 1j*noise2)
    else:
        return complex(0)




#-------------------------------------#
#---------- Configuration ------------#
#-------------------------------------#
f = open("chat_trognon.jpg", "rb")
bin = f.read()

#bin = struct.unpack("i" * ((len(bin) -24) // 4), bin[20:-4])
#print(bin)
# print(bin);print('zzzz')
fs = 44100                  # sampling rate
baud = 900                  # symbol rate
#Nbits = 1000000             # number of bits


# bin = int.from_bytes(bin)
# bits_str = ''.join(format(hex, '08b') for octet in bin)
bits_str = ''.join(f'{byte:08b}' for byte in bin)
# print(bits_str)
#inputBits = np.random.randn(Nbits,1) > 0 ; print(inputBits)
# print(type(inputBites))

inputBits = np.array([[int(c)] for c in bits_str])
print(inputBits)
Nbits = len(inputBits)
# print(len(bits))

f0 = 1800                   # carrier Frequency
Ns = int(fs/baud)           # number of Samples per Symbol
N = Nbits * Ns              # Total Number of Samples
t = r_[0.0:N]/fs            # time points
f = r_[0:N/2.0]/N*fs        # Frequency Points

# Limit for representation of time domain signals for better visibility. 
symbolsToShow = 20
timeDomainVisibleLimit = np.minimum(Nbits/baud,symbolsToShow/baud)     

#----------------------------#
#---------- QPSK ------------#
#----------------------------#
#Input of the modulator

# Conversion de chaque octet en une chaîne de bits



# bits_str = ''.join(format(octet, '08b') for octet in bin)


# inputBites = np.random.randn(Nbits,1) > 0 


inputSignal = (np.tile(inputBits*2-1,(1,Ns))).ravel()

#Only calculate when dividable by 2
if(len(inputBits)%2 == 0):

    #carrier signals used for modulation. carrier2 has 90° phaseshift compared to carrier1 -> IQ modulation
    carrier1 = cos(2*pi*f0*t)
    carrier2 = cos(2*pi*f0*t+pi/2)

    #Serial-to-Parallel Converter (1/2 of data rate)
    I_bits = inputBits[::2]
    Q_bits = inputBits[1::2]
       
    #Digital-to-Analog Conversion
    I_signal = (np.tile(I_bits*2-1,(1,2*Ns))).ravel()


    Q_signal = (np.tile(Q_bits*2-1,(1,2*Ns)) ).ravel()

    #Multiplicator / mixxer
    I_signal_modulated = I_signal * carrier1
    Q_signal_modulated = Q_signal * carrier2

    #Summation befor transmission
    QPSK_signal = I_signal_modulated + Q_signal_modulated

    #---------- Preperation QPSK Constellation Diagram ------------#
    dataSymbols = np.array([[GetQpskSymbol(I_bits[x],Q_bits[x])] for x in range(0,I_bits.size)])
    
    # amplitudes of I_signal and Q_signal (absolut values)
    amplitude_I_signal = 1
    amplitude_Q_signal = 1

    #Generate noise. Two sources for uncorelated noise for each amplitude.
    noiseStandardDeviation = 0.5
    noise1 = np.random.normal(0,noiseStandardDeviation,dataSymbols.size) 
    noise2 = np.random.normal(0,noiseStandardDeviation,dataSymbols.size)

    #Transmitted and received symbols. Rx symbols are generated under the presence of noise
    '''
    Tx_symbols = np.array([[QpskSymbolMapper(dataSymbols[x], 
                                             amplitude_I_signal, 
                                             amplitude_Q_signal,  
                                             phaseOffset1 = np.deg2rad(0), 
                                             phaseOffset2 = np.deg2rad(0)
                                             )] for x in range(0,dataSymbols.size)])'''
    Rx_symbols = np.array([[QpskSymbolMapper(dataSymbols[x], 
                                             amplitude_I_signal ,
                                             amplitude_Q_signal, 
                                             noise1=noise1[x], 
                                             noise2=noise2[x],
                                             )] for x in range(0,dataSymbols.size)])


def demodulate_qpsk_symbols(rx_symbols):
    """
    rx_symbols: array-like of complex QPSK symbols, shape (N,) or (N,1).
    Returns:    array of demodulated bits, shape (2N,).
                The bit-pairs correspond to:
                  Quadrant I   (I>0, Q>0)  -> symbol=0 -> bits (0,0)
                  Quadrant II  (I<0, Q>0)  -> symbol=1 -> bits (0,1)
                  Quadrant III (I<0, Q<0)  -> symbol=2 -> bits (1,0)
                  Quadrant IV  (I>0, Q<0)  -> symbol=3 -> bits (1,1)
    """
    # Flatten to 1-D if needed
    rx_symbols = np.ravel(rx_symbols)
    
    demod_bits = []
    for sym in rx_symbols:
        I = sym.real
        Q = sym.imag
        if I >= 0 and Q >= 0:
            # Quadrant I => symbol 0 => bits (0,0)
            demod_bits.extend([0, 0])
        elif I < 0 and Q >= 0:
            # Quadrant II => symbol 1 => bits (0,1)
            demod_bits.extend([0, 1])
        elif I < 0 and Q < 0:
            # Quadrant III => symbol 2 => bits (1,0)
            demod_bits.extend([1, 0])
        else:
            # Quadrant IV => symbol 3 => bits (1,1)
            demod_bits.extend([1, 1])
    
    return np.array(demod_bits, dtype=int)


# Example usage in your code:
received_bits = demodulate_qpsk_symbols(Rx_symbols)
print("Demodulated bits:", received_bits)

packed_bytes = np.packbits(received_bits) 

byte_data = packed_bytes.tobytes()

output_filename = "output_chat.jpg"
with open(output_filename, "wb") as f:
    f.write(byte_data)

print(f"Saved {len(byte_data)} bytes to {output_filename}")