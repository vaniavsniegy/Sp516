# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fftpack import fft
from numpy import pi
from numpy import r_

# Used for symbol creation. Returns a decimal number from a 1-bit input
def GetBpskSymbol(bit1: bool):
    if not bit1:
        return 0
    elif bit1:
        return 1
    else:
        return -1

# Maps a given symbol to a complex signal. Optionally, noise and phase offset can be added.
def BpskSymbolMapper(symbols: int, amplitude, noise1=0, noise2=0, phaseOffset=0):
    if symbols == 0:
        return amplitude * (np.cos(np.deg2rad(0) + phaseOffset)) + (noise1 + 1j * noise2)
    elif symbols == 1:
        return amplitude * (np.cos(np.deg2rad(180) + phaseOffset)) + (noise1 + 1j * noise2)
    else:
        return complex(0)


#-------------------------------------#
#---------- Configuration ------------#
#-------------------------------------#
fs = 44100                  # sampling rate
baud = 900                  # symbol rate
Nbits = 40                  # number of bits
f0 = 1800                   # carrier Frequency
Ns = int(fs / baud)         # number of Samples per Symbol
N = Nbits * Ns              # Total Number of Samples
t = r_[0.0:N] / fs          # time points
f = r_[0:N / 2.0] / N * fs  # Frequency Points


# Limits for time and frequency representation
symbolsToShow = 20
timeDomainVisibleLimit = np.minimum(Nbits / baud, symbolsToShow / baud)

sideLobesToShow = 9
sideLobeWidthSpectrum = baud
lowerLimit = np.maximum(0, f0 - sideLobeWidthSpectrum * (1 + sideLobesToShow))
upperLimit = f0 + sideLobeWidthSpectrum * (1 + sideLobesToShow)

carrier1 = np.cos(2 * pi * f0 * t)


#----------------------------#
#---------- BPSK ------------#
#----------------------------#

# Modulator Input : Génération des bits d'entrée
inputBits = np.random.randn(Nbits, 1) > 0

# Create Rectangular Pulse Train
pulse_duration = 1 / baud  # Duration of one bit (symbol period)
rect_pulse_samples = int(pulse_duration * fs)  # Number of samples for one rectangular pulse
rect_pulse = np.ones(rect_pulse_samples)  # Create the rectangular pulse

#Generate the Rectangular Pulse Train (Input Signal)
inputSignal = np.zeros(N)  # Initialize input signal
for i in range(Nbits):
    start_idx = i * rect_pulse_samples
    end_idx = start_idx + rect_pulse_samples
    inputSignal[start_idx:end_idx] = rect_pulse * (1 if inputBits[i] else -1)  # Map 1 to high, 0 to low

# Modulated Signal with BPSK
BPSK_signal = inputSignal * carrier1  # Multiply by the carrier to transmit




#---------- Prepare BPSK Constellation Diagram ------------#
amplitude = 1


####CHANNEL
# Generate noise for simulation
noiseStandardDeviation = 0.12
noise1 = np.random.normal(0, noiseStandardDeviation, Nbits)
noise2 = np.random.normal(0, noiseStandardDeviation, Nbits)


####RECEIVER
# Transmitted and received symbols
dataSymbols = np.array([[GetBpskSymbol(inputBits[x])] for x in range(0, inputBits.size)])
Tx_symbols = np.array([[BpskSymbolMapper(dataSymbols[x], amplitude, phaseOffset=np.deg2rad(0))] for x in range(0, dataSymbols.size)])
Rx_symbols = np.array([[BpskSymbolMapper(dataSymbols[x], amplitude, noise1=noise1[x], noise2=noise2[x], phaseOffset=np.deg2rad(0))] for x in range(0, dataSymbols.size)])

#rajout d'un passe-bas
from scipy.signal import butter, lfilter

def lowpass_filter(signal, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

rx_filtered = lowpass_filter(Rx_symbols.real, cutoff=baud, fs=fs)

samples_per_symbol = int(fs / baud)
rx_samples = rx_filtered[samples_per_symbol // 2::samples_per_symbol]




#----------------------------#
#----- Frequency Analysis ---#
#----------------------------#

# FFT de votre signal BPSK modulé
BPSK_fft = fft(BPSK_signal)
frequencies = np.fft.fftfreq(len(BPSK_signal), 1 / fs)

import scipy
from scipy.fftpack import fft, fftshift


# f contains the frequency components
# S is the PSD
(f, S) = scipy.signal.welch(BPSK_signal, fs)
(fb, Sb) = scipy.signal.welch(inputSignal, fs)

# Conversion en dB/Hz
S_dB = 10 * np.log10(S + 1e-12)  # Ajouter 1e-12 pour éviter log(0)
Sb_dB = 10 * np.log10(Sb + 1e-12)  # Ajouter 1e-12 pour éviter log(0)


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.semilogy(f, S)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title("Spectral density of BPSK signal (en dB)")
plt.grid()


plt.subplot(2, 1, 2)
plt.semilogy(fb, Sb)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title("Spectral density of input signal (en dB)")
plt.grid()

plt.show()



#---------- Plot BPSK Constellation Diagram ------------#
fig, axis = plt.subplots(2, 2, sharey='row')
fig.suptitle('Constellation Diagram BPSK', fontsize=12)

axis[0, 0].plot(Tx_symbols.real, Tx_symbols.imag, '.', color='C1')
axis[0, 0].set_title('Tx (Transmitted Symbols)')
axis[0, 0].set_xlabel('Inphase [V]')
axis[0, 0].set_ylabel('Quadrature [V]')
axis[0, 0].grid(True)

axis[0, 1].plot(Tx_symbols.real, Tx_symbols.imag, '-', Tx_symbols.real, Tx_symbols.imag, '.')
axis[0, 1].set_title('Tx with Trajectory')
axis[0, 1].set_xlabel('Inphase [V]')
axis[0, 1].set_ylabel('Quadrature [V]')
axis[0, 1].grid(True)

axis[1, 0].plot(Rx_symbols.real, Rx_symbols.imag, '.', color='C1')
axis[1, 0].set_title('Rx (Received Symbols)')
axis[1, 0].set_xlabel('Inphase [V]')
axis[1, 0].set_ylabel('Quadrature [V]')
axis[1, 0].grid(True)

axis[1, 1].plot(Rx_symbols.real, Rx_symbols.imag, '-', Rx_symbols.real, Rx_symbols.imag, '.')
axis[1, 1].set_title('Rx with Trajectory')
axis[1, 1].set_xlabel('Inphase [V]')
axis[1, 1].set_ylabel('Quadrature [V]')
axis[1, 1].grid(True)

plt.subplots_adjust(hspace=0.5)
plt.show()


#---------- Plot BPSK Modulation ------------#
fig, axis = plt.subplots(3, 1)
fig.suptitle('BPSK Modulation', fontsize=12)

axis[0].plot(t[:len(inputSignal)], inputSignal, color='C1')
axis[0].set_title('Input Signal (Rectangular Pulses)')
axis[0].set_xlabel('Time [s]')
axis[0].set_xlim(0, timeDomainVisibleLimit)
axis[0].set_ylabel('Amplitude [V]')
axis[0].grid(linestyle='dotted')

axis[1].plot(t[:len(carrier1)], carrier1, color='C2')
axis[1].set_title('Carrier Signal')
axis[1].set_xlabel('Time [s]')
axis[1].set_xlim(0, timeDomainVisibleLimit)
axis[1].set_ylabel('Amplitude [V]')
axis[1].grid(linestyle='dotted')

axis[2].plot(t[:len(BPSK_signal)], BPSK_signal, color='C3')
axis[2].set_title('BPSK Modulated Signal')
axis[2].set_xlabel('Time [s]')
axis[2].set_xlim(0, timeDomainVisibleLimit)
axis[2].set_ylabel('Amplitude [V]')
axis[2].grid(linestyle='dotted')

plt.subplots_adjust(hspace=0.5)
plt.show()

#---------- Spectrum Plots ------------#
fig, axs = plt.subplots(3, 1)
fig.suptitle('BPSK Modulation Spectrum (Rectangular Pulse Train)', fontsize=12)

axs[0].magnitude_spectrum(BPSK_signal, Fs=fs, color='C1')
axs[0].set_title('Magnitude Spectrum')
axs[0].set_xlim(lowerLimit, upperLimit)
axs[0].grid(linestyle='dotted')

axs[1].magnitude_spectrum(BPSK_signal, Fs=fs, scale='dB', color='C1')
axs[1].set_title('Log. Magnitude Spectrum')
axs[1].set_xlim(lowerLimit, upperLimit)
axs[1].grid(linestyle='dotted')

axs[2].psd(BPSK_signal, NFFT=len(BPSK_signal), Fs=fs)
axs[2].set_title('Power Spectrum Density (PSD)')
axs[2].set_xlim(lowerLimit, upperLimit)
axs[2].grid(linestyle='dotted')

plt.subplots_adjust(hspace=0.5)
plt.show()


#---------- PSD  ------------#
fig, axs = plt.subplots(2, 1)
fig.suptitle('Power Spectral density', fontsize=12)

axs[0].psd(BPSK_signal, NFFT=len(BPSK_signal), Fs=fs)
axs[0].set_title('BPSK Signal')
axs[0].set_xlim(lowerLimit, upperLimit)
axs[0].grid(linestyle='dotted')

axs[1].psd(inputSignal, NFFT=len(inputSignal), Fs=fs)
axs[1].set_title('Rect pulse signal')
axs[1].set_xlim(lowerLimit, upperLimit)
axs[1].grid(linestyle='dotted')


plt.subplots_adjust(hspace=0.5)
plt.show()


"""

#---------- PSD centrée ------------#
fig, axs = plt.subplots(2, 1)
fig.suptitle('Power Spectral Density (Centered on 0 Hz)', fontsize=12)

# Calculer la PSD avec fft et fftshift
frequencies = np.fft.fftfreq(len(BPSK_signal), d=1/fs)  # Fréquences associées
BPSK_fft = np.fft.fft(BPSK_signal)  # Transformée de Fourier
BPSK_psd = np.abs(BPSK_fft) ** 2 / len(BPSK_signal)  # Calcul de la PSD
BPSK_psd_shifted = np.fft.fftshift(BPSK_psd)  # Décalage pour centrer
frequencies_shifted = np.fft.fftshift(frequencies)  # Décalage des fréquences

# Affichage de la PSD du signal BPSK centrée
axs[0].plot(frequencies_shifted, 10 * np.log10(BPSK_psd_shifted + 1e-12), color='C1')
axs[0].set_title('BPSK Signal')
axs[0].set_xlabel('Frequency [Hz]')
axs[0].set_ylabel('PSD [dB/Hz]')
axs[0].grid(linestyle='dotted')

# Calculer et afficher la PSD pour le signal rectangulaire
input_fft = np.fft.fft(inputSignal)
input_psd = np.abs(input_fft) ** 2 / len(inputSignal)
input_psd_shifted = np.fft.fftshift(input_psd)
frequencies_shifted_input = np.fft.fftshift(frequencies)

axs[1].plot(frequencies_shifted_input, 10 * np.log10(input_psd_shifted + 1e-12), color='C2')
axs[1].set_title('Rect pulse signal')
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('PSD [dB/Hz]')
axs[1].grid(linestyle='dotted')

plt.subplots_adjust(hspace=0.5)
plt.show()
"""