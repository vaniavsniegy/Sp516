import numpy as np
import matplotlib.pyplot as plt
import struct
import matplotlib
matplotlib.use('WebAgg')

from numpy import pi, cos, sin, sqrt, r_
from scipy.fftpack import fft
from scipy.signal import lfilter

# ----------------------------------------------------------------
#                      QPSK FUNCTIONS
# ----------------------------------------------------------------

def GetQpskSymbol(bit1, bit2):
    """
    Convert two bits to a QPSK symbol index: 
      00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3
    """
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

def QpskSymbolMapper(symbol, amplitude_I, amplitude_Q, 
                     noise1=0, noise2=0, phaseOffset1=0, phaseOffset2=0):
    """
    Given a symbol (0..3), generate the corresponding complex QPSK sample.
    The nominal angles are 45°, 135°, 225°, 315° (diagonals).
    """
    r = sqrt(amplitude_I**2 + amplitude_Q**2)
    if symbol == 0:
        angle_deg = 45
    elif symbol == 1:
        angle_deg = 135
    elif symbol == 2:
        angle_deg = 225
    elif symbol == 3:
        angle_deg = 315
    else:
        return 0+0j
    
    return (r * (cos(np.deg2rad(angle_deg) + phaseOffset1)
                  + 1j*sin(np.deg2rad(angle_deg) + phaseOffset2))
            + (noise1 + 1j*noise2))

def demodulate_qpsk_symbols(rx_symbols):
    """
    rx_symbols: complex QPSK symbols (baseband)
    Return:     array of demodulated bits (2 bits per symbol).
    Decoding based on quadrant:
        I (I>=0, Q>=0)   -> symbol=0 -> bits (0,0)
        II (I<0, Q>=0)   -> symbol=1 -> bits (0,1)
        III (I<0, Q<0)   -> symbol=2 -> bits (1,0)
        IV (I>=0, Q<0)   -> symbol=3 -> bits (1,1)
    """
    rx_symbols = np.ravel(rx_symbols)
    demod_bits = []
    for sym in rx_symbols:
        I = sym.real
        Q = sym.imag
        if I >= 0 and Q >= 0:
            demod_bits.extend([0, 0])  # symbol 0
        elif I < 0 and Q >= 0:
            demod_bits.extend([0, 1])  # symbol 1
        elif I < 0 and Q < 0:
            demod_bits.extend([1, 0])  # symbol 2
        else:
            demod_bits.extend([1, 1])  # symbol 3
    return np.array(demod_bits, dtype=int)

# ----------------------------------------------------------------
#   SIMPLE RECTANGULAR PULSE SHAPER AND LOW-PASS FILTER EXAMPLES
# ----------------------------------------------------------------

def rectangular_pulse(num_samples):
    """
    Return a simple rectangular pulse of length num_samples.
    """
    return np.ones(num_samples)

def lowpass_filter(signal, cutoff_Hz, fs):
    """
    Very simple FIR lowpass filter design using a rectangular window.
    In a real system, use a more robust design (e.g., scipy.signal.firwin).
    """
    # Filter length (in samples)
    N = 101  
    # Normalized cutoff in [0..1] for firwin, but let's do a naive approach here
    # and just build a rectangular impulse response
    fc_norm = cutoff_Hz / (fs / 2)
    # Create an ideal sinc-based filter or a simpler rectangular approach
    # For brevity, let's do a basic windowed sinc approach:
    n = np.arange(-int((N-1)/2), int((N-1)/2)+1)
    # Avoid division by zero:
    h = np.sinc(2*fc_norm * n)
    # Windowing (Hamming for example)
    w = 0.54 - 0.46 * np.cos(2*pi * (n + (N-1)/2) / (N-1))
    h = h * w
    h = h / np.sum(h)
    
    # Filter the signal
    filtered = np.convolve(signal, h, mode='same')
    return filtered

# ----------------------------------------------------------------
#                   MAIN SIMULATION CODE
# ----------------------------------------------------------------

# 1. READ THE FILE
filename_in = "chat.jpg"
with open(filename_in, "rb") as f:
    file_data = f.read()

# 2. SPLIT THE FILE INTO HEADER (PROTECTED) + BODY (NOISY)
#    Adjust header_size as you see fit (e.g., 512, 1024, etc.).
header_size = 2048
header_data = file_data[:header_size]  # protected
body_data   = file_data[header_size:]  # noisy

# 3. CONVERT BOTH PARTS TO BITS
#    (A) Convert header_data directly to bits
header_bits_str = ''.join(f'{byte:08b}' for byte in header_data)
header_bits = np.array([int(c) for c in header_bits_str], dtype=np.uint8)

#    (B) Convert body_data to bits
body_bits_str = ''.join(f'{byte:08b}' for byte in body_data)
body_bits = np.array([int(c) for c in body_bits_str], dtype=np.uint8)

# 4. DEFINE SIMULATION PARAMETERS
fs      = 44100    # sampling rate
baud    = 900      # symbol rate
f0      = 1800     # carrier frequency
Ns      = int(fs/baud)  # samples per symbol
amp_I   = 1.0
amp_Q   = 1.0
noise_std_dev = 1  # std dev of noise

# ------------------- DOPPLER SHIFT & COMPENSATION -------------------
doppler_shift = 50    # Hz shift in channel
doppler_est   = 50    # Hz estimate used in receiver (assume we know perfectly)

# ----------------------------------------------------------------
#                  TRANSMITTER
# ----------------------------------------------------------------

# (1) SYMBOLS CREATION
if len(body_bits) % 2 != 0:
    body_bits = np.append(body_bits, 0)  # pad with one zero if needed

symbols = []
for i in range(0, len(body_bits), 2):
    b1 = body_bits[i]
    b2 = body_bits[i+1]
    sym_idx = GetQpskSymbol(b1, b2)
    symbols.append(sym_idx)
symbols = np.array(symbols, dtype=int)

# (2) CHOICE OF PULSE FUNCTION
pulse = rectangular_pulse(Ns)

# (3) Baseband QPSK (no noise here)
tx_symbols_baseband = np.array([
    QpskSymbolMapper(sym, amp_I, amp_Q, noise1=0, noise2=0) 
    for sym in symbols
], dtype=complex)

# Upsample & pulse-shape
tx_baseband_upsampled = np.zeros(len(tx_symbols_baseband)*Ns, dtype=complex)
for i, val in enumerate(tx_symbols_baseband):
    start = i * Ns
    tx_baseband_upsampled[start:start+Ns] = val * pulse

time = np.arange(len(tx_baseband_upsampled)) / fs
I_comp = np.real(tx_baseband_upsampled)
Q_comp = np.imag(tx_baseband_upsampled)

# (4) GENERATION OF THE TX SIGNAL at frequency f0
tx_signal = I_comp * np.cos(2*pi*f0*time) - Q_comp * np.sin(2*pi*f0*time)

# ----------------------------------------------------------------
#                         CHANNEL
# ----------------------------------------------------------------

# (A) DOPPLER SHIFT
# Instead of transmitting exactly at f0, 
# the channel adds doppler_shift => signal is effectively at f0 + doppler_shift
tx_signal_doppler = I_comp * np.cos(2*pi*(f0 + doppler_shift)*time) \
                  - Q_comp * np.sin(2*pi*(f0 + doppler_shift)*time)

# (B) ADDITIVE NOISE
noise = np.random.normal(0, noise_std_dev, len(tx_signal_doppler))
rx_signal = tx_signal_doppler + noise

# (C) NOISE POWER LEVEL (Optional diagnostic)
measured_noise_power = np.mean(noise**2)
print(f"Measured noise power in the channel: {measured_noise_power:.4f}")

# ----------------------------------------------------------------
#                        RECEIVER
# ----------------------------------------------------------------

# We must compensate for doppler_shift by using a local oscillator at f0 + doppler_est

# (1) DEMOD OF I AND Q
# "Down-conversion": multiply by cos((f0 + doppler_est)*t) and sin((f0 + doppler_est)*t)
r_cos = rx_signal * np.cos(2*pi*(f0 + doppler_est)*time)
r_sin = -rx_signal * np.sin(2*pi*(f0 + doppler_est)*time)  # minus for Q arm

# (2) LP FILTERING
cutoff_Hz = baud  # simplistic choice
I_filt = lowpass_filter(r_cos, cutoff_Hz, fs)
Q_filt = lowpass_filter(r_sin, cutoff_Hz, fs)

# (3) DETECTION FILTER (integrate-and-dump)
I_filt = I_filt.reshape((-1, Ns))
Q_filt = Q_filt.reshape((-1, Ns))

I_sym = np.sum(I_filt, axis=1)
Q_sym = np.sum(Q_filt, axis=1)

# (4) CONVERSION ANALOGUE -> DISCRETE
rx_symbols_baseband = I_sym + 1j * Q_sym

# QPSK decision
rx_bits = demodulate_qpsk_symbols(rx_symbols_baseband)
rx_body_bits = rx_bits[:len(body_bits)]  # remove padding if any

# ----------------------------------------------------------------
#          RECOMBINE HEADER + BODY BITS AND WRITE OUTPUT
# ----------------------------------------------------------------
header_rx_bits = header_bits.copy()
combined_bits = np.concatenate((header_rx_bits, rx_body_bits))
packed_bytes = np.packbits(combined_bits)

filename_out = 'output_' + filename_in
with open(filename_out, "wb") as f_out:
    f_out.write(packed_bytes.tobytes())

print(f"Saved to {filename_out}.")
print(f"Header protected = {header_size} bytes. Body with noise = {len(body_data)} bytes.")

# Optional: Check bit errors if you want (only for the body)
bit_errors = np.sum(body_bits != rx_body_bits)
print(f"Doppler Shift = {doppler_shift} Hz, Doppler Estimate = {doppler_est} Hz")
print(f"Bit Errors in body: {bit_errors}")
