import numpy as np
import matplotlib.pyplot as plt
import struct

from numpy import pi, cos, sin, sqrt, r_
from scipy.fftpack import fft

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
    The nominal angles are 45°, 135°, 225°, 315° (i.e., diagonals).
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
    rx_symbols: complex QPSK symbols
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
#                   MAIN SIMULATION CODE
# ----------------------------------------------------------------

# 1. READ THE FILE
filename_in = "chat_trognon.jpg"
with open(filename_in, "rb") as f:
    file_data = f.read()

# 2. SPLIT THE FILE INTO HEADER (PROTECTED) + BODY (NOISY)
#    Adjust header_size as you see fit (e.g., 512, 1024, etc.).
header_size = 1024  
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
noise_std_dev = 0.25  # Increase noise here

# 5. PROTECT THE HEADER
#    We'll skip the header from the QPSK + noise path.
#    Instead, we either keep them unmodified or send them through a no-noise chain.

### If you want to transmit the header with *no noise* for demonstration,
### you can do the following:
# header_rx_bits = header_bits.copy()
# (Alternatively, run it through QPSK with zero noise.)

# For simplicity, let's just keep the header bits as-is:
header_rx_bits = header_bits.copy()

# 6. TRANSMIT & RECEIVE THE BODY BITS THROUGH QPSK (NOISY)

# Ensure an even number of bits (QPSK = 2 bits/symbol)
if len(body_bits) % 2 != 0:
    body_bits = np.append(body_bits, 0)  # pad with one zero if needed

# 6a. Convert bits to QPSK symbols (ideal, no noise initially)
# 1) Group bits into pairs -> symbol index
symbols = []
for i in range(0, len(body_bits), 2):
    b1 = body_bits[i]
    b2 = body_bits[i+1]
    sym_idx = GetQpskSymbol(b1, b2)
    symbols.append(sym_idx)
symbols = np.array(symbols, dtype=int)

# 2) Map symbols to complex points with noise
noise1 = np.random.normal(0, noise_std_dev, len(symbols))
noise2 = np.random.normal(0, noise_std_dev, len(symbols))
tx_symbols = []
for i, sym in enumerate(symbols):
    tx_symbols.append(
        QpskSymbolMapper(sym, amp_I, amp_Q, noise1=noise1[i], noise2=noise2[i])
    )
tx_symbols = np.array(tx_symbols, dtype=complex)

# 6b. Demodulate 
rx_bits = demodulate_qpsk_symbols(tx_symbols)

# 7. RECOMBINE HEADER + BODY BITS
#    We used 'header_rx_bits' as the "protected" header
#    and 'rx_bits' as the noisy body.
#    If we padded one bit, we remove it from the tail:
rx_body_bits = rx_bits[:len(body_bits)]
combined_bits = np.concatenate((header_rx_bits, rx_body_bits))

# 8. PACK BITS BACK INTO BYTES
packed_bytes = np.packbits(combined_bits)

# 9. WRITE THE RESULTING FILE
filename_out = 'output'+filename_in
with open(filename_out, "wb") as f_out:
    f_out.write(packed_bytes.tobytes())

print(f"Saved to {filename_out}.")
print(f"Header protected = {header_size} bytes. Body with noise = {len(body_data)} bytes.")
