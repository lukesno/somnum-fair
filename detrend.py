import cv2
import dlib
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from database import send_data_to_db

INPUT_VIDEO = "cheeks_output.avi"

def extract_green_channel_data_and_create_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), False)  # False for grayscale output
    
    green_values = []  # To store the average green values of each frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract the green channel
        green_channel = frame[:, :, 1]
        
        # Write the green channel to the output video
        out.write(green_channel)
        
        # Calculate the average green value for the current frame
        avg_green_value = np.mean(green_channel)
        green_values.append(avg_green_value)
        
    cap.release()
    out.release()
    
    return green_values

def moving_average(signal, window_size):
    cumsum_vec = np.cumsum(np.insert(signal, 0, 0)) 
    ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    # To align with the original signal size, pad the moving average array
    ma = np.pad(ma, (window_size-1, 0), 'edge')
    return ma

def detrend_signal(signal, N):
    detrended_signal = np.zeros_like(signal)

    # Loop through each frame in the signal
    for t in range(N):
        # Calculate the start and end of the window
        window_start = max(0, t - 15)
        window_end = min(N, t + 15 + 1)  # Add 1 because the upper bound is exclusive in Python

        # Calculate the moving average for the window
        moving_avg = np.mean(signal[window_start:window_end])

        # Detrend the signal at frame t
        detrended_signal[t] = signal[t] - moving_avg

    return detrended_signal

def smooth_signal(signal, window_size=5):
    return moving_average(signal, window_size)


def plot_signals(raw_signal, detrended, smoothed):
    plt.figure(figsize=(14, 7))

    # Plot raw green channel values
    plt.subplot(3, 1, 1)
    plt.plot(raw_signal, label='Raw Signal')
    plt.title('Raw Green Channel Values')
    plt.xlabel('Frame Number')
    plt.ylabel('Value')
    plt.legend()

    # Plot detrended signal
    plt.subplot(3, 1, 2)
    plt.plot(detrended, label='Detrended Signal', color='orange')
    plt.title('Detrended Signal')
    plt.xlabel('Frame Number')
    plt.ylabel('Value')
    plt.legend()

    # Plot smoothed signal
    plt.subplot(3, 1, 3)
    plt.plot(smoothed, label='Smoothed Signal', color='green')
    plt.title('Smoothed Signal')
    plt.xlabel('Frame Number')
    plt.ylabel('Value')
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()




def calculate_fft_and_graph(green_channel_data, fps):
    # The length of the data
    N = len(green_channel_data)
    
    # Subtract the mean to detrend the data and remove the DC component
    green_channel_data_detrended = green_channel_data - np.mean(green_channel_data)
    
    # Perform the FFT on the detrended data to find the frequency components
    yf = fft(green_channel_data_detrended)
    
    # Generate the frequency bins for the x-axis of the graph
    # This gives us a range from 0 to the Nyquist frequency (half the sampling rate)
    xf = fftfreq(N, 1 / fps)[:N//2]
    
    # Calculate the magnitude of the FFT, which represents the contribution of each frequency
    yf_magnitude = np.abs(yf[:N//2])
    
    # Create a plot with appropriate labels and titles
    plt.figure(figsize=(14, 7))
    plt.plot(xf, yf_magnitude)
    plt.title('FFT of Green Channel Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    # Highlight the peaks using the find_peaks function
    peaks, _ = find_peaks(yf_magnitude, height=np.max(yf_magnitude)/4)
    plt.plot(xf[peaks], yf_magnitude[peaks], "x", label="Detected Peaks", color='red')
    
    # Display the legend and show the plot
    plt.legend()
    plt.show()

    # Return the frequencies and magnitudes for further analysis if necessary
    return xf, yf_magnitude


def calculate_heart_rate(signal, frame_rate, window_size=10, shift_size=1, zero_padding=450):
    # Calculate the number of frames to move for each second
    frames_per_second = frame_rate
    shift_frames = int(frames_per_second * shift_size)

    # The number of samples in each moving window
    window_frames = int(frames_per_second * window_size)

    heart_rates = []
    for start in range(0, len(signal) - window_frames + 1, shift_frames):
        # Extract the windowed signal
        windowed_signal = signal[start:start + window_frames]

        # Normalize the signal as per Equation 4.2 (Normalization step to be added)
        # windowed_signal = normalize(windowed_signal)  # You need to define the normalize function based on your equation

        # Zero-pad the signal
        zero_padded_signal = np.pad(windowed_signal, (0, zero_padding - window_frames), 'constant')

        # Compute the FFT
        fft_result = np.abs(np.fft.rfft(zero_padded_signal))
        print("fft_result")
        print(fft_result)
        # Find the peaks in the FFT result
        peaks, _ = find_peaks(fft_result)
        peak_frequencies = peaks * (frame_rate / zero_padding)
        valid_peaks = peaks[(peak_frequencies >= 0.8) & (peak_frequencies <= 3)]
        # Assuming the first peak corresponds to the heart rate (this may need adjustment based on your data)
        # if peaks.size > 0:
        #     # Convert peak location to bpm using the frequency resolution
        #     heart_rate_frequency = peaks[0] * (frame_rate / zero_padding)
        #     heart_rate_bpm = heart_rate_frequency * 60  # Convert Hz to bpm
        #     heart_rates.append(heart_rate_bpm)

        # Use highest peak instead
        if valid_peaks.size > 0:
            # Convert peak location to bpm using the frequency resolution
            highest_peak = valid_peaks[np.argmax(fft_result[valid_peaks])]
            heart_rate_frequency = highest_peak * (frame_rate / zero_padding)
            heart_rate_bpm = heart_rate_frequency * 60  # Convert Hz to bpm
            heart_rates.append(heart_rate_bpm)


    return heart_rates


def main():
    f_name = input("Input participant's first name: ")
    l_name = input("Input participant's last name: ")
    
    green_channel_values = extract_green_channel_data_and_create_video(INPUT_VIDEO, "green2.avi")
    
    raw_signal = np.array(green_channel_values)  
    detrended = detrend_signal(raw_signal, len(raw_signal))
    smoothed = smooth_signal(detrended)
    
    plot_signals(raw_signal, detrended, smoothed)
    calculate_fft_and_graph(smoothed, 30)
    
    heart_rates = calculate_heart_rate(smoothed, 30)
    print("Heart rates:", heart_rates)

    decision = input("Send results?")
    if decision == "1":
        start_time_unix = time.time()
        send_data_to_db(f_name, l_name, start_time_unix, heart_rates)
if __name__ == "__main__":
    main()
