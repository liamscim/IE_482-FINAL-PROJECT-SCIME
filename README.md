# IE_482-FINAL-PROJECT-SCIME
Sound Localization with 2 microphones

# Motivation / Overview of your project.
- Using 2 USB microphones identify which mic is closer to the sound and approximate location of sound
  - Only the distance can be determined using the two mic configuration
- Could be used for locating people after a natural disaster
# Demonstration
[![Watch the video](https://i.sstatic.net/Vp2cE.png)](https://ubuffalo-my.sharepoint.com/:v:/g/personal/liamscim_buffalo_edu/Ea9lNS5nJ91HnZax4GvgdfMBv3CYnCt4u14b-YSAlq9Zhw?e=xUvaNd)
- Download for this is also available in this repo, this should work however!
# Installation Instructions
`NOTE` This has only been tested and utilized on windows 11

First the following packages need to be pip installed
```
pip install sounddevice
```
- For multichannel recording input
```
pip install numpy
```
- for numerical operations and array handling
```
pip install matplotlib
```
- for plotting the audio signals and other data
```
pip install scipy
```
- for signal processing (cross-correlation, etc.)
```
pip install soundfile
```
- for reading and writing audio files
```
pip install pyaudio
```
- For direct audio recording

The rest is just python commands that can be ran within a jupyter notebook
- This code will be highlighted in the next section
# How to Run the Code
## Code 1 - Index Verification
Run this code to see what indices the USB mics are in:
```
import sounddevice as sd

print(sd.query_devices())
```
Sample output:

![Code 1 output](https://github.com/user-attachments/assets/c518e951-7cab-46b7-aef5-e4a146849350)

Next run this code and speak/blow into one mic only, this was my way of verifying the index of the mics:
```
#Speak into one mic only this will help identify which mic is which
#LEFT MIC HAS WHITE DOT
#VERIFY MIC IDECIES

import pyaudio
import numpy as np

p = pyaudio.PyAudio()
fs = 44100  # Sample rate
chunk = 1024
seconds = 2  # Short test recording

for mic_index in [1, 2]:  # Replace with actual device indices
    print(f"Testing Microphone {mic_index}. Speak into it now...")
    
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, input_device_index=mic_index, frames_per_buffer=chunk)
    
    frames = []
    for _ in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    
    stream.stop_stream()
    stream.close()

    # Convert to numpy array and print summary
    audio_data = np.concatenate(frames)
    print(f"Microphone {mic_index} Average Amplitude: {np.mean(np.abs(audio_data))}\n")

p.terminate()
```
- Run this and speak directly into one mic only, the mic you were speaking into should have the louder amplitude which will allow you to verify if the indeices for the mics are correct

Sample output:

![Code 1b output](https://github.com/user-attachments/assets/276ce6dc-2009-4b14-945a-7bdb0e3638fe)
You can see here I was blowing into the left mic and therefore the left mic is the USB mic in index 1.

## Code 2 - Sound Recording
- Run the following code to record ~ 1 sec sound clip from each mic at the same time
```
#Record sound clip from both mics at the same time

import sounddevice as sd
import soundfile as sf
import numpy as np
import time

# Replace these with the correct indices you found
LEFT_MIC_INDEX = 1
RIGHT_MIC_INDEX = 2

DURATION = 1.0  # seconds
SAMPLERATE = 44100

# Buffer to hold both mic recordings
recordings = {}

# Record from both devices using threads
def record_from_device(index, key):
    recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, device=index, dtype='float32')
    sd.wait()
    recordings[key] = recording.flatten()

# Start both recordings nearly simultaneously
from threading import Thread

t1 = Thread(target=record_from_device, args=(LEFT_MIC_INDEX, 'left'))
t2 = Thread(target=record_from_device, args=(RIGHT_MIC_INDEX, 'right'))

start_time = time.time()
t1.start(); t2.start()
t1.join(); t2.join()
print(f"Recording finished. Elapsed: {time.time() - start_time:.3f}s")

# Save to .wav files
sf.write('mic_left.wav', recordings['left'], SAMPLERATE)
sf.write('mic_right.wav', recordings['right'], SAMPLERATE)
print("Saved mic_left.wav and mic_right.wav")
```
- This will save the two recording files to the folder where this notebook is saved
- IMPORTANT: Make sure mic indecies are correct!

## Code 3 - Correction Factor
- This code is used to ensure that the sound clips from each mic are in line
  - This is very important as since we are using TDOA from the peak amplitudes to determine which mic is closer and the distance from the sound source its critcal that the records are aligned exactly
- Record a single clap from the middle of the mics (code 2), the closer to directly center the better
```
#Determine correction factor
#To do so clap directly between the mics then run this chunk of code
#Manually add correlation factor into the rest of the code (this way it wont needed to be changed every time you run the code)

import numpy as np
import soundfile as sf
from scipy.signal import correlate

# Load recordings
left, sr = sf.read('mic_left.wav')
right, _ = sf.read('mic_right.wav')

# Ensure same length
min_len = min(len(left), len(right))
left = left[:min_len]
right = right[:min_len]

# Normalize both signals
left = (left - np.mean(left)) / np.std(left)
right = (right - np.mean(right)) / np.std(right)

# Cross-correlation to estimate delay
corr = correlate(left, right, mode='full')
lags = np.arange(-len(left) + 1, len(right))
offset = lags[np.argmax(corr)]

# Print the correction factor
print(f"Estimated correction factor (manual_offset_samples) = {offset} samples")

#postive value means left lags behind right
#Negative value means right lags behind left
```
- Run this code and it will tell you the correction factor
  - Postive factor means the left mic is lagging, negative means that the right mic is lagging
  - Mine correction factor was consistently around ~257 meaning the left mic was lagging
- `NOTE` Even when mics are flipped the left seemed to still be delayed. This delay could be from the code itself or even my laptop but it doesn't seem to be due to the mics themselves.

Sample output:

![Code 3 output](https://github.com/user-attachments/assets/9de1b805-3fb5-4521-aa71-c398a506fc34)

## Code 4 - Plots from each mic
```
import matplotlib.pyplot as plt

# Manually set correction in samples (positive = left mic delayed)
manual_offset_samples = 257  # example value, adjust as needed

# Apply correction
if manual_offset_samples > 0:
    left_corr = left[manual_offset_samples:]
    right_corr = right[:len(left_corr)]
elif manual_offset_samples < 0:
    right_corr = right[-manual_offset_samples:]
    left_corr = left[:len(right_corr)]
else:
    left_corr = left
    right_corr = right

# Find peaks
left_peak_corr = np.argmax(np.abs(left_corr))
right_peak_corr = np.argmax(np.abs(right_corr))

# Plot Left mic (corrected)
plt.figure(figsize=(12, 4))
plt.plot(left_corr, label='Left mic (corrected)', alpha=0.7, color='blue')
plt.scatter(left_peak_corr, left_corr[left_peak_corr], color='red', label=f"Left Peak: {left_peak_corr}")
plt.legend()
plt.title("Corrected Left Microphone Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Plot Right mic (corrected)
plt.figure(figsize=(12, 4))
plt.plot(right_corr, label='Right mic (corrected)', alpha=0.7, color='red')
plt.scatter(right_peak_corr, right_corr[right_peak_corr], color='blue', label=f"Right Peak: {right_peak_corr}")
plt.legend()
plt.title("Corrected Right Microphone Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Print Peak Index Locations
print(f"Left Peak Index: {left_peak_corr}")
print(f"Right Peak Index: {right_peak_corr}")
```
- The following code will plot the amplitude from each mic on two seperate plots
- Make sure the manual_offset_samples is updated with the proper correction factor, use 0 if you dont want a correction factor
- Although the correction factor is something you will manually change you still need to run that code (code 3) for the plots to be updated
Sample output:

![Code 4a](https://github.com/user-attachments/assets/bc54b413-e109-4e2c-9f24-62c74f0a4458)

The following will plot the amplitudes from each mic overlayed
```
import matplotlib.pyplot as plt
import numpy as np

# Ensure left_corr and right_corr are same length
min_len = min(len(left_corr), len(right_corr))
left_trimmed = left_corr[:min_len]
right_trimmed = right_corr[:min_len]

# Create x-axis
x = np.arange(min_len)

# Define overlap threshold (amplitude difference)
threshold = 100  # Adjust as needed for your signal scale

# Boolean mask for overlap
overlap_mask = np.abs(left_trimmed - right_trimmed) < threshold

# Plotting
plt.figure(figsize=(14, 5))
plt.plot(x, left_trimmed, label='Left Mic', color='blue', alpha=0.5)
plt.plot(x, right_trimmed, label='Right Mic', color='red', alpha=0.5)

# Highlight overlapping samples
plt.plot(x[overlap_mask], left_trimmed[overlap_mask], 'o', color='purple', markersize=2, label='Overlap')

plt.title("Left and Right Microphone Signals with Overlap Highlighted")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
Sample output:

![Code 4b](https://github.com/user-attachments/assets/79aa066c-9667-4fc7-ad45-6eac9c6d1755)

## Code 5 - Final Results
- This code will tell you which mic is closer
  - You can infer which mic is closer using the plots as whichever mic has the peak amplitude sooner is the mic that was closer to the sound
```
# Apply manual correction
manual_offset_samples = -267  # replace `offset` with your saved value Negative shifts left index back
left_corr = np.roll(left, manual_offset_samples)
right_corr = right

# Peak detection after manual correction
left_peak_corr = np.argmax(np.abs(left_corr))
right_peak_corr = np.argmax(np.abs(right_corr))

# Constants
SAMPLERATE = 48000
v_sound_cm_s = 34300
mic_distance_cm = 37.465

# Time Difference of Arrival (based on corrected peak indices)
tdoa = (left_peak_corr - right_peak_corr) / SAMPLERATE
distance_diff_cm = round(tdoa * v_sound_cm_s)  # rounded to nearest cm

# Determine which mic is closer
if left_peak_corr < right_peak_corr:
    closer_mic = "Left"
elif right_peak_corr < left_peak_corr:
    closer_mic = "Right"
else:
    closer_mic = "Same"

# Output
print(f"Corrected Left Peak Index: {left_peak_corr}, Right Peak Index: {right_peak_corr}")
print(f"Sound is closer to the {closer_mic} mic.")
print(f"Estimated distance difference: {abs(distance_diff_cm)} cm")
```
- Ensure that mic_distance_cm is the correct spacing between the mics
- This distance is only accurate if the sound is inbetween the two mics as techinally if its outside the two mics the model should return the distance between the 2 mics (the width of laptop, in my case 37.465cm)
- Make sure that manual_offset_samples is updated. In this code the sign needs to be flipped
  - So if the correction factor is 257 meaning the left mic recording is lagging than the factor here needs to be negative 257

Sample output:

![Code 5](https://github.com/user-attachments/assets/fe3ac0d7-2276-4d27-aeaf-c7a39ded54b4)

**This code is available under Localization_Final_Iteration, A version that continouly outputs these reults is available under Localization_Final_Iteration_Live**
- The live version of the code just doesn't seem to be as accurate, still uses correction factor
- The live version is also just hard to keep up with as its outputing alot of data fast

## Other Cool Code
### Sound Classification with YAMNet
Need:
```
pip install sounddevice tensorflow tensorflow-hub numpy scipy
```
Then run the following code:
```
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import csv
import time
import threading

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class names
def class_names_from_csv(class_map_csv_path):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

# Set recording parameters
duration = 2  # seconds
fs = 16000  # Hz

# Control flag
keep_running = True

# Classification loop
def classify_loop():
    global keep_running
    print("Starting classification loop...")
    while keep_running:
        print("\nRecording...")
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        waveform = tf.reshape(audio_data, [-1])
        scores, embeddings, spectrogram = model(waveform)

        mean_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.argmax(mean_scores).numpy()

        print("Predicted class:", class_names[top_class])
        time.sleep(0.5)

    print("Classification loop stopped.")

# Start the thread
classify_thread = threading.Thread(target=classify_loop)
classify_thread.start()

# Stop function you can call manually
def stop_classification():
    global keep_running
    keep_running = False
    classify_thread.join()
    print("Stopped classification.")
```
To stop code:
```
stop_classification()
```
- This will predict what the noise is, this was cool and actually quite accurate but slows down the localization code so thats why it wasn't integrated

![Sounds](https://github.com/user-attachments/assets/34d816bb-c5c8-42ca-8504-deaf814a8581)
- Will predict was sound is continously every 3 seconds until the stop command is ran
# Note
## Angle Estimation:
Angle estimation wont work with this model, this is because the model is assuming the distance is in the same horizonal line that the mics are on. The orginal idea was to draw two circles with the radius equal to how far the mic is away from the sound but say the closest mic is 7cm from the source of the sound, the farther mic is going to be 7 cm + the distance between the mics (the width of my laptop) which is 37.465. This means when the circles are put around the mics they will only ever intersect at one point that lies on the horizontal line connecting the mics. Below was the graph i was going to make pointing the 2 possible points of intersections of the circles. But as you can see it will only ever pick one point.

### Example image
![Angle Graph](https://github.com/user-attachments/assets/b6877956-8665-4224-8a84-5687a93089c5)

# References
YAMNet
- [Sound Classification](https://www.tensorflow.org/hub/tutorials/yamnet)

Not super helpful but cool
- [Acoustic Localization](https://opensoundscape.org/en/latest/tutorials/acoustic_localization.html)
  - Could not get the localization package within opensoundscape to import
# Future Work
- Ideally I would like the code to be able to identify a rotation factor and distance that way we could do something like track the sound
- Recording that is always updating the location of the sound would be cool, would make this more practical
- Having someway where the correction factor can be configured automatically would be cool and make this code easier to run
- Implement 3rd mic for easier localization
