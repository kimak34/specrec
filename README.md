# Song-Recognition
This project can classify a song from songs already in its database, either from a file or a microphone recording of a defined length.
## Process
### 1. Converting Audio Recordings
First, an audio recording is converted into a list of amplitude samples, either from a file or from the microphone (functions `samples_mic` and `samples_file`).
### 2. Audio Processing
A spectrogram of the amplitude samples is created with `produce_spectrogram`. Then, pairs of local peaks are found with `extract_local_peak_idxs`, and a fingerprint for the recording is formed with `form_fingerprints`.
### 3. Database
Peak pairs are stored in a python dictionary, with the key being the peak pair and the value being IDs of all the songs that contain that pair. We can add fingerprints into the database from songs we know with `store_fingerprint`, and when we're recognizing a song, we use `query_database` to keep tallies of which song in the database shares the most fingerprints with the song we're identifying .
### 4. Testing
We also created a function that can generate random clips of a larger track randomly (`random_samples`), and this can be used to evaluate the accuracy of our model.