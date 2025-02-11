import database_utils
import ConvertAudioRecordings
import audio_processing
import database_keyfunctions

# Load database
database = database_utils.load_database("database")
#songID list of Tuple("SongName", "SongArtist")
songIDs = database_utils.load_database("songids")

# Read type of input file
input_type = input("Please input what ('mic' or 'file') you want to record (\"q\" to quit): ")

while input_type not in ["q", "Q"]:
    
    # Read audio file to get Numpy array of samples and sampling_rate
    if (input_type == 'mic'):
        listen_time = input("How long would you like to record for: ")
        recorded_audio, sampling_rate = ConvertAudioRecordings.samples_mic(listen_time)
    else:
        path = input("Please input path to your recording (\"q\" to quit): ")
        recorded_audio, sampling_rate = ConvertAudioRecordings.samples_file(path)
    
    # Generate spectrogram
    spectrogram, freqs, times = audio_processing.produce_spectrogram(recorded_audio, sampling_rate=sampling_rate)
    
    # Extract peaks
    local_peaks_ind = audio_processing.extract_local_peak_idxs(spectrogram)
    
    # Generate fingerprint
    fingerprints = audio_processing.form_fingerprints(local_peaks_ind)
    
    # Query database and tally results
    result = database_keyfunctions.query_database(fingerprints, database, songIDs)

    # print result
    print(result)

    input_type = input("Please input what ('mic' or 'file') you want to record (\"q\" to quit): ")
