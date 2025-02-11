"""Module containing useful database functions."""
import pickle
from typing import Union
import ConvertAudioRecordings
import RandomSamples
import audio_processing
import database_keyfunctions
import database_utils


def save_database(database: Union[dict, list], file_path: str):
    """
    Saves the database to file_path using pickle.

    Parameters
    ----------
    database: dict | list
        Dictionary or list storing relevant song or fingerprint data
    file_path: str
        A file path like string denoting the destination of the save
    """
    with open(file_path, "wb") as file:
        pickle.dump(database, file)

def load_database(file_path: str) -> Union[dict, list]:
    """
    Loads a database stored at file_path using pickle.

    Parameters
    ----------
    file_path: str
        A file path like string denoting the source of the load
        
    Returns
    ----------
    dict | list
        The loaded database as a dictionary or list object
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)
    
def get_info(id_to_info: list):
    """
    Prints the number of total songs and the unique artists 
    in the database. Each song and artist is printed individually.

    Parameters
    ----------
    id_to_info: list
        A list where the kth index is the (song name, artist name) tuple of the song with song ID of k
    """
    song_database = [item[0] for item in id_to_info]
    artist_database = [item[1] for item in id_to_info]
    print("Total songs in the database:    {}".format(len(song_database)))
    print("Unqiue artists in the database: {}".format(len(set(artist_database))))
    
    print("\nComplete list of songs:")
    for song_id, (song, artist) in enumerate(id_to_info):
        print("\tID: {:>4} - {:>25} by {:>25}".format(song_id, song, artist))
        
    print("\nComplete list of artists:")
    for artist in sorted(set(artist_database)):
        print("\t" + artist)
    
from collections import defaultdict
def populate_database(database_path: str, id_path: str, song_paths: list, song_names: list, artist_names: list):
    database = defaultdict(list)
    id_to_info = []
    
    # FOR PATH IN SONG_PATHS
    for i, (path, song_name, artist_name) in enumerate(zip(song_paths, song_names, artist_names)):
        recorded_audio, sampling_rate = ConvertAudioRecordings.samples_file(path) # NumPy array of samples
        
        # Generate random samples of audio
        samples = RandomSamples.random_samples(recorded_audio, sampling_rate, 20, 5)
        
        for sample in samples:
            # Generate spectrogram
            spectrogram, freqs, times = audio_processing.produce_spectrogram(recorded_audio, sampling_rate=sampling_rate)

            # Extract peaks
            local_peaks_ind = audio_processing.extract_local_peak_idxs(spectrogram)

            # Generate fingerprint
            fingerprints = audio_processing.form_fingerprints(local_peaks_ind)
    
            # Append fingerprint to database
            database_keyfunctions.store_fingerprint(fingerprints, i, database)
            
        # Append song metadata to database
        id_to_info.append((song_name, artist_name))
            
    # Save databases to file_path
    database_utils.save_database(database, database_path)
    database_utils.save_database(id_to_info, id_path)
    
