import database_utils
import os

database_utils.populate_database(
    database_path="database", 
    id_path="songids", 
    song_paths=["data/Reference MP3s/" + file for file in os.listdir("data/Reference MP3s/")][:2], 
    song_names=[file[:-4] for file in os.listdir("data/Reference MP3s/")][:2], 
    artist_names=["The Weeknd", "Wolfgang Amadeus Mozart", "Frank Sinatra", "Ludwig van Beethoven", "Eagles", "Nirvana", "Kanye West", "Rihanna", "Louis Armstrong", "Taylor Swift"][:2]
)