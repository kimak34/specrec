# Writing the core functionality for storing fingerprints in the database, as well as querying the database and tallying the results of the query.

"""
Need to maintain a list of associated song-IDs 
for all of the songs that a given peak-pair occurred in, 
along with the absolute time associated with each fanout pattern 
"""

def store_fingerprint(fingerprints, songID: int, database):
    for fingerprint in fingerprints:
        for (fm, fn, dt), ts in fingerprint:
            database[(fm, fn, dt)].append((songID, ts))
            
            
from collections import defaultdict

def query_database(fingerprints, database, songIDs):
    # dict for keeping track of tallies
    tallies = defaultdict(int)

    # for each fingerprint, add a tally to its specific key and time offset
    for fingerprint in fingerprints:
        for (fm, fn, dt), tc in fingerprint:
            for songID, ts in database[(fm, fn, dt)]:
                t_offset = int(ts-tc)
                tallies[(songID, t_offset)] += 1

    
    # IF SONG IN DATABASE: Print song artist and title
    # IF SONG NOT IN DATABASE: Print that not matches were found
    if (max(tallies.values())==0):
        return "This song does not match any other song in the database"
    else:
        final_ID = max(tallies, key=tallies.get)[0]
        return f"The song that matches this is {songIDs[final_ID][0]} by {songIDs[final_ID][1]}." 
    
