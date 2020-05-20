from read_file import read_file
from get_carac import get_carac
from write_file_carac import write_file_carac

filename = 'Spots in tracks statistics.csv'
X,Y,T,TRACK_ID = read_file(filename)
CARAC_WHO = get_carac(X,Y,T,min_detections=3)
write_file_carac(CARAC_WHO)