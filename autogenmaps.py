import googlemaps
from datetime import datetime, timedelta
import re 


API_KEY = 'AIzaSyBNrLBdMT3ueFOcDVEQajH60zb6nVYrWRI'
gmaps = googlemaps.Client(key=API_KEY)

origin = "2/31 Stevens Street, Yeronga QLD 4104, Australia"
destination = "Southport park shopping centre, Gold Coast, QLD, Australia"

directions = gmaps.directions(origin, destination, mode = "driving")




if directions: 
    route = directions[0]['legs'][0]
    distance = route['distance']['text']
    duration = route['duration']['text']
    start_address = route['start_address']
    end_address = route['end_address']
    


    match = re.search(r"(?:(\d+)\s*hour)?\s*(?:(\d+)\s*min)?", duration)
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    now = datetime.now()
    arrival_time = now + timedelta(hours=hours, minutes=minutes)
    
    print(duration)
    #print(f"Distance: {distance}")
    #print(f"Duration: {duration}")
    #print(f"From: {start_address}")
    #print(f"To: {end_address}")
    print(arrival_time)
else:
    print("Nothing found")


