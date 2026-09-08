# -*- coding: utf-8 -*-
"""
sign_info.py
============
Shared lookup of the 43 German traffic-sign classes of the GTSRB dataset
(class id == GTSRB folder/label id, 0..42).

Each entry: {
    "en":   short English name,
    "de":   German sign name (what is printed on / known as the sign),
    "note": one-line meaning / driving rule
}

Used by train_model.py (to label artifacts) and app.py (to explain the
top predictions to the user).
"""

CLASS_INFO = {
    0:  {"en": "Speed limit 20 km/h", "de": "Tempo-20-Schild", "note": "Maximum speed 20 km/h."},
    1:  {"en": "Speed limit 30 km/h", "de": "Tempo-30-Schild", "note": "Maximum speed 30 km/h (often in residential zones)."},
    2:  {"en": "Speed limit 50 km/h", "de": "Tempo-50-Schild", "note": "Maximum speed 50 km/h (standard in built-up areas)."},
    3:  {"en": "Speed limit 60 km/h", "de": "Tempo-60-Schild", "note": "Maximum speed 60 km/h."},
    4:  {"en": "Speed limit 70 km/h", "de": "Tempo-70-Schild", "note": "Maximum speed 70 km/h."},
    5:  {"en": "Speed limit 80 km/h", "de": "Tempo-80-Schild", "note": "Maximum speed 80 km/h."},
    6:  {"en": "End of speed limit 80 km/h", "de": "Ende der Tempo-80-Zone", "note": "The 80 km/h limit ends here."},
    7:  {"en": "Speed limit 100 km/h", "de": "Tempo-100-Schild", "note": "Maximum speed 100 km/h."},
    8:  {"en": "Speed limit 120 km/h", "de": "Tempo-120-Schild", "note": "Maximum speed 120 km/h."},
    9:  {"en": "No passing", "de": "Überholverbot", "note": "Overtaking other vehicles is forbidden."},
    10: {"en": "No passing for trucks", "de": "Überholverbot für Lkw", "note": "Trucks > 3.5 t may not overtake."},
    11: {"en": "Right-of-way at next intersection", "de": "Vorfahrt an der nächsten Kreuzung", "note": "You have right of way at the next intersection."},
    12: {"en": "Priority road", "de": "Vorfahrtstraße", "note": "Yellow diamond — right of way along this road."},
    13: {"en": "Yield", "de": "Vorfahrt gewähren", "note": "Inverted triangle — give way to cross traffic."},
    14: {"en": "Stop", "de": "Stoppschild", "note": "Red octagon — come to a full stop, then yield."},
    15: {"en": "No vehicles", "de": "Verbot für Fahrzeuge", "note": "All motor vehicles prohibited."},
    16: {"en": "No trucks", "de": "Verbot für Lkw", "note": "Vehicles over 3.5 t prohibited."},
    17: {"en": "No entry", "de": "Verbot der Einfahrt", "note": "Red circle, white bar — do not enter."},
    18: {"en": "General caution", "de": "Gefahrstelle", "note": "Generic danger warning ahead."},
    19: {"en": "Dangerous curve left", "de": "Linkskurve", "note": "Sharp curve to the left ahead."},
    20: {"en": "Dangerous curve right", "de": "Rechtskurve", "note": "Sharp curve to the right ahead."},
    21: {"en": "Double curve", "de": "Doppelkurve", "note": "Two consecutive curves ahead."},
    22: {"en": "Bumpy road", "de": "Unebene Fahrbahn", "note": "Rough / uneven road surface ahead."},
    23: {"en": "Slippery road", "de": "Schleudergefahr", "note": "Risk of skidding when wet or icy."},
    24: {"en": "Road narrows on right", "de": "Verengte Fahrbahn (rechts)", "note": "Carriageway narrows from the right."},
    25: {"en": "Road work", "de": "Baustelle", "note": "Construction / road works ahead."},
    26: {"en": "Traffic signals", "de": "Ampel", "note": "Traffic light (signal) ahead."},
    27: {"en": "Pedestrians", "de": "Achtung Fußgänger", "note": "Pedestrians may cross the road ahead."},
    28: {"en": "Children crossing", "de": "Achtung Kinder", "note": "Children may cross — school zone warning."},
    29: {"en": "Bicycles crossing", "de": "Radfahrer kreuzen", "note": "Cyclists may cross the road ahead."},
    30: {"en": "Beware of ice/snow", "de": "Schnee- oder Eisglätte", "note": "Danger of snow or ice on the road."},
    31: {"en": "Wild animals crossing", "de": "Wildwechsel", "note": "Deer / wildlife may cross the road."},
    32: {"en": "End of all speed & passing limits", "de": "Ende sämtlicher Verbote", "note": "All speed limits and passing bans end."},
    33: {"en": "Turn right ahead", "de": "Vorgeschriebene Fahrtrichtung rechts", "note": "Mandatory direction: turn right."},
    34: {"en": "Turn left ahead", "de": "Vorgeschriebene Fahrtrichtung links", "note": "Mandatory direction: turn left."},
    35: {"en": "Ahead only", "de": "Fahrt geradeaus", "note": "Blue circle, white arrow up — drive straight ahead."},
    36: {"en": "Go straight or right", "de": "Geradeaus oder rechts", "note": "Mandatory direction: straight or right."},
    37: {"en": "Go straight or left", "de": "Geradeaus oder links", "note": "Mandatory direction: straight or left."},
    38: {"en": "Keep right", "de": "Rechts vorbeifahren", "note": "Pass / keep to the right side of the obstacle."},
    39: {"en": "Keep left", "de": "Links vorbeifahren", "note": "Pass / keep to the left side of the obstacle."},
    40: {"en": "Roundabout mandatory", "de": "Kreisverkehr", "note": "Blue circle — roundabout ahead; yield inside."},
    41: {"en": "End of no passing", "de": "Ende des Überholverbots", "note": "Overtaking ban for cars ends."},
    42: {"en": "End of no passing for trucks", "de": "Ende des Überholverbots für Lkw", "note": "Overtaking ban for trucks ends."},
}
