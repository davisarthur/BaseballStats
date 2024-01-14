import numpy as np

TEAMS = ['HOU', 'PHI', 'NYY', 'SD', 'CLE',
         'SEA', 'LAD', 'ATL', 'NYM', 'TOR',
         'STL', 'BAL', 'BOS', 'CIN', 'OAK',
         'PIT', 'TEX', 'CWS', 'MIA', 'MIL',
         'LAA', 'CHC', 'DET', 'SF', 'WSH',
         'MIN', 'AZ', 'COL', 'KC', 'TB']

EVENT_VALUE = {
    'single': 1,
    'double': 2,
    'triple': 3,
    'home_run': 4,
}

EVENT_COLOR = {
    'out': '#C200FB', # purple
    'single': '#EC0868', # raspberry
    'double': '#FC2F00', # scarlet
    'triple': '#EC7D10', # tangerine
    'home_run': '#FFBC0A', # yellow
}

EVENT_MARKER = {
    'out': 'x',
    'single': '>', # triangle right
    'double': '^', # triangle up
    'triple': '<', # triangle left
    'home_run': '*'
}

def horizontal_angle(hc_x, hc_y):
    '''
    Converts (hc_x, hc_y) statcast data to horizontal launch angle
    hc_x - x hit position from statcast data
    hc_y - y hit position from statcast data
    '''
    normalized_x, normalized_y = normalize_hc(hc_x, hc_y)
    return 180.0 * np.arctan2(normalized_x, normalized_y) / np.pi

def normalize_hc(hc_x, hc_y):
    '''
    Translates (hc_x, hc_y) statcast coordinates to coordinate system
    with home plate centered at 0 and right and left foul lines oriented
    at 45 and 135 degrees respectively
    '''
    normalized_x = hc_x - 128
    normalized_y = 200 - hc_y
    return (normalized_x, normalized_y)
