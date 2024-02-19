import numpy as np
from sklearn.preprocessing import StandardScaler

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

RESULT_NAME = {
    0: 'out',
    1: 'single',
    2: 'double',
    3: 'triple',
    4: 'home run',
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

def statcast_df_to_polynomial_data_matrix(df, scaler=None):
    '''
    Creates a data matrix from statcast batted balls data. All features in the data matrix
    are computed from the launch speed, vertical launch angle, and horizontal launch angle
    of the batted ball. The labels array indicates if batted ball was an out (0), single (1),
    double (2), triple (3), or home run (4).

    Returns (data_matrix, labels_array)
        data_matrix - 2D np array data matrix with shape (num_samples, num_features+1)
        labels_array - 1D np array with length num_samples
        scaler - scaler fitted to data if none. Returns the input scaler otherwise
    '''
    launch_speed = df['launch_speed'].to_numpy()
    vertical_launch_angle = df['launch_angle'].to_numpy()
    horizontal_launch_angle = df['horizontal_angle'].to_numpy()
    X, scaler = statcast_raw_data_to_polynomial_matrix(launch_speed, vertical_launch_angle, horizontal_launch_angle, scaler)
    Y = df['result'].to_numpy()
    return X, Y, scaler

def statcast_raw_data_to_polynomial_matrix(launch_speed, vertical_launch_angle, horizontal_launch_angle, scaler=None):
    '''
    Creates a polynomial data matrix from statcast launch speed, vertical launch angle, horizontal angle.

    Returns (data_matrix, labels_array)
        data_matrix - 2D np array data matrix with shape (num_samples, num_features+1)
        scaler - scaler fitted to data if none. Returns the input scaler otherwise
    '''
    num_features = 12
    num_samples = len(launch_speed)
    X = np.zeros((num_samples, num_features))
    X[:,0] = launch_speed
    X[:,1] = launch_speed ** 2.0
    X[:,2] = launch_speed ** 3.0
    X[:,3] = vertical_launch_angle
    X[:,4] = vertical_launch_angle ** 2.0
    X[:,5] = horizontal_launch_angle
    X[:,6] = horizontal_launch_angle ** 2.0
    X[:,7] = horizontal_launch_angle ** 3.0
    X[:,8] = horizontal_launch_angle ** 4.0
    X[:,9] = vertical_launch_angle * launch_speed
    X[:,10] = horizontal_launch_angle * launch_speed
    X[:,11] = vertical_launch_angle * horizontal_launch_angle
    if not scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    X_with_bias = np.zeros((len(launch_speed), num_features+1))
    X_with_bias[:,1:] = X
    X_with_bias[:,0] = np.ones(num_samples)
    return X_with_bias, scaler