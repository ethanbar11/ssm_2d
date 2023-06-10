"""Utilities to assist with learning."""
import numpy as np
import sklearn
import sklearn.model_selection
from pathlib import Path

from numba import njit
import pandas as pd
import scipy.ndimage.filters as filters

ALL_STUDIES = [1, 2, 3, 4, 5, 7, 8, 11, 12, 17, 18, 19, 20, 22, 23, 24, 26, 29,
               32, 33]
TRAINING_STUDIES = [1, 2, 3, 4, 5, 7, 12, 17, 18, 20]


def minutes_to_ms(m):
    return m * 60 * 1000


def timestamp_from_clock(clock):
    segments = clock.split(":")
    assert len(segments) > 1
    hour, minute = segments[:2]
    second = 0
    if len(segments) == 3:
        second = segments[2]
    return int(hour) * 60 * 60 * 1000 + int(minute) * 60 * 1000 + int(second) \
        * 1000


def timestamp_to_clock(timestamp, debug_absolute=False):
    hours = timestamp / 1000 / 60 / 60
    if not debug_absolute:
        hours = hours % 24
    minutes = round((hours - int(hours)) * 60, 3) % 60
    seconds = round((minutes - int(minutes)) * 60, 3) % 60
    segments = [hours, minutes, seconds]
    clock = []
    for segment in segments:
        converted = str(int(segment))
        if len(converted) != 2:
            converted = "0{}".format(converted)
        clock.append(converted)
    return ":".join(clock)


def get_closest(a, b, threshold=minutes_to_ms(1.0)):
    """
    Get items in `a` that are closest to the items in `b` but at least as
    close as `threshold`. Essentially a fuzzy search algorithm.
    """
    result = []
    for x in a:
        found = []
        for i, y in enumerate(b):
            diff = abs(y - x)
            if diff < threshold:
                found.append((i, diff, y))
        if found:
            # Remove the closest event from consideration.
            closest, _, y = found[np.argmin([e[1] for e in found])]
            b = np.delete(b, closest)
            result.append(y)
    return result


def remove_closest(a, b, threshold=minutes_to_ms(1.0)):
    """
    Remove items in `a` that are closest to the items in `b` but at least as
    close as `threshold`. Essentially a fuzzy deletion algorithm.
    """
    closest = get_closest(a, b, threshold)
    return sorted(list(set(b) - set(closest)))



@njit(fastmath=True)
def _find_dips(signal, ratios, threshold, maxlen, minlen, wf, testlen):
    dips = []
    # scan signal from left to right
    i = 0
    while i < signal.shape[0]:
        # if ratio of forward mean to backward mean is less than the threshold,
        # i.e., the signal is dropping off, continue searching until an exit is
        # found.
        if ratios[i] < threshold:
            # mark entry index
            entry, exit = i, None
            while True:
                i += 1
                # if the exit is not found within 'maxlen' positions of entry,
                # abandon this drop
                if i - entry > maxlen:
                    break
                # if the ratio is greater than the threshold, check if this
                # is a real exit
                if ratios[i] > threshold:
                    exit_tentative = i
                    # check stability of signal after this tentative exit
                    # to determine if it is really an exit
                    test_points = ratios[exit_tentative:
                                         exit_tentative + testlen]
                    N_test_points = test_points.shape[0]
                    N_stable_points = np.nonzero(test_points > threshold)[0]\
                        .shape[0]
                    # if a majority of the following signal is stable,
                    # then mark this as true exit and break
                    # TODO: make stability ratio configurable
                    stable = N_stable_points/(N_test_points + 1e-3) >= 0.5
                    if stable:
                        exit = exit_tentative
                        break
                    # else:  continue
            if exit is not None and exit - entry >= minlen:
                dips.append((entry, exit))
        i += 1
    return dips


def find_dips(signal, windows, threshold, minlen=None, maxlen=None,
              testlen=None):
    assert len(windows) == 2
    # calculate backward and forward means for each point
    wb, wf = windows
    # TODO: make sure `origin` is correct
    bk = filters.uniform_filter1d(signal, size=wb, origin=-wb//2+1)
    fw = filters.uniform_filter1d(signal, size=wf, origin=wf//2-1)
    ratios = fw/bk

    if maxlen is None:
        maxlen = signal.shape[0]
    if minlen is None:
        minlen = 1
    if testlen is None:
        testlen = wf * 4

    return _find_dips(signal, ratios, threshold, maxlen, minlen, wf, testlen)


# use whatever heuristics/algorithms are necessary and return predicted
# events for `study`
def predict(study):
    study.z.downsample(8)
    secs = study.z.sampling_rate
    dips = find_dips(study.z.sensors[-1], windows=(10*secs, 1*secs),
                     threshold=0.7, minlen=1, maxlen=5*60*secs,
                     testlen=5*secs)
    timestamps = study.z.timestamps[[(x[0] + x[1])//2 for x in dips]]
    return timestamps




# parse a blob containing impedance and pH data.
def parse_txt(filename):
    # split the file into lines
    with open(filename) as f:
        lines = f.readlines()

    # line keeps track of the line (zero-indexed) that the parser is on
    line = 0

    # parses the pH/impedance array section
    def _parse_array():
        nonlocal line
        line += 1
        # There are two types of headers. Check which kind and parse the
        # header accordingly.
        # N = number of records
        # S = number of sensors
        if lines[line].startswith("Number of records"):
            N = int(lines[line].split("=")[1])
            line += 1
            assert lines[line].startswith("Number of sensors")
            S = int(lines[line].split("=")[1])
        else:
            N, S = [int(e) for e in lines[line].split()]
        # Skip column labels/empty lines, go directly to the records.
        line += 2

        line_max = line + N
        while line < line_max:
            yield [float(e) for e in lines[line].split()]
            line += 1

    ph, z = [], []
    while line < len(lines):
        if lines[line].startswith("Ph Array"):
            arr = _parse_array()
            for j, record in enumerate(arr):
                ph.append(record)
        elif lines[line].startswith("Impedance Array"):
            arr = _parse_array()
            for j, record in enumerate(arr):
                z.append(record)
        else:
            line += 1

    return np.array(ph), np.array(z)


class SensorData:
    def __init__(self, signals, sampling_rate, exclusions=[]):
        self.readings = signals[:, 1:]
        self.sensors = self.readings.T
        self.timestamps = signals[:, 0]
        self.exclusions = exclusions
        self.sampling_rate = sampling_rate

    def construct_clips(self, clip_duration=30*1000, points=[]):
        """Create clips of duration `clip_duration` around points."""
        # actual shape of the required array
        length = int(clip_duration // 1000) * self.sampling_rate
        clip_shape = (length,
                      self.readings.shape[1])

        # construct clips
        clips = []
        for point in points:
            # find index of point using binary search
            index = np.searchsorted(self.timestamps, point)
            # slice array
            start, end = (index - int(length // 2),
                          index - int(length // 2) + length)
            # bounds check
            if start < 0:
                start = 0
            if end >= self.readings.shape[0]:
                end = self.readings.shape[0]
            # slice out clip
            clip = self.readings[start:end]
            # sanity check
            # TODO: remove this
            assert clip.shape[0] <= length
            # pad array if its length is small due to being
            # on the edges of the data array
            if clip.shape[0] < length:
                shortfall = length - clip.shape[0]
                tpad = bpad = shortfall // 2
                if shortfall % 2 != 0:
                    bpad += 1
                # mean pad the array
                # TODO: explore other options for padding modes
                clip = np.pad(clip, pad_width=((tpad, bpad), (0, 0)),
                              mode="mean")
                # print("Clip too small: {}".format(point))
            clip = clip[:length]
            # another sanity check
            # TODO: remove this
            assert clip.shape == clip_shape
            clips.append(clip)
        return np.array(clips)

    def downsample(self, factor):
        assert self.sampling_rate % factor == 0
        self.readings = self.readings[::factor]
        self.sensors = self.readings.T
        self.timestamps = self.timestamps[::factor]
        self.sampling_rate //= factor

    def get_sensor_mean(self, sensor=-1):
        return np.mean(self.readings[:, sensor])

    def get_mask(self):
        inclusions = np.ones(self.timestamps.shape[0]).astype(bool)
        ts = self.timestamps
        for exclusion in self.exclusions:
            inclusions &= (ts < exclusion[0]) | (ts > exclusion[1])
        return inclusions

    def apply_mask(self, timestamps):
        result = []
        for timestamp in timestamps:
            included = True
            for start, end in self.exclusions:
                if start <= timestamp <= end:
                    included = False
                    break
            if included:
                result.append(timestamp)
        return np.array(result)

DAY_MS = 86_400_000  # 24 * 60 * 60 * 1000


class Truth:
    def __init__(self, raw_truth=None, filename=""):
        if filename:
            assert raw_truth is None
            with open(filename) as f:
                raw_truth = f.read()
        self.parse(raw_truth)

    def __repr__(self):
        return "Start Time: {}\nMeals: {}\nEvents: {}".format(
            self.epoch, self.meals, self.events
        )

    def parse(self, raw_truth):
        """
        Parse truth data. Times specified in raw_truth must be in order of
        occurrence.
        """
        truth = [line.strip() for line in raw_truth.split("\n")]
        if truth[0].strip().startswith("#"):
            self.meta = truth.pop(0)
        self.epoch = timestamp_from_clock(truth[0][len("Start: "):])
        truth = truth[1:]

        self.meals = []
        self.events = []
        # 0-indexed day of study, used to calculate exact timestamp by adding
        # 24h since ground truth data records timestamps in clock time.
        day = 0
        meals_day = 0
        for line in truth:
            ts = None
            if line.startswith("Meal "):
                start, end = [
                    timestamp_from_clock(segment)
                    for segment in line[len("Meal "):].strip().split("-")
                ]
                # TODO: review this code: might be a bug.
                if self.meals and start < self.meals[-1][0] - (meals_day *
                                                               DAY_MS):
                    meals_day += 1
                start += meals_day * DAY_MS
                end += meals_day * DAY_MS
                self.meals.append((start, end))
            else:
                clock = line.split("#")[0]
                if clock:
                    ts = timestamp_from_clock(clock)
                    # TODO: review this code: might be a bug.
                    if self.events and ts < (self.events[-1] - (day * DAY_MS)):
                        day += 1
                    ts += day * DAY_MS  # add a multiple of 24 hours
                    self.events.append(ts)
        self.events = np.array(self.events)
        self.meals = np.array(self.meals)


class Study:
    """Convenience wrapper for sensor data."""

    def __init__(self, ph=None, z=None, filename="", hz=(1, 64),
                 downsample=(None, None)):
        if filename:
            assert ph is None and z is None
            if filename.endswith('.txt'):
                ph, z = parse_txt(filename)
            elif filename.endswith('.npz'):
                npz = np.load(filename)
                ph, z = npz["ph"], npz["z"]
        else:
            assert ph is not None and z is not None
        self.ph = SensorData(ph, sampling_rate=hz[0])
        self.z = SensorData(z, sampling_rate=hz[1])

    def set_exclusions(self, exclusions):
        self.ph.exclusions = exclusions
        self.z.exclusions = exclusions



def build_clipset(path, study_ids, clip_duration=30*1000):
    """Build training dataset."""
    # loop over all studies required to be in training dataset and
    # successively append their clips to `result`
    result = [[], []]  # [[P_0, P_1, P_2, ...], [N_0, N_1, N_2, ...]]
    for S in study_ids:
        # load study and ground truth data
        study, truth = load_one(S, path)
        study.set_exclusions(truth.meals)

        # fetch predictions
        predictions = predict(study)

        # split predictions into positive and negative
        split_predictions = [
            get_closest(truth.events, predictions),  # positive
            remove_closest(truth.events, predictions)  # negative
        ]

        # construct clips and add them to `result`
        for i, p in enumerate(split_predictions):
            clips = study.z.construct_clips(
                points=p,
                clip_duration=clip_duration
            )
            result[i].extend(clips)

    # convert resulting lists to np arrays
    result = [np.array(x) for x in result]
    return result[0], result[1]

def build_gold_clipset(path, study_ids, clip_duration=30*1000, variant=0):

    result = [[], []]  # [[P_0, P_1, P_2, ...], [N_0, N_1, N_2, ...]]
    gold_df = pd.read_csv(path + '/studies/truth/goldv2.csv') #gold.csv had some mistakes
    gold_df = gold_df[['patient_number', 'Time_start', 'consensus']]
    # gold_df = gold_df.sort_values(by=['patient_number', 'Time_start'])

    for S in study_ids:
        study = Study(filename=path + "/studies/data/{}.npz".format(S))
        gold = gold_df[gold_df['patient_number'] == S]
        _, times, labels = gold.to_numpy().T


        ts = np.array([timestamp_from_clock(clock) for clock in times])
        sort_indices = np.argsort(ts)
        ts = np.array(ts[sort_indices])
        labels = np.array(labels[sort_indices])

        mask = np.zeros_like(ts)
        mask[np.where(ts < study.z.timestamps.min())[0]] = DAY_MS

        ts = ts + mask
        sort_indices = np.argsort(ts)
        events = ts[sort_indices]
        labels = labels[sort_indices]

        # events = np.array(events)
        events, not_events = events[labels == 1], events[labels == 0]

        # fetch predictions
        predictions = predict(study)

        if variant == 0:
            # split predictions into positive and negative
            a = get_closest(events, predictions)
            b = remove_closest(events, predictions)
            print(S, len(events) - len(a), len(events), len(predictions), len(a), len(b))
            split_predictions = [
                a,  # positive
                b # negative
            ]
        elif variant == 1:
            split_predictions = [events, not_events]

        # construct clips and add them to `result`
        for i, p in enumerate(split_predictions):
            clips = study.z.construct_clips(points=p,
                                            clip_duration=clip_duration)
            result[i].extend(clips)

    # convert resulting lists to np arrays
    result = [np.array(x) for x in result]
    return result[0], result[1]

def load_one(N, path):
    study = Study(filename=path + "/studies/data/{}.npz".format(N))
    truth = Truth(filename=path + "/studies/truth/{}.txt".format(N))
    return study, truth


def load(*ids):
    result = []
    for N in ids:
        result.append(load_one(N))
    return result


class Transformer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        pass

    def fit(self, X, Y=None):
        assert Y is None
        self.mean_ = np.mean(X, axis=(0, 1))
        self.std_ = np.std(X, axis=(0, 1))

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X)


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X, Y=None):
        assert Y is None
        self.min_ = np.min(X) - 1e-8
        self.max_ = np.max(X)

    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X)


def _reorder_clipset(clipset, order):
    order = order.upper()
    assert order in ['NHWC', 'NCHW', 'NHW'], \
        "Unknown order"
    positives, negatives = clipset
    assert positives.ndim == 3 and negatives.ndim == 3, \
        "Clipset dimensions malformed"
    if order == 'NHWC':
        new_shape = (positives.shape[:3] + (1,), negatives.shape[:3] + (1,))
    elif order == 'NCHW':
        new_shape = ((positives.shape[0], 1) + positives.shape[1:],
                     (negatives.shape[0], 1) + negatives.shape[1:])
    elif order == 'NHW':
        new_shape = (positives.shape, negatives.shape)
    positives = positives.reshape(*new_shape[0])
    negatives = negatives.reshape(*new_shape[1])
    return positives, negatives


def _binary_one_hot_labels(class_, n):
    """Generate N one-hot labels of class class_."""
    assert class_ == 0 or class_ == 1
    if class_ == 0:
        return np.array([[1, 0]] * n)
    elif class_ == 1:
        return np.array([[0, 1]] * n)


def _check_y_format(y_format):
    assert y_format in ['onehot', 'classindex'], \
        "Format must be either one-hot or class indexed"


def prepare_dataset(clipset,
                    order='NHWC',
                    y_format='onehot',
                    dev_size=0.2,
                    scaling='standardize',
                    ):
    """Prepare a dataset from `clipset`, returning `X_train`, `X_test`,
    `Y_train`, `Y_test`.

    Parameters
    ----------
    clipset : array
        The clipset from which to generate a dataset.
    order : str, optional
        Order of training data set, where N is the number of samples, H is the
        'height' or first dimension of the data, W is the 'width' or second
        dimension, and C is the channel dimension. Defaults to 'NHWC'.
    y_format : str, optional
        Format of labels. Either 'onehot' or 'classindex'. Defaults to
        'onehot'.
    dev_size : float, optional
        Ratio of data to use for test set. Defaults to 0.2.

    Returns
    -------
    X_train, X_test, Y_train, Y_test : array
    """
    _check_y_format(y_format)
    positives, negatives = _reorder_clipset(clipset, order)

    X = np.concatenate((negatives, positives))
    # one-hot labels
    Y = np.concatenate((_binary_one_hot_labels(0, negatives.shape[0]),
                        _binary_one_hot_labels(1, positives.shape[0])))
    if y_format == 'classindex':
        Y = Y.argmax(axis=1)
    Y = Y.astype(np.int64)
    split = sklearn.model_selection.train_test_split(X, Y, test_size=dev_size)
    X_train, X_test, Y_train, Y_test = split

    # scale data
    if scaling == 'standardize':
        transformer = Transformer()
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)

    elif scaling == 'minmax':
        minmaxscaler = MinMaxScaler()
        X_train = minmaxscaler.fit_transform(X_train)
        X_test = minmaxscaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test
