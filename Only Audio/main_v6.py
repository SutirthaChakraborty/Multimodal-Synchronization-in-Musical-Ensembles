import numpy as np
import wave

from kuramoto import Kuramoto
from swarmalator import Swarmalator
# constants

win_s = 1024
hop_s = win_s // 2
beat_vol = 0.00005

beat_a = wave.open("./drum-a.wav")
beat_b = wave.open("./drum-a.wav")
beat_c = wave.open("./drum-b.wav")
beat_d = wave.open("./drum-a.wav")

beats = [beat_a, beat_b, beat_c, beat_d]
beat_track = []


for i in range(len(beats)):
    beat = beats[i]
    beat_data = []
    beat_buff = np.frombuffer(beat.readframes(hop_s), np.int16)
    shape_size = np.shape(beat_buff)[0]

    while(shape_size > 0):
        if (shape_size < hop_s):
            size_difference = hop_s - shape_size
            beat_buff = np.pad(
                beat_buff, (0, size_difference), "constant")
        beat_data.append(beat_buff)
        beat_buff = np.frombuffer(beat.readframes(hop_s), np.int16) * beat_vol
        shape_size = np.shape(beat_buff)[0]
    beat_track.append(beat_data)

if __name__ == "__main__":
    # Live Audio
    init = [[0.5, 0.5], [0, 0]]
    for attempt in [
        Swarmalator(init),
        # Kuramoto(init),
        # Janus(init),
    ]:
        attempt.set_in_mode(0)
        attempt.set_out_mode(0)
        attempt.set_beat_track(beat_track)
        attempt.run_live("a.wav", 12)
    pass

