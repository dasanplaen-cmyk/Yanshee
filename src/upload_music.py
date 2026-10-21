"""Upload the songs in music/ to the robot (only the ones it does not have yet), for --music.

  python src/upload_music.py [--ip IP]
"""
import argparse
import os
import sys
import time

# YanAPI.py and lib_ukit/ live in the project root, not in src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import YanAPI as yan  # noqa: E402
import config as C  # noqa: E402

VOLUME = 30
TIME_SEC = 10


def upload_new_songs(music_dir: str):
    song_list = [i['name'] for i in yan.get_media_music_list()['data']['music']]
    new_songs = []
    for song in sorted(os.listdir(music_dir)):
        if song not in song_list:
            new_songs.append(song)
            yan.upload_media_music(os.path.join(music_dir, song))
    print(f"Uploaded songs: {', '.join(new_songs)}" if new_songs else "No new songs to upload")


def play_song(music_file, time_sec):
    """Play one song for `time_sec` seconds (to try the volume)."""
    try:
        yan.set_robot_volume_value(VOLUME)
        print(yan.start_play_music(music_file))
        time.sleep(time_sec)
    finally:
        yan.stop_play_music()
        print('Music stopped')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=C.ROBOT_IP)
    args = ap.parse_args()
    yan.yan_api_init(args.ip)
    upload_new_songs(os.path.join(ROOT, "music"))

    # play_song("PartyRock.mp3", TIME_SEC)


if __name__ == "__main__":
    main()
