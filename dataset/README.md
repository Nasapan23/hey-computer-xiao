# Dataset Folder

Put your local training audio here.

Expected labels:

- `authorized_user_wake`
- `unknown_user_wake`
- `unknown_speech`

Folder layout:

```text
dataset/
  raw/
    authorized_user_wake/
      my_hey_computer_001.wav
      my_hey_computer_002.wav
    unknown_user_wake/
      other_user_hey_computer_001.wav
    unknown_speech/
      hello_001.wav
      random_words_001.wav
  processed/
```

## Audio Format

Use WAV files when possible:

- mono
- 16-bit PCM
- 16000 Hz sample rate
- 3 to 10 seconds is OK for this local trainer

The ESP32 sketch and trainer both use `16000 Hz`, so the model sample rate and firmware sample rate stay matched.

## How Much Data

For a first usable model:

- `authorized_user_wake`: at least 50 clips from your voice
- `unknown_user_wake`: at least 50 clips from other people saying the same phrase
- `unknown_speech`: at least 100 clips (include both other speech and ambient non-wake audio)

Better models usually need more data from different distances, rooms, and noise conditions. Speaker authorization specifically needs real clips from your voice and from other people saying the same phrase.

## Recording Tips

- Record your authorized wake word naturally, not always with the same rhythm.
- Ask other people to record the same wake phrase for `unknown_user_wake`.
- Include quiet and noisy rooms.
- Include close and far microphone distances.
- Put similar but wrong words in `unknown_speech`.
- Put silence, fans, typing, desk noise, music, and other non-speech sounds in `unknown_speech`.

Legacy folders `dataset/raw/wake_word/` and `dataset/raw/background_noise/` are ignored by the current trainer.

## Longer Recordings

Your files can be 3 to 10 seconds long. The local training script scans each file with overlapping 2 second windows.

For wake folders, the file should contain the phrase somewhere inside it. The script keeps the loudest 2 second windows, which usually contain the spoken phrase.

After training starts, check:

```text
dataset/processed/
```

Those are the exact chunks selected for training. If a selected wake chunk does not contain the full phrase, trim or re-record that source file.
