# Dataset Folder

Put your training audio here before uploading it to Edge Impulse.

Expected labels:

- `wake_word`
- `unknown_speech`
- `background_noise`

Folder layout:

```text
dataset/
  raw/
    wake_word/
      jarvis_001.wav
      jarvis_002.wav
    unknown_speech/
      hello_001.wav
      random_words_001.wav
    background_noise/
      room_001.wav
      fan_001.wav
  processed/
```

## Audio Format

Use WAV files when possible:

- mono
- 16-bit PCM
- 16000 Hz sample rate
- 3 to 10 seconds is OK for this local trainer

The ESP32 sketch uses the sample rate exported by Edge Impulse as `EI_CLASSIFIER_FREQUENCY`, so the model sample rate and firmware sample rate stay matched.

## How Much Data

For a first usable model:

- `wake_word`: at least 50 to 100 clips
- `unknown_speech`: at least 100 clips
- `background_noise`: at least 50 clips

Better models usually need more data from different speakers, distances, rooms, and noise conditions.

## Recording Tips

- Record the wake word naturally, not always with the same rhythm.
- Include quiet and noisy rooms.
- Include close and far microphone distances.
- Put similar but wrong words in `unknown_speech`.
- Put silence, fans, typing, desk noise, music, and other non-speech sounds in `background_noise`.

## Longer Recordings

Your files can be 3 to 10 seconds long. The local training script scans each file with overlapping 2 second windows.

For `wake_word`, the file should contain the phrase somewhere inside it. The script keeps the loudest 2 second windows, which usually contain the spoken phrase.

After training starts, check:

```text
dataset/processed/
```

Those are the exact chunks selected for training. If a selected `wake_word` chunk does not contain the full phrase, trim or re-record that source file.
