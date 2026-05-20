# MicroPython and TinyML

Short answer: for this experiment, use Arduino C++ for the ESP32 firmware.

You wanted MicroPython, but the TinyML model from Edge Impulse is exported as native C++ code. That export contains:

- the audio feature extraction code,
- the TensorFlow Lite Micro runtime,
- the trained model weights,
- the classifier function.

Standard MicroPython on ESP32S3 cannot import that Arduino library directly.

## Simple Working Path

```text
Edge Impulse
  -> export Arduino library
  -> flash ESP32 with esp32_wake_word.ino
  -> run TinyML wake-word detection on the ESP32
  -> send recorded audio to the Python PC server
```

This is still a TinyML project because the model runs locally on the microcontroller.

## Advanced Path, Not Recommended First

You could build a custom MicroPython firmware that wraps the Edge Impulse C++ SDK as a native module. That is much harder because you would need to build and maintain custom ESP32S3 firmware.

For the first working experiment, do the Arduino version. After it works, MicroPython can still be used on the PC side or in a separate non-TinyML ESP32 experiment.
