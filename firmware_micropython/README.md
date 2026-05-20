# MicroPython Firmware Notes

This folder is intentionally documentation-only for now.

The requested wake-word firmware cannot be implemented as a normal MicroPython `main.py` while still using an Edge Impulse Arduino export. Edge Impulse's ESP32 TinyML runtime is native C/C++ firmware code.

Use the working firmware in:

```text
esp32_wake_word/esp32_wake_word.ino
```

For the detailed reason and the possible advanced custom MicroPython firmware path, read:

```text
docs/MICROPYTHON_AND_TINYML.md
```

