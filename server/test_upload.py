from __future__ import annotations

import math
import struct
import urllib.request


SERVER_URL = "http://127.0.0.1:8000/upload"
SAMPLE_RATE = 16000
DURATION_SECONDS = 1
TONE_HZ = 440


def make_test_pcm() -> bytes:
    samples = []
    total_samples = SAMPLE_RATE * DURATION_SECONDS

    for n in range(total_samples):
        value = int(12000 * math.sin(2 * math.pi * TONE_HZ * n / SAMPLE_RATE))
        samples.append(struct.pack("<h", value))

    return b"".join(samples)


def main() -> None:
    pcm = make_test_pcm()
    request = urllib.request.Request(
        SERVER_URL,
        data=pcm,
        method="POST",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Bits-Per-Sample": "16",
            "X-Audio-Channels": "1",
        },
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
