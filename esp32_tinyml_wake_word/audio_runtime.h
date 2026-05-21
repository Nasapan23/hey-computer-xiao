#pragma once

void set_led(bool on) {
  digitalWrite(LED_PIN, LED_ACTIVE_LOW ? !on : on);
}

bool start_microphone() {
  I2S.setPinsPdmRx(PDM_CLK_PIN, PDM_DATA_PIN);

  if (!I2S.begin(I2S_MODE_PDM_RX,
                 kWakeWordSampleRate,
                 I2S_DATA_BIT_WIDTH_16BIT,
                 I2S_SLOT_MODE_MONO)) {
    Serial.println("Failed to initialize PDM microphone");
    return false;
  }

  Serial.print("Microphone started at ");
  Serial.print(kWakeWordSampleRate);
  Serial.println(" Hz, 16-bit mono");
  return true;
}

bool read_pcm_bytes(uint8_t *destination, size_t bytes_to_read) {
  size_t bytes_read = 0;

  while (bytes_read < bytes_to_read) {
    const size_t n = I2S.readBytes(reinterpret_cast<char *>(destination + bytes_read),
                                   bytes_to_read - bytes_read);
    if (n == 0) {
      delay(1);
      continue;
    }
    bytes_read += n;
  }

  return true;
}

bool read_pcm_samples(int16_t *destination, size_t samples_to_read) {
  return read_pcm_bytes(reinterpret_cast<uint8_t *>(destination),
                        samples_to_read * sizeof(int16_t));
}

bool discard_pcm_samples(size_t samples_to_discard) {
  size_t remaining = samples_to_discard;
  while (remaining > 0) {
    const size_t chunk = remaining > 512 ? 512 : remaining;
    if (!read_pcm_samples(discard_buffer, chunk)) {
      return false;
    }
    remaining -= chunk;
  }
  return true;
}

void *allocate_memory(size_t bytes) {
  void *ptr = ps_malloc(bytes);
  if (ptr == nullptr) {
    ptr = malloc(bytes);
  }
  return ptr;
}

bool ensure_audio_window() {
  if (audio_window != nullptr) {
    return true;
  }

  audio_window = static_cast<int16_t *>(allocate_memory(kWakeWordInputSamples * sizeof(int16_t)));
  if (audio_window == nullptr) {
    Serial.println("Failed to allocate audio window buffer.");
    Serial.println("Enable PSRAM in board menu if available.");
    return false;
  }

  return true;
}

bool ensure_command_audio_buffer() {
  if (command_audio != nullptr) {
    return true;
  }

  command_audio = static_cast<int16_t *>(allocate_memory(COMMAND_CAPTURE_SAMPLES * sizeof(int16_t)));
  if (command_audio == nullptr) {
    Serial.println("Failed to allocate command audio buffer.");
    Serial.println("Enable PSRAM in board menu if available.");
    return false;
  }

  return true;
}

void print_audio_stats(const int16_t *samples, int count) {
  int16_t min_value = 32767;
  int16_t max_value = -32768;
  double sum_sq = 0.0;

  for (int i = 0; i < count; i++) {
    const int16_t value = samples[i];
    if (value < min_value) min_value = value;
    if (value > max_value) max_value = value;
    const double normalized = static_cast<double>(value) / 32768.0;
    sum_sq += normalized * normalized;
  }

  const float rms = sqrtf(static_cast<float>(sum_sq / static_cast<double>(count)));

  Serial.print("Audio min=");
  Serial.print(min_value);
  Serial.print(" max=");
  Serial.print(max_value);
  Serial.print(" rms=");
  Serial.println(rms, 4);
}

void refill_full_audio_window() {
  read_pcm_samples(audio_window, kWakeWordInputSamples);
}

void slide_audio_window() {
  const int new_samples = WINDOW_HOP_SAMPLES;
  const int keep_samples = kWakeWordInputSamples - new_samples;

  memmove(audio_window,
          audio_window + new_samples,
          keep_samples * sizeof(int16_t));

  read_pcm_samples(audio_window + keep_samples, new_samples);
}

