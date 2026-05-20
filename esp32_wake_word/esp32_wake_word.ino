/*
  TinyML wake-word forwarder for Seeed Studio XIAO ESP32S3 Sense.

  Flow:
  1. Capture microphone audio from the onboard PDM microphone.
  2. Run Edge Impulse continuous audio inference.
  3. When "wake_word" is above WAKE_WORD_THRESHOLD, turn the LED on.
  4. Record raw 16-bit mono PCM for RECORD_SECONDS.
  5. HTTP POST the PCM bytes to the PC FastAPI server.

  Target: Arduino-ESP32 3.x with Seeed Studio XIAO ESP32S3 selected.
*/

#include <Arduino.h>

// Change this to the header created by your Edge Impulse Arduino library export.
#include <wake_word_inferencing.h>

#include <ESP_I2S.h>
#include <HTTPClient.h>
#include <WiFi.h>

// ===== Edit these values first =====
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char *PC_SERVER_URL = "http://192.168.1.50:8000/upload";

constexpr float WAKE_WORD_THRESHOLD = 0.85f;
constexpr uint32_t WAKE_DEBOUNCE_MS = 3000;
constexpr uint32_t RECORD_SECONDS = 4;
// ==================================

// XIAO ESP32S3 Sense onboard PDM microphone pins.
constexpr int PDM_CLK_PIN = 42;
constexpr int PDM_DATA_PIN = 41;

#ifndef LED_BUILTIN
#define LED_BUILTIN 21
#endif

// Many XIAO onboard LEDs are active-low. Set this false for an external active-high LED.
constexpr int LED_PIN = LED_BUILTIN;
constexpr bool LED_ACTIVE_LOW = true;

constexpr uint32_t SAMPLE_RATE = EI_CLASSIFIER_FREQUENCY;
constexpr uint16_t SAMPLE_BITS = 16;
constexpr uint16_t CHANNELS = 1;
constexpr size_t BYTES_PER_SAMPLE = SAMPLE_BITS / 8;
constexpr size_t RECORD_BYTES = SAMPLE_RATE * RECORD_SECONDS * CHANNELS * BYTES_PER_SAMPLE;

#if EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "This sketch requires an Edge Impulse audio/microphone impulse."
#endif

I2SClass I2S;

static int16_t inference_buffer[EI_CLASSIFIER_SLICE_SIZE];
static int wake_word_index = -1;
static uint32_t last_wake_ms = 0;

void set_led(bool on) {
  digitalWrite(LED_PIN, LED_ACTIVE_LOW ? !on : on);
}

void connect_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Wi-Fi connected, IP address: ");
  Serial.println(WiFi.localIP());
}

void ensure_wifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("Wi-Fi disconnected; reconnecting");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  const uint32_t started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 10000) {
    delay(250);
  }
}

bool start_microphone() {
  I2S.setPinsPdmRx(PDM_CLK_PIN, PDM_DATA_PIN);

  if (!I2S.begin(I2S_MODE_PDM_RX,
                 SAMPLE_RATE,
                 I2S_DATA_BIT_WIDTH_16BIT,
                 I2S_SLOT_MODE_MONO)) {
    Serial.println("Failed to initialize PDM microphone");
    return false;
  }

  Serial.print("Microphone started at ");
  Serial.print(SAMPLE_RATE);
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

int microphone_signal_get_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float(&inference_buffer[offset], out_ptr, length);
  return 0;
}

int find_label_index(const char *label) {
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (strcmp(ei_classifier_inferencing_categories[ix], label) == 0) {
      return static_cast<int>(ix);
    }
  }
  return -1;
}

uint8_t *allocate_recording_buffer() {
  uint8_t *buffer = static_cast<uint8_t *>(ps_malloc(RECORD_BYTES));
  if (buffer == nullptr) {
    buffer = static_cast<uint8_t *>(malloc(RECORD_BYTES));
  }
  return buffer;
}

bool post_audio_to_server(const uint8_t *audio, size_t audio_bytes) {
  ensure_wifi();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Cannot send audio: Wi-Fi is not connected");
    return false;
  }

  WiFiClient client;
  HTTPClient http;

  if (!http.begin(client, PC_SERVER_URL)) {
    Serial.println("HTTP begin failed");
    return false;
  }

  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-Audio-Sample-Rate", String(SAMPLE_RATE));
  http.addHeader("X-Audio-Bits-Per-Sample", String(SAMPLE_BITS));
  http.addHeader("X-Audio-Channels", String(CHANNELS));

  Serial.print("Posting ");
  Serial.print(audio_bytes);
  Serial.println(" bytes of PCM audio");

  const int status_code = http.POST(const_cast<uint8_t *>(audio), audio_bytes);
  const String response = http.getString();
  http.end();

  Serial.print("Server status: ");
  Serial.println(status_code);
  Serial.println(response);

  return status_code >= 200 && status_code < 300;
}

void record_and_send_audio() {
  uint8_t *recording = allocate_recording_buffer();
  if (recording == nullptr) {
    Serial.println("Failed to allocate recording buffer");
    return;
  }

  Serial.print("Recording ");
  Serial.print(RECORD_SECONDS);
  Serial.println(" seconds");

  const bool recorded = read_pcm_bytes(recording, RECORD_BYTES);
  bool sent = false;

  if (recorded) {
    sent = post_audio_to_server(recording, RECORD_BYTES);
  }

  free(recording);

  if (sent) {
    Serial.println("Audio sent successfully");
  } else {
    Serial.println("Audio send failed");
  }
}

void print_inference_scores(const ei_impulse_result_t &result) {
  Serial.print("Predictions:");
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    Serial.print(" ");
    Serial.print(ei_classifier_inferencing_categories[ix]);
    Serial.print("=");
    Serial.print(result.classification[ix].value, 3);
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(LED_PIN, OUTPUT);
  set_led(false);

  Serial.println("XIAO ESP32S3 Sense TinyML wake-word forwarder");

  wake_word_index = find_label_index("wake_word");
  if (wake_word_index < 0) {
    Serial.println("Model does not contain required label: wake_word");
    while (true) {
      delay(1000);
    }
  }

  connect_wifi();

  if (!start_microphone()) {
    while (true) {
      delay(1000);
    }
  }

  run_classifier_init();
}

void loop() {
  if (!read_pcm_samples(inference_buffer, EI_CLASSIFIER_SLICE_SIZE)) {
    return;
  }

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
  signal.get_data = &microphone_signal_get_data;

  ei_impulse_result_t result = {0};
  const EI_IMPULSE_ERROR error = run_classifier_continuous(&signal, &result, false);
  if (error != EI_IMPULSE_OK) {
    Serial.print("run_classifier_continuous failed: ");
    Serial.println(error);
    return;
  }

  print_inference_scores(result);

  const float wake_score = result.classification[wake_word_index].value;
  const uint32_t now = millis();
  const bool debounce_elapsed = last_wake_ms == 0 || now - last_wake_ms >= WAKE_DEBOUNCE_MS;

  if (wake_score > WAKE_WORD_THRESHOLD && debounce_elapsed) {
    last_wake_ms = now;

    Serial.print("Wake word detected with confidence ");
    Serial.println(wake_score, 3);

    set_led(true);
    record_and_send_audio();
    set_led(false);

    run_classifier_init();
  }
}
