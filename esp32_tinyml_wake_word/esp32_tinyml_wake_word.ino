/*
  Local TinyML wake-word detector for Seeed Studio XIAO ESP32S3 Sense.

  The generated model lives in model_data.h. The rest of this sketch is split
  into small helper headers so the firmware is easier to present:
  - audio_runtime.h: microphone, buffers, rolling audio window
  - tinyml_runtime.h: TFLite Micro setup, feature extraction, inference
  - network_client.h: HTTP calls to the PC server
  - collection_mode.h: dataset collection commands
  - wake_detection.h: post-wake command capture/upload
*/

#include <Arduino.h>
#include <ESP_I2S.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <TFLIteMicro.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// ===== Edit these values first =====
const char *WIFI_SSID = "nisipeanu";
const char *WIFI_PASSWORD = "12345678";
const char *PC_SERVER_PING_URL = "http://10.123.21.208:8000/ping";
const char *PC_SERVER_UPLOAD_URL = "http://10.123.21.208:8000/upload";
const char *PC_SERVER_COLLECT_URL = "http://10.123.21.208:8000/collect";

enum DeviceMode {
  MODE_WAKE_DETECT = 0,
  MODE_COLLECT_DATA = 1,
};

constexpr DeviceMode DEVICE_MODE = MODE_WAKE_DETECT;
// constexpr DeviceMode DEVICE_MODE = MODE_COLLECT_DATA;

constexpr float AUTHORIZED_WAKE_THRESHOLD = 0.65f;
constexpr float UNKNOWN_USER_WAKE_THRESHOLD = 0.65f;
constexpr float AUTHORIZED_WAKE_MARGIN = 0.10f;
constexpr uint32_t WAKE_DEBOUNCE_MS = 3000;
constexpr uint32_t UNKNOWN_WAKE_DEBOUNCE_MS = 2000;
constexpr bool INPUT_PREEMPHASIS_ENABLED = true;
constexpr float INPUT_PREEMPHASIS_ALPHA = 0.97f;
constexpr bool INPUT_BANDPASS_ENABLED = true;
constexpr float INPUT_BANDPASS_HIGHPASS_CUTOFF_HZ = 80.0f;
constexpr float INPUT_BANDPASS_LOWPASS_CUTOFF_HZ = 3600.0f;
constexpr bool INPUT_REMOVE_DC_OFFSET = true;
constexpr bool SPEECH_GATE_ENABLED = true;
constexpr float SPEECH_GATE_MIN_CENTERED_RMS = 0.0010f;
constexpr float SPEECH_GATE_MIN_P2P = 0.0100f;
constexpr bool SPEECH_GATE_REQUIRE_BOTH_SIGNALS = false;
constexpr uint32_t SPEECH_GATE_HANGOVER_MS = 500;
constexpr bool SCORE_SMOOTHING_ENABLED = true;
constexpr float SCORE_SMOOTHING_ALPHA = 0.35f;
constexpr float DETECTION_HOP_SECONDS = 0.25f;
constexpr float COMMAND_CAPTURE_SECONDS = 5.0f;
constexpr uint32_t SERIAL_SCORE_EVERY_N = 4;
constexpr uint32_t COLLECT_POST_UPLOAD_DELAY_MS = 800;
// ==================================

constexpr int PDM_CLK_PIN = 42;
constexpr int PDM_DATA_PIN = 41;
constexpr int WINDOW_HOP_SAMPLES = static_cast<int>(DETECTION_HOP_SECONDS * kWakeWordSampleRate);
constexpr float WINDOW_SECONDS = static_cast<float>(kWakeWordInputSamples) / static_cast<float>(kWakeWordSampleRate);
constexpr float WINDOW_HOP_SECONDS = static_cast<float>(WINDOW_HOP_SAMPLES) / static_cast<float>(kWakeWordSampleRate);
constexpr int COMMAND_CAPTURE_SAMPLES = static_cast<int>(COMMAND_CAPTURE_SECONDS * kWakeWordSampleRate);

constexpr int VOICE_FEATURE_FRAME_COUNT = 32;
constexpr int VOICE_FEATURE_COUNT = VOICE_FEATURE_FRAME_COUNT * 3;
constexpr float ENERGY_FEATURE_GAIN = 200.0f;
constexpr float DIFF_FEATURE_GAIN = 400.0f;
constexpr float ZCR_FEATURE_GAIN = 4.0f;

constexpr int AUTHORIZED_USER_WAKE_INDEX = 0;
constexpr int UNKNOWN_USER_WAKE_INDEX = 1;
constexpr int NOT_WAKE_INDEX = 2;
constexpr int DETECTOR_LABEL_COUNT = 3;
const char *DETECTOR_LABELS[DETECTOR_LABEL_COUNT] = {
    "authorized_user_wake",
    "unknown_user_wake",
    "not_wake",
};

#ifndef LED_BUILTIN
#define LED_BUILTIN 21
#endif

constexpr int LED_PIN = LED_BUILTIN;
constexpr bool LED_ACTIVE_LOW = true;
constexpr int TENSOR_ARENA_SIZE_PSRAM = 300 * 1024;
constexpr int TENSOR_ARENA_SIZE_NO_PSRAM = 180 * 1024;

I2SClass I2S;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

uint8_t *tensor_arena = nullptr;
int16_t *audio_window = nullptr;
int16_t *command_audio = nullptr;
size_t tensor_arena_size = 0;

uint32_t last_wake_ms = 0;
uint32_t last_unknown_wake_ms = 0;
uint32_t inference_counter = 0;
int16_t discard_buffer[512];
float last_centered_rms = 0.0f;
float last_centered_p2p = 0.0f;
float smoothed_scores[DETECTOR_LABEL_COUNT] = {0.0f, 0.0f, 0.0f};
bool smoothed_scores_initialized = false;
uint32_t speech_gate_hold_until_ms = 0;

enum CollectLabel {
  COLLECT_AUTHORIZED_WAKE = 0,
  COLLECT_UNKNOWN_USER_WAKE = 1,
  COLLECT_NOT_WAKE = 2,
};

CollectLabel pending_collect_label = COLLECT_AUTHORIZED_WAKE;

#include "audio_runtime.h"
#include "network_client.h"
#include "tinyml_runtime.h"
#include "collection_mode.h"
#include "wake_detection.h"

void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(LED_PIN, OUTPUT);
  set_led(false);

  Serial.println("XIAO ESP32S3 Sense local TinyML wake-word detector");
  Serial.print("Device mode: ");
  Serial.println(DEVICE_MODE == MODE_COLLECT_DATA ? "DATA_COLLECTION" : "WAKE_DETECTION");
  Serial.print("Listening window: ");
  Serial.print(WINDOW_SECONDS, 2);
  Serial.println("s");
  Serial.print("Detection hop: ");
  Serial.print(WINDOW_HOP_SECONDS, 2);
  Serial.println("s");
  Serial.print("Authorized wake threshold: ");
  Serial.println(AUTHORIZED_WAKE_THRESHOLD, 2);
  Serial.print("Unknown-user wake threshold: ");
  Serial.println(UNKNOWN_USER_WAKE_THRESHOLD, 2);
  Serial.print("Preemphasis: ");
  Serial.println(INPUT_PREEMPHASIS_ENABLED ? "on" : "off");
  if (INPUT_PREEMPHASIS_ENABLED) {
    Serial.print("Preemphasis alpha: ");
    Serial.println(INPUT_PREEMPHASIS_ALPHA, 3);
  }
  Serial.print("Bandpass: ");
  Serial.println(INPUT_BANDPASS_ENABLED ? "on" : "off");
  if (INPUT_BANDPASS_ENABLED) {
    Serial.print("Bandpass HP cutoff Hz: ");
    Serial.println(INPUT_BANDPASS_HIGHPASS_CUTOFF_HZ, 1);
    Serial.print("Bandpass LP cutoff Hz: ");
    Serial.println(INPUT_BANDPASS_LOWPASS_CUTOFF_HZ, 1);
  }
  Serial.print("Speech gate: ");
  Serial.println(SPEECH_GATE_ENABLED ? "on" : "off");
  if (SPEECH_GATE_ENABLED) {
    Serial.print("Speech gate mode: ");
    Serial.println(SPEECH_GATE_REQUIRE_BOTH_SIGNALS ? "strict (rms AND p2p)" : "distance-friendly (rms OR p2p)");
    Serial.print("Speech gate min centered RMS: ");
    Serial.println(SPEECH_GATE_MIN_CENTERED_RMS, 5);
    Serial.print("Speech gate min p2p: ");
    Serial.println(SPEECH_GATE_MIN_P2P, 4);
    Serial.print("Speech gate hangover ms: ");
    Serial.println(SPEECH_GATE_HANGOVER_MS);
  }
  Serial.print("Score smoothing: ");
  Serial.println(SCORE_SMOOTHING_ENABLED ? "on" : "off");
  Serial.print("Post-wake command capture: ");
  Serial.print(COMMAND_CAPTURE_SECONDS, 1);
  Serial.println("s");
  Serial.println("Predictions run continuously on a rolling audio window.");

  connect_wifi();

  if (!start_microphone()) {
    while (true) delay(1000);
  }

  if (!ensure_audio_window()) {
    while (true) delay(1000);
  }

  if (DEVICE_MODE == MODE_WAKE_DETECT && !setup_tinyml()) {
    while (true) delay(1000);
  }

  Serial.println("Filling first audio window");
  refill_full_audio_window();
  if (DEVICE_MODE == MODE_COLLECT_DATA) {
    Serial.println("Collection mode ready.");
    print_collect_help();
  } else {
    Serial.println("Warmup complete.");
  }
}

void loop() {
  if (DEVICE_MODE == MODE_COLLECT_DATA) {
    run_collection_mode();
    return;
  }

  inference_counter++;
  const uint32_t now = millis();

  float scores[DETECTOR_LABEL_COUNT];
  const bool verbose = (inference_counter % SERIAL_SCORE_EVERY_N) == 0;
  if (!run_inference(scores, verbose)) {
    delay(50);
    return;
  }

  const bool authorized_debounce_elapsed = last_wake_ms == 0 || now - last_wake_ms >= WAKE_DEBOUNCE_MS;
  const bool unknown_debounce_elapsed = last_unknown_wake_ms == 0 || now - last_unknown_wake_ms >= UNKNOWN_WAKE_DEBOUNCE_MS;

  if (is_authorized_wake(scores) && authorized_debounce_elapsed) {
    last_wake_ms = now;
    const float wake_score = scores[AUTHORIZED_USER_WAKE_INDEX];

    Serial.print("Authorized wake detected with confidence ");
    Serial.println(wake_score, 3);

    set_led(true);
    if (send_detection_event_to_server(DETECTOR_LABELS[AUTHORIZED_USER_WAKE_INDEX], wake_score, true)) {
      Serial.println("Authorized event sent");
    } else {
      Serial.println("Authorized event failed");
    }
    capture_and_upload_command_audio(wake_score);
    set_led(false);

    refill_full_audio_window();
    return;
  }

  if (is_unknown_user_wake(scores) && unknown_debounce_elapsed) {
    last_unknown_wake_ms = now;
    const float unknown_score = scores[UNKNOWN_USER_WAKE_INDEX];
    Serial.print("Unknown user wake phrase rejected with confidence ");
    Serial.println(unknown_score, 3);
    send_detection_event_to_server(DETECTOR_LABELS[UNKNOWN_USER_WAKE_INDEX], unknown_score, false);
  }

  slide_audio_window();
}
