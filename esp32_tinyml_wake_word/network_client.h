#pragma once

constexpr uint32_t WIFI_CONNECT_TIMEOUT_MS = 15000;
constexpr uint32_t WIFI_RETRY_COOLDOWN_MS = 5000;

const char *wifi_status_to_text(wl_status_t status) {
  switch (status) {
    case WL_IDLE_STATUS: return "WL_IDLE_STATUS";
    case WL_NO_SSID_AVAIL: return "WL_NO_SSID_AVAIL";
    case WL_SCAN_COMPLETED: return "WL_SCAN_COMPLETED";
    case WL_CONNECTED: return "WL_CONNECTED";
    case WL_CONNECT_FAILED: return "WL_CONNECT_FAILED";
    case WL_CONNECTION_LOST: return "WL_CONNECTION_LOST";
    case WL_DISCONNECTED: return "WL_DISCONNECTED";
    default: return "WL_UNKNOWN";
  }
}

bool wait_for_wifi(uint32_t timeout_ms) {
  const uint32_t started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < timeout_ms) {
    delay(500);
    Serial.print(".");
  }
  return WiFi.status() == WL_CONNECTED;
}

void begin_wifi_connect() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void connect_wifi() {
  Serial.print("Connecting to Wi-Fi");
  begin_wifi_connect();
  if (wait_for_wifi(WIFI_CONNECT_TIMEOUT_MS)) {
    Serial.println();
    Serial.print("Wi-Fi connected, IP address: ");
    Serial.println(WiFi.localIP());
    return;
  }

  Serial.println();
  Serial.print("Wi-Fi connect timeout, status=");
  Serial.println(wifi_status_to_text(WiFi.status()));
  Serial.println("Continuing without Wi-Fi; network actions will retry in background.");
}

void ensure_wifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  static uint32_t last_retry_ms = 0;
  const uint32_t now = millis();
  if (now - last_retry_ms < WIFI_RETRY_COOLDOWN_MS) {
    return;
  }
  last_retry_ms = now;

  Serial.print("Wi-Fi disconnected; reconnecting (status=");
  Serial.print(wifi_status_to_text(WiFi.status()));
  Serial.println(")");

  WiFi.disconnect();
  begin_wifi_connect();
  wait_for_wifi(3000);

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Wi-Fi reconnected, IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.print("Wi-Fi reconnect pending, status=");
    Serial.println(wifi_status_to_text(WiFi.status()));
  }
}

const char *collect_label_name(CollectLabel label) {
  if (label == COLLECT_AUTHORIZED_WAKE) return "authorized_user_wake";
  if (label == COLLECT_UNKNOWN_USER_WAKE) return "unknown_user_wake";
  return "not_wake";
}

bool send_detection_event_to_server(const char *event_label, float score, bool authorized) {
  ensure_wifi();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Cannot send detection event: Wi-Fi is not connected");
    return false;
  }

  WiFiClient client;
  HTTPClient http;

  if (!http.begin(client, PC_SERVER_PING_URL)) {
    Serial.println("HTTP begin failed");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  const String payload =
      "{\"device\":\"xiao_esp32s3_sense\",\"event\":\"" + String(event_label) +
      "\",\"score\":" + String(score, 3) +
      ",\"authorized\":" + String(authorized ? "true" : "false") + "}";

  const int status_code = http.POST(payload);
  const String response = http.getString();
  http.end();

  Serial.print("Server status: ");
  Serial.println(status_code);
  Serial.println(response);

  return status_code >= 200 && status_code < 300;
}

bool upload_command_audio_to_server(const int16_t *samples, size_t sample_count, const char *trigger_label, float trigger_score) {
  ensure_wifi();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Cannot upload command audio: Wi-Fi is not connected");
    return false;
  }

  WiFiClient client;
  HTTPClient http;

  if (!http.begin(client, PC_SERVER_UPLOAD_URL)) {
    Serial.println("HTTP begin failed for /upload");
    return false;
  }

  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-Audio-Sample-Rate", String(kWakeWordSampleRate));
  http.addHeader("X-Audio-Bits-Per-Sample", "16");
  http.addHeader("X-Audio-Channels", "1");
  http.addHeader("X-Trigger-Label", trigger_label);
  http.addHeader("X-Trigger-Score", String(trigger_score, 3));

  const size_t byte_count = sample_count * sizeof(int16_t);
  uint8_t *payload = reinterpret_cast<uint8_t *>(const_cast<int16_t *>(samples));
  const int status_code = http.POST(payload, byte_count);
  const String response = http.getString();
  http.end();

  Serial.print("Command upload status: ");
  Serial.println(status_code);
  Serial.println(response);

  return status_code >= 200 && status_code < 300;
}

bool send_collection_clip_to_server(const int16_t *samples, size_t sample_count, CollectLabel label) {
  ensure_wifi();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Cannot upload clip: Wi-Fi is not connected");
    return false;
  }

  WiFiClient client;
  HTTPClient http;

  if (!http.begin(client, PC_SERVER_COLLECT_URL)) {
    Serial.println("HTTP begin failed for /collect");
    return false;
  }

  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-Audio-Sample-Rate", String(kWakeWordSampleRate));
  http.addHeader("X-Audio-Bits-Per-Sample", "16");
  http.addHeader("X-Audio-Channels", "1");
  http.addHeader("X-Clip-Label", collect_label_name(label));

  const size_t byte_count = sample_count * sizeof(int16_t);
  uint8_t *payload = reinterpret_cast<uint8_t *>(const_cast<int16_t *>(samples));
  const int status_code = http.POST(payload, byte_count);
  const String response = http.getString();
  http.end();

  Serial.print("Collect upload status: ");
  Serial.println(status_code);
  Serial.println(response);

  return status_code >= 200 && status_code < 300;
}
