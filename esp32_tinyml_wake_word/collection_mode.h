#pragma once

void print_collect_help() {
  Serial.println("Collection commands:");
  Serial.println("  a -> capture/upload YOUR authorized HEY COMPUTER clip");
  Serial.println("  u -> capture/upload SOMEONE ELSE saying HEY COMPUTER");
  Serial.println("  n -> capture/upload one NOT_WAKE clip");
  Serial.println("  h -> show commands");
}

void run_collection_mode() {
  if (!Serial.available()) {
    delay(20);
    return;
  }

  const int incoming = Serial.read();
  if (incoming == 'h' || incoming == 'H') {
    print_collect_help();
    return;
  }

  if (incoming != 'a' && incoming != 'A' &&
      incoming != 'u' && incoming != 'U' &&
      incoming != 'n' && incoming != 'N') {
    return;
  }

  if (incoming == 'a' || incoming == 'A') {
    pending_collect_label = COLLECT_AUTHORIZED_WAKE;
  } else if (incoming == 'u' || incoming == 'U') {
    pending_collect_label = COLLECT_UNKNOWN_USER_WAKE;
  } else {
    pending_collect_label = COLLECT_NOT_WAKE;
  }

  Serial.print("Speak now: capturing ");
  Serial.print(WINDOW_SECONDS, 1);
  Serial.print("s for label=");
  Serial.println(collect_label_name(pending_collect_label));

  if (!read_pcm_samples(audio_window, kWakeWordInputSamples)) {
    Serial.println("Audio capture failed in collection mode");
    return;
  }

  print_audio_stats(audio_window, kWakeWordInputSamples);

  if (send_collection_clip_to_server(audio_window, kWakeWordInputSamples, pending_collect_label)) {
    Serial.println("Clip uploaded");
  } else {
    Serial.println("Clip upload failed");
  }

  Serial.print("Pause ");
  Serial.print(static_cast<float>(COLLECT_POST_UPLOAD_DELAY_MS) / 1000.0f, 1);
  Serial.println("s...");
  delay(COLLECT_POST_UPLOAD_DELAY_MS);
  Serial.println("Ready for next command (a/u/n)");
}

