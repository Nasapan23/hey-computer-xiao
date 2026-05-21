#pragma once

void capture_and_upload_command_audio(float wake_score) {
  if (!ensure_command_audio_buffer()) {
    Serial.println("Command capture skipped because no buffer is available.");
    return;
  }

  Serial.print("Authorized wake accepted. Capturing next ");
  Serial.print(COMMAND_CAPTURE_SECONDS, 1);
  Serial.println("s for the server...");

  if (!read_pcm_samples(command_audio, COMMAND_CAPTURE_SAMPLES)) {
    Serial.println("Command audio capture failed");
    return;
  }

  if (upload_command_audio_to_server(command_audio, COMMAND_CAPTURE_SAMPLES, DETECTOR_LABELS[AUTHORIZED_USER_WAKE_INDEX], wake_score)) {
    Serial.println("Command audio uploaded");
  } else {
    Serial.println("Command audio upload failed");
  }
}

