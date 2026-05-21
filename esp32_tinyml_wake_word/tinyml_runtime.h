#pragma once

bool setup_tinyml() {
  const bool has_psram = psramFound();
  tensor_arena_size = has_psram ? TENSOR_ARENA_SIZE_PSRAM : TENSOR_ARENA_SIZE_NO_PSRAM;

  Serial.print("PSRAM found: ");
  Serial.println(has_psram ? "yes" : "no");
  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());
  if (has_psram) {
    Serial.print("Free PSRAM: ");
    Serial.println(ESP.getFreePsram());
  }

  tensor_arena = static_cast<uint8_t *>(allocate_memory(tensor_arena_size));
  if (!ensure_audio_window()) {
    return false;
  }

  if (tensor_arena == nullptr) {
    Serial.print("Failed to allocate TinyML memory. arena=");
    Serial.print(tensor_arena_size);
    Serial.println(" bytes.");
    Serial.println("Enable PSRAM in board menu if available.");
    return false;
  }

  model = tflite::GetModel(wake_word_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema version mismatch");
    return false;
  }

  static tflite::MicroMutableOpResolver<4> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("Failed to register TinyML ops");
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      tensor_arena_size);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed. Increase tensor arena or enable PSRAM.");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  const size_t expected_input_bytes =
      (input->type == kTfLiteFloat32)
          ? VOICE_FEATURE_COUNT * sizeof(float)
          : VOICE_FEATURE_COUNT;
  const size_t expected_output_bytes =
      (output->type == kTfLiteFloat32)
          ? DETECTOR_LABEL_COUNT * sizeof(float)
          : DETECTOR_LABEL_COUNT;

  if (input->bytes != expected_input_bytes || output->bytes != expected_output_bytes) {
    Serial.println("Model tensor shape does not match the current authorized-user wake detector.");
    Serial.print("Expected input bytes=");
    Serial.print(expected_input_bytes);
    Serial.print(" actual=");
    Serial.println(input->bytes);
    Serial.print("Expected output bytes=");
    Serial.print(expected_output_bytes);
    Serial.print(" actual=");
    Serial.println(output->bytes);
    Serial.println("Retrain with training/train_local_wake_word.py before flashing this firmware.");
    return false;
  }

  Serial.print("TinyML model loaded. Input samples: ");
  Serial.println(kWakeWordInputSamples);
  Serial.print("Voice features: ");
  Serial.println(VOICE_FEATURE_COUNT);
  return true;
}

void write_feature_to_input(int feature_index, float feature) {
  feature = constrain(feature, 0.0f, 1.0f);

  if (input->type == kTfLiteInt8) {
    int32_t quantized = static_cast<int32_t>(roundf(feature / input->params.scale)) + input->params.zero_point;
    quantized = constrain(quantized, -128, 127);
    input->data.int8[feature_index] = static_cast<int8_t>(quantized);
    return;
  }

  if (input->type == kTfLiteFloat32) {
    input->data.f[feature_index] = feature;
  }
}

float frontend_highpass_alpha() {
  if (INPUT_BANDPASS_HIGHPASS_CUTOFF_HZ <= 0.0f) {
    return 0.0f;
  }
  constexpr float kPi = 3.14159265358979323846f;
  const float dt = 1.0f / static_cast<float>(kWakeWordSampleRate);
  const float rc = 1.0f / (2.0f * kPi * INPUT_BANDPASS_HIGHPASS_CUTOFF_HZ);
  return rc / (rc + dt);
}

float frontend_lowpass_alpha() {
  if (INPUT_BANDPASS_LOWPASS_CUTOFF_HZ <= 0.0f) {
    return 1.0f;
  }
  constexpr float kPi = 3.14159265358979323846f;
  const float dt = 1.0f / static_cast<float>(kWakeWordSampleRate);
  const float rc = 1.0f / (2.0f * kPi * INPUT_BANDPASS_LOWPASS_CUTOFF_HZ);
  return dt / (rc + dt);
}

struct FrontendState {
  float previous_input = 0.0f;
  float hp_previous_input = 0.0f;
  float hp_previous_output = 0.0f;
  float lp_previous_output = 0.0f;
  bool primed = false;
};

float apply_frontend_dsp_sample(float raw_sample, FrontendState &state) {
  float sample = raw_sample;

  if (INPUT_PREEMPHASIS_ENABLED) {
    const float previous = state.primed ? state.previous_input : raw_sample;
    sample = raw_sample - (INPUT_PREEMPHASIS_ALPHA * previous);
    state.previous_input = raw_sample;
  }

  if (INPUT_BANDPASS_ENABLED) {
    if (INPUT_BANDPASS_HIGHPASS_CUTOFF_HZ > 0.0f) {
      const float alpha_hp = frontend_highpass_alpha();
      const float previous_input = state.primed ? state.hp_previous_input : sample;
      const float previous_output = state.primed ? state.hp_previous_output : 0.0f;
      const float highpassed = alpha_hp * (previous_output + sample - previous_input);
      state.hp_previous_input = sample;
      state.hp_previous_output = highpassed;
      sample = highpassed;
    }

    if (INPUT_BANDPASS_LOWPASS_CUTOFF_HZ > 0.0f) {
      const float alpha_lp = frontend_lowpass_alpha();
      const float previous_output = state.primed ? state.lp_previous_output : sample;
      const float lowpassed = previous_output + (alpha_lp * (sample - previous_output));
      state.lp_previous_output = lowpassed;
      sample = lowpassed;
    }
  }

  state.primed = true;
  return sample;
}

void put_audio_in_input_tensor(const int16_t *samples) {
  FrontendState mean_state;
  float mean = 0.0f;
  if (INPUT_REMOVE_DC_OFFSET) {
    for (int i = 0; i < kWakeWordInputSamples; i++) {
      const float normalized = static_cast<float>(samples[i]) / 32768.0f;
      mean += apply_frontend_dsp_sample(normalized, mean_state);
    }
    mean /= static_cast<float>(kWakeWordInputSamples);
  }

  FrontendState feature_state;
  float centered_min = 1.0f;
  float centered_max = -1.0f;
  float sum_sq = 0.0f;
  const int frame_size = kWakeWordInputSamples / VOICE_FEATURE_FRAME_COUNT;
  for (int frame_index = 0; frame_index < VOICE_FEATURE_FRAME_COUNT; frame_index++) {
    float sum_abs = 0.0f;
    float sum_diff = 0.0f;
    int sign_changes = 0;

    const int start = frame_index * frame_size;
    const float first_raw = static_cast<float>(samples[start]) / 32768.0f;
    float previous = apply_frontend_dsp_sample(first_raw, feature_state) - mean;
    bool previous_positive = previous >= 0.0f;

    for (int i = 0; i < frame_size; i++) {
      float centered = previous;
      if (i > 0) {
        const float raw_sample = static_cast<float>(samples[start + i]) / 32768.0f;
        centered = apply_frontend_dsp_sample(raw_sample, feature_state) - mean;
      }
      sum_abs += fabsf(centered);
      if (centered < centered_min) centered_min = centered;
      if (centered > centered_max) centered_max = centered;
      sum_sq += centered * centered;

      if (i > 0) {
        sum_diff += fabsf(centered - previous);
        const bool current_positive = centered >= 0.0f;
        if (current_positive != previous_positive) {
          sign_changes++;
        }
        previous_positive = current_positive;
      }

      previous = centered;
    }

    const float denominator = static_cast<float>(frame_size);
    const float delta_denominator = denominator > 1.0f ? denominator - 1.0f : 1.0f;
    const float energy_feature = (sum_abs / denominator) * ENERGY_FEATURE_GAIN;
    const float diff_feature = (sum_diff / delta_denominator) * DIFF_FEATURE_GAIN;
    const float zcr_feature = (static_cast<float>(sign_changes) / delta_denominator) * ZCR_FEATURE_GAIN;

    write_feature_to_input(frame_index, energy_feature);
    write_feature_to_input(VOICE_FEATURE_FRAME_COUNT + frame_index, diff_feature);
    write_feature_to_input((VOICE_FEATURE_FRAME_COUNT * 2) + frame_index, zcr_feature);
  }

  last_centered_rms = sqrtf(sum_sq / static_cast<float>(kWakeWordInputSamples));
  last_centered_p2p = centered_max - centered_min;
}

void smooth_scores(float *scores) {
  if (!SCORE_SMOOTHING_ENABLED) {
    return;
  }

  const float alpha = constrain(SCORE_SMOOTHING_ALPHA, 0.0f, 1.0f);
  if (!smoothed_scores_initialized) {
    for (int i = 0; i < DETECTOR_LABEL_COUNT; i++) {
      smoothed_scores[i] = scores[i];
      scores[i] = smoothed_scores[i];
    }
    smoothed_scores_initialized = true;
    return;
  }

  for (int i = 0; i < DETECTOR_LABEL_COUNT; i++) {
    smoothed_scores[i] = (alpha * scores[i]) + ((1.0f - alpha) * smoothed_scores[i]);
    scores[i] = smoothed_scores[i];
  }
}

float output_score(int index) {
  if (output->type == kTfLiteInt8) {
    return (static_cast<int>(output->data.int8[index]) - output->params.zero_point) * output->params.scale;
  }

  if (output->type == kTfLiteUInt8) {
    return (static_cast<int>(output->data.uint8[index]) - output->params.zero_point) * output->params.scale;
  }

  if (output->type == kTfLiteFloat32) {
    return output->data.f[index];
  }

  return 0.0f;
}

void print_scores() {
  Serial.print("Predictions:");
  for (int i = 0; i < DETECTOR_LABEL_COUNT; i++) {
    Serial.print(" ");
    Serial.print(DETECTOR_LABELS[i]);
    Serial.print("=");
    Serial.print(output_score(i), 3);
  }
  Serial.println();
}

int best_label_index(const float *scores) {
  int best_index = 0;
  for (int i = 1; i < DETECTOR_LABEL_COUNT; i++) {
    if (scores[i] > scores[best_index]) {
      best_index = i;
    }
  }
  return best_index;
}

bool run_inference(float *scores, bool verbose) {
  put_audio_in_input_tensor(audio_window);

  const bool rms_pass = last_centered_rms >= SPEECH_GATE_MIN_CENTERED_RMS;
  const bool p2p_pass = last_centered_p2p >= SPEECH_GATE_MIN_P2P;
  const bool speech_present = SPEECH_GATE_REQUIRE_BOTH_SIGNALS ? (rms_pass && p2p_pass) : (rms_pass || p2p_pass);

  if (SPEECH_GATE_ENABLED && speech_present) {
    speech_gate_hold_until_ms = millis() + SPEECH_GATE_HANGOVER_MS;
  }

  const bool speech_gate_open =
      !SPEECH_GATE_ENABLED || speech_present || millis() <= speech_gate_hold_until_ms;
  if (!speech_gate_open) {
    for (int i = 0; i < DETECTOR_LABEL_COUNT; i++) {
      scores[i] = 0.0f;
    }
    scores[NOT_WAKE_INDEX] = 1.0f;
    smooth_scores(scores);
    if (verbose) {
      Serial.print("Speech gate: no speech -> not_wake (rms=");
      Serial.print(last_centered_rms, 5);
      Serial.print(" p2p=");
      Serial.print(last_centered_p2p, 4);
      Serial.println(")");
    }
    return true;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("TinyML inference failed");
    return false;
  }

  for (int i = 0; i < DETECTOR_LABEL_COUNT; i++) {
    scores[i] = output_score(i);
  }
  smooth_scores(scores);

  if (verbose) {
    Serial.print("Signal centered_rms=");
    Serial.print(last_centered_rms, 5);
    Serial.print(" p2p=");
    Serial.println(last_centered_p2p, 4);
    print_scores();
  }

  return true;
}

bool is_authorized_wake(const float *scores) {
  const float nearest_competing_score =
      scores[UNKNOWN_USER_WAKE_INDEX] > scores[NOT_WAKE_INDEX]
          ? scores[UNKNOWN_USER_WAKE_INDEX]
          : scores[NOT_WAKE_INDEX];
  return best_label_index(scores) == AUTHORIZED_USER_WAKE_INDEX &&
         scores[AUTHORIZED_USER_WAKE_INDEX] >= AUTHORIZED_WAKE_THRESHOLD &&
         (scores[AUTHORIZED_USER_WAKE_INDEX] - nearest_competing_score) >= AUTHORIZED_WAKE_MARGIN;
}

bool is_unknown_user_wake(const float *scores) {
  return best_label_index(scores) == UNKNOWN_USER_WAKE_INDEX &&
         scores[UNKNOWN_USER_WAKE_INDEX] >= UNKNOWN_USER_WAKE_THRESHOLD;
}
