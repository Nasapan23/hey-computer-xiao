import gc
import socket
import time

import camera
import network

try:
    import ujson as json
except ImportError:
    import json

try:
    from secrets import WIFI_PASSWORD, WIFI_SSID
except ImportError:
    WIFI_SSID = "YOUR_WIFI_SSID"
    WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

try:
    import machine
except ImportError:
    machine = None


BOARD_STATUS = {
    "status": "NORMAL",
    "score": None,
    "updated_at_ms": 0,
}
CAMERA_READY = False
LAST_CAMERA_RETRY_MS = 0

LED = None
if machine is not None:
    try:
        LED = machine.Pin(2, machine.Pin.OUT)
        LED.off()
    except Exception:
        LED = None


def set_status(new_status, score=None):
    normalized = str(new_status).upper()
    if normalized not in ("NORMAL", "ABNORMAL"):
        normalized = "NORMAL"

    BOARD_STATUS["status"] = normalized
    BOARD_STATUS["score"] = score
    BOARD_STATUS["updated_at_ms"] = time.ticks_ms()

    if LED is not None:
        if normalized == "ABNORMAL":
            LED.on()
        else:
            LED.off()


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting WiFi...")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        wait_s = 0
        while not wlan.isconnected():
            time.sleep(1)
            wait_s += 1
            if wait_s % 5 == 0:
                print("Still connecting...")
    print("WiFi connected:", wlan.ifconfig())
    return wlan.ifconfig()[0]


def init_camera():
    init_attempts = [
        ("init()", lambda: camera.init()),
        ("init(0)", lambda: camera.init(0)),
    ]

    initialized = False
    for label, init_call in init_attempts:
        try:
            result = init_call()
            if result is False:
                print("Camera init returned False for", label)
                continue
            initialized = True
            print("Camera initialized with", label)
            break
        except Exception as exc:
            print("Camera init failed for", label, ":", exc)
            time.sleep_ms(120)

    if not initialized:
        return False

    # This firmware exposes camera tuning via dedicated functions.
    try:
        camera.framesize(10)  # 800x600 on KAKI5 build
    except Exception:
        pass
    try:
        camera.quality(12)
    except Exception:
        pass
    return True


def capture_jpeg(max_retries=3):
    global CAMERA_READY, LAST_CAMERA_RETRY_MS

    if not CAMERA_READY:
        now = time.ticks_ms()
        if time.ticks_diff(now, LAST_CAMERA_RETRY_MS) > 3000:
            LAST_CAMERA_RETRY_MS = now
            CAMERA_READY = init_camera()
            if not CAMERA_READY:
                print("Camera not ready yet; retry later.")
        return None

    for _ in range(max_retries):
        try:
            frame = camera.capture()
            if frame and hasattr(frame, "__len__") and len(frame) > 0:
                return frame
            if frame is False:
                CAMERA_READY = False
                return None
        except Exception as exc:
            print("Capture failed:", exc)
            CAMERA_READY = False
        time.sleep_ms(120)
    return None


def send_all(client, payload):
    view = memoryview(payload)
    while len(view):
        sent = client.send(view)
        if sent <= 0:
            break
        view = view[sent:]


def send_text(client, code, content_type, body_text):
    body = body_text.encode("utf-8")
    headers = (
        "HTTP/1.1 {code}\r\n"
        "Content-Type: {ctype}\r\n"
        "Content-Length: {length}\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-store\r\n\r\n"
    ).format(code=code, ctype=content_type, length=len(body))
    send_all(client, headers.encode("utf-8"))
    send_all(client, body)


def send_json(client, code, payload):
    send_text(client, code, "application/json", json.dumps(payload))


def send_jpeg(client, jpeg_bytes):
    headers = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: {length}\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-store\r\n\r\n"
    ).format(length=len(jpeg_bytes))
    send_all(client, headers.encode("utf-8"))
    send_all(client, jpeg_bytes)


def parse_request(raw):
    req_text = raw.decode("utf-8", "ignore")
    first_line = req_text.split("\r\n", 1)[0]
    parts = first_line.split(" ")
    method = parts[0] if len(parts) > 0 else "GET"
    path = parts[1] if len(parts) > 1 else "/"
    body = b""
    sep = b"\r\n\r\n"
    if sep in raw:
        body = raw.split(sep, 1)[1]
    return method, path, body


def web_page():
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>XIAO Visual Guard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .box { max-width: 640px; }
    .status { font-size: 1.1rem; margin-bottom: 12px; }
    img { width: 100%; border: 1px solid #ccc; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="box">
    <h2>XIAO Visual Fingerprint Guard</h2>
    <div class="status" id="status">Status: loading...</div>
    <img id="cam" src="/capture" alt="camera frame" />
  </div>
  <script>
    async function refresh() {
      const res = await fetch('/health');
      const data = await res.json();
      document.getElementById('status').textContent =
        `Status: ${data.status} | score: ${data.score}`;
      document.getElementById('cam').src = '/capture?t=' + Date.now();
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


def handle_request(client, raw_request):
    method, path, body = parse_request(raw_request)
    route = path.split("?", 1)[0]

    if method == "GET" and route == "/":
        send_text(client, "200 OK", "text/html", web_page())
        return

    if method == "GET" and route == "/health":
        uptime_ms = time.ticks_ms()
        payload = {
            "status": BOARD_STATUS["status"],
            "score": BOARD_STATUS["score"],
            "updated_at_ms": BOARD_STATUS["updated_at_ms"],
            "uptime_ms": uptime_ms,
            "camera_ready": CAMERA_READY,
        }
        send_json(client, "200 OK", payload)
        return

    if method == "GET" and route == "/capture":
        jpeg = capture_jpeg()
        if jpeg is None:
            send_json(client, "503 Service Unavailable", {"error": "capture_failed"})
            return
        send_jpeg(client, jpeg)
        return

    if method == "POST" and route == "/status":
        try:
            payload = json.loads(body.decode("utf-8"))
            status = payload.get("status", "NORMAL")
            score = payload.get("score")
            set_status(status, score)
            send_json(
                client,
                "200 OK",
                {"ok": True, "status": BOARD_STATUS["status"], "score": BOARD_STATUS["score"]},
            )
            return
        except Exception as exc:
            send_json(client, "400 Bad Request", {"ok": False, "error": str(exc)})
            return

    send_json(client, "404 Not Found", {"error": "not_found"})


def serve_forever(ip):
    addr = socket.getaddrinfo("0.0.0.0", 80)[0][-1]
    sock = socket.socket()
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass
    sock.bind(addr)
    sock.listen(2)
    print("HTTP server ready: http://{}/".format(ip))

    while True:
        client = None
        try:
            client, remote_addr = sock.accept()
            raw = client.recv(4096)
            if raw:
                handle_request(client, raw)
            print("Handled request from", remote_addr)
        except Exception as exc:
            print("Request error:", exc)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            gc.collect()


def main():
    global CAMERA_READY
    set_status("NORMAL", None)
    ip = connect_wifi()
    CAMERA_READY = init_camera()
    if not CAMERA_READY:
        print("Camera init failed, server will continue and retry in background.")
    serve_forever(ip)


main()
