#!/usr/bin/env python3
"""
Прокси автозапуска GPU-сервера на Stop/Start.

Тонкая будилка, не оркестратор: смотрит статус, при необходимости будит,
дожидается готовности и переливает запрос в GPU-сервер без изменений.
Логики продукта внутри нет.

Работает на маленькой всегда включённой машине. Зависимостей нет —
только стандартная библиотека, чтобы не разводить venv на прокси-хосте.

Запуск:
    OS_PASSWORD=... GPU_SERVER_ID=... GPU_SERVER_HOST=... python3 wake_proxy.py

Переменные окружения:
    OS_AUTH_URL         https://api.immers.cloud:5000/v3
    OS_USERNAME         имя пользователя
    OS_PASSWORD         пароль (в логи не попадает)
    OS_PROJECT_ID       id проекта
    OS_USER_DOMAIN_NAME / OS_PROJECT_DOMAIN_NAME   обычно Default
    NOVA_URL            https://api.immers.cloud:8774/v2.1
    GPU_SERVER_ID       uuid GPU-сервера в OpenStack
    GPU_SERVER_HOST     адрес GPU-сервера (куда переливать запрос)
    GPU_SERVER_PORT     8300
    PROXY_PORT          8080
    IDLE_STOP_MINUTES   через сколько простоя гасить; 0 — не гасить
    WAKE_TIMEOUT_S      сколько ждать ACTIVE после Start (по умолчанию 180)
    READY_TIMEOUT_S     сколько ждать demo.ready после ACTIVE (по умолчанию 120)
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CFG = {
    "auth_url":   os.environ.get("OS_AUTH_URL", "https://api.immers.cloud:5000/v3"),
    "username":   os.environ.get("OS_USERNAME", ""),
    "password":   os.environ.get("OS_PASSWORD", ""),
    "project_id": os.environ.get("OS_PROJECT_ID", ""),
    "user_dom":   os.environ.get("OS_USER_DOMAIN_NAME", "Default"),
    "proj_dom":   os.environ.get("OS_PROJECT_DOMAIN_NAME", "Default"),
    "nova":       os.environ.get("NOVA_URL", "https://api.immers.cloud:8774/v2.1"),
    "server_id":  os.environ.get("GPU_SERVER_ID", ""),
    "gpu_host":   os.environ.get("GPU_SERVER_HOST", ""),
    "gpu_port":   int(os.environ.get("GPU_SERVER_PORT", "8300")),
    "port":       int(os.environ.get("PROXY_PORT", "8080")),
    "idle_min":   float(os.environ.get("IDLE_STOP_MINUTES", "40")),
    "wake_to":    float(os.environ.get("WAKE_TIMEOUT_S", "180")),
    "ready_to":   float(os.environ.get("READY_TIMEOUT_S", "120")),
    "page":       os.environ.get("DEMO_PAGE", "/opt/wake-proxy/demo.html"),
}

# Страница отдаётся прокси напрямую и НЕ будит GPU-машину: открытие панели
# ничего не должно стоить. Будит только настоящий заказ (POST).
PAGE_PATHS = ("/", "/demo", "/index.html")

def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------- OpenStack API

class Nova:
    """Минимум вызовов: статус, Start, Stop. Токен кэшируется."""

    def __init__(self):
        self._token = None
        self._expires = 0.0
        self._lock = threading.Lock()

    def _auth(self):
        body = json.dumps({"auth": {
            "identity": {"methods": ["password"], "password": {"user": {
                "name": CFG["username"],
                "domain": {"name": CFG["user_dom"]},
                "password": CFG["password"]}}},
            "scope": {"project": {"id": CFG["project_id"],
                                  "domain": {"name": CFG["proj_dom"]}}}}}).encode()
        req = urllib.request.Request(CFG["auth_url"] + "/auth/tokens", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            tok = r.headers.get("X-Subject-Token")
        if not tok:
            raise RuntimeError("Keystone не вернул токен")
        # Токен живёт часами; обновляем с запасом.
        self._token, self._expires = tok, time.time() + 3000
        log("получен новый токен OpenStack")

    def token(self):
        with self._lock:
            if not self._token or time.time() > self._expires:
                self._auth()
            return self._token

    def _call(self, path, body=None, retry=True):
        req = urllib.request.Request(
            CFG["nova"] + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"X-Auth-Token": self.token(),
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code == 401 and retry:          # токен протух — перевыпустить
                with self._lock:
                    self._token = None
                return self._call(path, body, retry=False)
            raise

    def status(self):
        return self._call(f"/servers/{CFG['server_id']}")["server"]["status"]

    def start(self):
        self._call(f"/servers/{CFG['server_id']}/action", {"os-start": None})

    def stop(self):
        self._call(f"/servers/{CFG['server_id']}/action", {"os-stop": None})


NOVA = Nova()
LAST_REQUEST = time.time()
WAKE_LOCK = threading.Lock()          # будим один раз, даже если запросов много


# ------------------------------------------------------------------ пробуждение

def gpu_url(path):
    return f"http://{CFG['gpu_host']}:{CFG['gpu_port']}{path}"

def demo_ready():
    try:
        with urllib.request.urlopen(gpu_url("/health/ready"), timeout=5) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()                # /health/ready штатно отвечает 503 — это норма
    except Exception:
        return False
    try:
        return bool(json.loads(raw)["models"]["demo"]["ready"])
    except Exception:
        return False

def ensure_awake():
    """Возвращает (ok, сообщение). Держит запрос, пока сервер не готов."""
    with WAKE_LOCK:
        st = NOVA.status()
        if st == "SHUTOFF":
            log("сервер остановлен — Start")
            NOVA.start()
            deadline = time.time() + CFG["wake_to"]
            while time.time() < deadline:
                time.sleep(3)
                if NOVA.status() == "ACTIVE":
                    log(f"ACTIVE через {CFG['wake_to'] - (deadline - time.time()):.0f} с")
                    break
            else:
                return False, "сервер не поднялся за отведённое время"
        elif st != "ACTIVE":
            # BUILD, а также любые переходные состояния
            return False, f"сервер в состоянии {st}, повторите позже"

        deadline = time.time() + CFG["ready_to"]
        while time.time() < deadline:
            if demo_ready():
                return True, "готов"
            time.sleep(3)
        return False, "сервис не дошёл до готовности"


def idle_watch():
    """Гасит машину после простоя. Отдельный поток."""
    if CFG["idle_min"] <= 0:
        log("автогашение выключено")
        return
    while True:
        time.sleep(30)
        idle = (time.time() - LAST_REQUEST) / 60
        if idle < CFG["idle_min"]:
            continue
        try:
            if NOVA.status() == "ACTIVE":
                log(f"простой {idle:.0f} мин — Stop")
                NOVA.stop()
        except Exception as e:
            log(f"не удалось погасить: {e}")


# ---------------------------------------------------------------------- сервер

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log("proxy " + fmt % args)

    def _reply(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(
            payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/health":
            try:
                st = NOVA.status()
                self._reply(200, {"proxy": "ok", "gpu_server": st,
                                  "demo_ready": demo_ready() if st == "ACTIVE" else False,
                                  "idle_min": round((time.time() - LAST_REQUEST) / 60, 1)})
            except Exception as e:
                self._reply(503, {"proxy": "ok", "error": str(e)})
            return

        if path in PAGE_PATHS:
            try:
                body = open(CFG["page"], "rb").read()
            except OSError as e:
                self._reply(500, {"detail": f"страница не читается: {e}"})
                return
            self._reply(200, body, "text/html; charset=utf-8")
            return

        if path == "/favicon.ico":
            self._reply(204, b"", "image/x-icon")
            return

        # Прочие GET проксируются, но НЕ будят машину: опрос здоровья и случайный
        # трафик из интернета не должны включать карту и жечь деньги.
        self._forward(wake=False)

    def do_POST(self):
        self._forward(wake=True)

    def _forward(self, wake=True):
        global LAST_REQUEST

        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length) if length else None

        if wake:
            LAST_REQUEST = time.time()
            try:
                ok, msg = ensure_awake()
            except Exception as e:
                log(f"ошибка пробуждения: {e}")
                self._reply(503, {"detail": f"не удалось разбудить сервер: {e}"})
                return
            if not ok:
                # 504 — не уложились по времени; панель различает это и «занято».
                self._reply(504, {"detail": msg})
                return
        else:
            try:
                if NOVA.status() != "ACTIVE":
                    self._reply(503, {"detail": "сервер рендера сейчас выключен; "
                                                "он включится при первом заказе"})
                    return
            except Exception as e:
                self._reply(503, {"detail": f"статус сервера недоступен: {e}"})
                return

        req = urllib.request.Request(
            gpu_url(self.path), data=payload,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
            method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                self._reply(r.status, r.read(),
                            r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            # Ошибку GPU-сервера отдаём как есть, не подменяя своей.
            self._reply(e.code, e.read(),
                        e.headers.get("Content-Type", "application/json"))
        except Exception as e:
            self._reply(502, {"detail": f"GPU-сервер недоступен: {e}"})
        finally:
            if wake:
                LAST_REQUEST = time.time()


def main():
    missing = [k for k in ("username", "password", "project_id", "server_id", "gpu_host")
               if not CFG[k]]
    if missing:
        sys.exit(f"не заданы переменные: {', '.join(missing)}")

    log(f"прокси на :{CFG['port']} -> {CFG['gpu_host']}:{CFG['gpu_port']}")
    log(f"сервер {CFG['server_id']}, автогашение через {CFG['idle_min']} мин простоя")
    try:
        log(f"статус GPU-сервера сейчас: {NOVA.status()}")
    except Exception as e:
        log(f"ВНИМАНИЕ: API недоступен: {e}")

    threading.Thread(target=idle_watch, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", CFG["port"]), Handler).serve_forever()


if __name__ == "__main__":
    main()
