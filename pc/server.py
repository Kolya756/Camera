import asyncio
import datetime
import ipaddress
import json
import math
import os
import platform
import re
import socket
import ssl
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BOXY_DATA") or Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "boxy")
MODEL = os.environ.get("BOXY_MODEL", "yolo26x.pt")
MAX_SIDE = int(os.environ.get("BOXY_MAX", "1280"))
HTTPS_PORT = int(os.environ.get("BOXY_PORT", "8443"))
HTTP_PORT = int(os.environ.get("BOXY_HTTP", "8000"))
DEVICE = os.environ.get("BOXY_DEVICE", "")
KEEP = 0.15
IOU = 0.45
TOP = 100
TRACK = dict(tracker_type="botsort", track_high_thresh=0.3, track_low_thresh=0.12, new_track_thresh=0.4, track_buffer=60,
             match_thresh=0.8, fuse_score=True, gmc_method="sparseOptFlow", proximity_thresh=0.5, appearance_thresh=0.8,
             with_reid=False, model="auto")
SHOW = {"person": 0.5}
SHOW_OTHER = 0.4
REID_SEC = 10.0
REID_SIM = 0.6
PET = {"dog": 1.5, "cat": 1.5}
WANT = {"person": "person", "bird": "bird", "cat": "cat", "dog": "dog", "horse": "horse", "sheep": "sheep", "cow": "cow", "elephant": "elephant", "bear": "bear", "zebra": "zebra", "giraffe": "giraffe", "teddy bear": "dog"}


def say(text=""):
    print(text, flush=True)


class Detector:
    def __init__(self, weights, device=""):
        import torch
        import torchvision
        from ultralytics import YOLO

        self.torch = torch
        self.batched_nms = torchvision.ops.batched_nms
        cuda = torch.cuda.is_available() and device != "cpu"
        self.dev = torch.device("cuda" if cuda else "cpu")
        self.half = cuda
        self.dtype = torch.float16 if cuda else torch.float32
        self.device_name = torch.cuda.get_device_name(0) if cuda else (platform.processor() or "CPU")
        y = YOLO(str(weights))
        y.fuse()
        net = y.model.to(self.dev).eval()
        self.net = net.half() if cuda else net
        names = net.names if isinstance(net.names, dict) else dict(enumerate(net.names))
        self.nc = len(names)
        self.labels = sorted(set(WANT.values()))
        group = [self.labels.index(WANT[names[i]]) if names[i] in WANT else -1 for i in range(self.nc)]
        self.group = torch.tensor(group, device=self.dev)
        self.name = Path(str(weights)).stem
        if cuda:
            torch.backends.cudnn.benchmark = True

    def warmup(self):
        for h, w in ((960, 544), (544, 960), (1280, 720), (720, 1280)):
            img = np.full((h, w, 3), 114, np.uint8)
            for _ in range(3):
                self(img)

    def run(self, img):
        torch = self.torch
        h, w = img.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1:
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        H, W = -(-h // 32) * 32, -(-w // 32) * 32
        empty = (img, np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32), None)
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(img)).to(self.dev)
            t = t.permute(2, 0, 1).flip(0).unsqueeze(0).to(self.dtype).div_(255)
            if H != h or W != w:
                t = torch.nn.functional.pad(t, (0, W - w, 0, H - h), value=114 / 255)
            out = self.net(t)
            p = (out[0] if isinstance(out, (list, tuple)) else out)[0]
            conf, cls = p[4:4 + self.nc].max(0)
            g = self.group[cls]
            idx = torch.nonzero((conf >= KEEP) & (g >= 0)).flatten()
            if idx.numel() == 0:
                return empty
            conf = conf[idx].float()
            g = g[idx]
            b = p[:4, idx].float()
            xyxy = torch.stack((b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2), 1)
            keep = self.batched_nms(xyxy, conf, g, IOU)[:TOP]
            xyxy = xyxy[keep]
            xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
            xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
            res = (img, xyxy.cpu().numpy(), conf[keep].cpu().numpy(), g[keep].float().cpu().numpy(), None)
        return res

    def __call__(self, img):
        img, xyxy, conf, lab, _ = self.run(img)
        h, w = img.shape[:2]
        dets = []
        for (x1, y1, x2, y2), c, k in zip(xyxy.tolist(), conf.tolist(), lab.tolist()):
            if x2 - x1 > 1 and y2 - y1 > 1:
                dets.append([self.labels[int(k)], round(c, 3), round(x1 / w, 5), round(y1 / h, 5), round((x2 - x1) / w, 5), round((y2 - y1) / h, 5)])
        return dets


def hist_of(img, b):
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    w, h = x2 - x1, y2 - y1
    xa, ya, xb, yb = int(x1 + 0.2 * w), int(y1 + 0.15 * h), int(x1 + 0.8 * w), int(y1 + 0.85 * h)
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(img.shape[1], xb), min(img.shape[0], yb)
    if xb - xa < 4 or yb - ya < 4:
        return None
    q = cv2.resize(img[ya:yb, xa:xb], (16, 16), interpolation=cv2.INTER_AREA).reshape(-1, 3) >> 6
    return np.bincount(q[:, 2].astype(np.int64) * 16 + q[:, 1] * 4 + q[:, 0], minlength=64).astype(np.float32) / 256


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


class Session:
    def __init__(self, labels):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace

        self.make = lambda: BOTSORT(IterableSimpleNamespace(**TRACK))
        self.labels = labels
        self.lock = threading.Lock()
        self.zones, self.lines, self.mode = [], [], "all"
        self.zstat, self.lstat = [], []
        self.used = time.monotonic()
        self.reset()

    def reset(self):
        self.tracker = self.make()
        self.ids, self.info, self.votes, self.peak = {}, {}, {}, {}
        self.gone = []
        self.next = 1
        self.seen = set()
        self.reid = 0
        self.side, self.zin = {}, {}
        self.last = 0.0
        self.dt = 1 / 30

    def command(self, m):
        with self.lock:
            if m.get("reset"):
                self.reset()
            cfg = m.get("cfg")
            if not isinstance(cfg, dict):
                return
            zones = [[float(v) for v in z[:4]] for z in cfg.get("zones", []) if len(z) >= 4]
            lines = [[float(v) for v in l[:4]] for l in cfg.get("lines", []) if len(l) >= 4]
            if zones != self.zones:
                old = dict(zip(map(tuple, self.zones), self.zstat))
                self.zstat = [old.get(tuple(z), [0, 0]) for z in zones]
                self.zones = zones
                self.zin = {}
            if lines != self.lines:
                old = dict(zip(map(tuple, self.lines), self.lstat))
                self.lstat = [old.get(tuple(l), [0, 0]) for l in lines]
                self.lines = lines
                self.side = {}
            if cfg.get("mode") in ("person", "animal", "all"):
                self.mode = cfg["mode"]

    def match(self, f, grp, b, now, side):
        best, bs = None, -1.0
        sz0 = max(b[2] - b[0], b[3] - b[1])
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for g in self.gone:
            if g["grp"] != grp:
                continue
            gb = g["box"]
            ratio = (b[3] - b[1]) / max(1.0, gb[3] - gb[1])
            if ratio < 0.5 or ratio > 2:
                continue
            el = now - g["t"]
            sz = max(gb[2] - gb[0], gb[3] - gb[1])
            px = (gb[0] + gb[2]) / 2 + g["v"][0] * min(el, 1.5)
            py = (gb[1] + gb[3]) / 2 + g["v"][1] * min(el, 1.5)
            dist = math.hypot(cx - px, cy - py)
            reach = max(side * 0.0625, 1.5 * sz * (1 + el))
            score = -1.0
            if f is not None and g["feat"] is not None and sz >= side * 0.03:
                sim = float(np.minimum(f, g["feat"]).sum())
                if sim >= REID_SIM and (dist <= reach or g["edge"]):
                    score = sim + (0.2 if dist <= reach else 0.0)
            elif dist <= reach * 0.6 and el < 3:
                score = 0.5
            if score > bs:
                best, bs = g, score
        if best is not None:
            self.gone = [g for g in self.gone if g is not best]
            self.reid += 1
            return best["did"]
        did = self.next
        self.next += 1
        return did

    def step(self, img, xyxy, conf, lab, feats):
        from ultralytics.engine.results import Boxes

        with self.lock:
            now = time.monotonic()
            if self.last:
                self.dt = self.dt * 0.9 + min(0.5, max(0.005, now - self.last)) * 0.1
            self.last = now
            self.used = now
            h, w = img.shape[:2]
            side = max(w, h)
            n = len(conf)
            data = np.concatenate([xyxy, conf[:, None], lab[:, None]], 1).astype(np.float32) if n else np.zeros((0, 6), np.float32)
            out = self.tracker.update(Boxes(data, (h, w)), img)
            fid = self.tracker.frame_id
            live = {}
            for r in out:
                live[int(r[4])] = (r[:4].astype(float), float(r[5]), int(r[6]), False)
            for t in self.tracker.lost_stracks:
                if t.is_activated and fid - t.end_frame <= 3 and t.track_id not in live:
                    live[t.track_id] = (np.asarray(t.xyxy, float), float(t.score), int(t.cls), True)
            vel = {t.track_id: (float(t.mean[4]) / self.dt, float(t.mean[5]) / self.dt) for t in self.tracker.tracked_stracks if t.mean is not None}
            alive = {t.track_id for t in self.tracker.tracked_stracks} | {t.track_id for t in self.tracker.lost_stracks}
            for tid in [k for k in self.ids if k not in alive]:
                did = self.ids.pop(tid)
                inf = self.info.pop(tid, None)
                if inf is not None:
                    self.gone.append(dict(inf, did=did, t=now))
            self.gone = [g for g in self.gone if now - g["t"] < REID_SEC][-40:]
            boxes = {tid: v[0] for tid, v in live.items() if not v[3]}
            m = side * 0.04
            tracks = []
            for tid, (b, sc, k, lost) in live.items():
                name = self.labels[k] if 0 <= k < len(self.labels) else "person"
                if tid not in self.ids:
                    if lost:
                        continue
                    f = hist_of(img, b)
                    self.ids[tid] = self.match(f, "person" if name == "person" else "animal", b, now, side)
                    self.info[tid] = {"feat": f, "ft": now}
                did = self.ids[tid]
                v = self.votes.setdefault(did, {})
                if not lost:
                    for c in v:
                        v[c] *= 0.8
                    v[name] = v.get(name, 0.0) + sc * PET.get(name, 1.0)
                    self.peak[did] = max(self.peak.get(did, 0.0), sc)
                label = max(v, key=v.get) if v else name
                grp = "person" if label == "person" else "animal"
                if not lost:
                    inf = self.info.setdefault(tid, {"feat": None, "ft": 0.0})
                    if now - inf.get("ft", 0.0) >= 1.0 and all(iou(b, o) < 0.3 for t2, o in boxes.items() if t2 != tid):
                        f = hist_of(img, b)
                        if f is not None:
                            inf["feat"] = f if inf.get("feat") is None else inf["feat"] * 0.5 + f * 0.5
                            inf["ft"] = now
                    inf.update(grp=grp, box=b, v=vel.get(tid, (0.0, 0.0)), edge=b[0] <= m or b[1] <= m or b[2] >= w - m or b[3] >= h - m)
                    if self.peak.get(did, 0.0) >= SHOW.get(label, SHOW_OTHER):
                        self.seen.add(did)
                if self.peak.get(did, 0.0) < SHOW.get(label, SHOW_OTHER):
                    continue
                x1, y1 = max(0.0, min(w, b[0])), max(0.0, min(h, b[1]))
                x2, y2 = max(0.0, min(w, b[2])), max(0.0, min(h, b[3]))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                tracks.append([did, label, round(sc, 3), round(x1 / w, 5), round(y1 / h, 5), round((x2 - x1) / w, 5), round((y2 - y1) / h, 5), 1 if lost else 0])
            if len(self.votes) > 500:
                keep = set(self.ids.values()) | {g["did"] for g in self.gone}
                self.votes = {k: v for k, v in self.votes.items() if k in keep}
                self.peak = {k: v for k, v in self.peak.items() if k in keep}
            self.count(tracks)
            return tracks, {"u": len(self.seen), "r": self.reid, "z": self.zstat, "l": self.lstat}

    def count(self, tracks):
        for z in self.zstat:
            z[0] = 0
        for did, label, sc, x, y, bw, bh, lost in tracks:
            if lost:
                continue
            grp = "person" if label == "person" else "animal"
            if self.mode != "all" and self.mode != grp:
                continue
            cx = x + bw / 2
            zi = self.zin.setdefault(did, {})
            fy = y + bh * 0.85
            for i, z in enumerate(self.zones):
                inside = z[0] <= cx <= z[0] + z[2] and z[1] <= fy <= z[1] + z[3]
                if inside:
                    self.zstat[i][0] += 1
                    if not zi.get(i):
                        self.zstat[i][1] += 1
                zi[i] = inside
            sd = self.side.setdefault(did, {})
            fy = y + bh
            for i, l in enumerate(self.lines):
                sgn = (l[2] - l[0]) * (fy - l[1]) - (l[3] - l[1]) * (cx - l[0])
                sgn = 1 if sgn > 0 else -1 if sgn < 0 else 0
                prev = sd.get(i)
                if prev is not None and sgn != 0 and prev != 0 and sgn != prev:
                    self.lstat[i][0 if sgn > 0 else 1] += 1
                if sgn != 0:
                    sd[i] = sgn


def decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8, offset=8), cv2.IMREAD_COLOR)


def private_ips():
    found = []

    def add(ip):
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            return
        if a.version == 4 and a.is_private and not a.is_loopback and not a.is_link_local and ip not in found:
            found.append(ip)

    for target in ("8.8.8.8", "192.168.1.1", "10.0.0.1", "172.16.0.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect((target, 9))
                add(s.getsockname()[0])
            finally:
                s.close()
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    return found


def free_port(ports):
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("0.0.0.0", port))
            return port
        except OSError:
            continue
        finally:
            s.close()
    return None


def host_name():
    h = socket.gethostname().strip().lower()
    return h if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", h or "") else ""


def make_certs(folder, ips):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    host = host_name()
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"Boxy {host or 'PC'} {now:%Y-%m-%d %H:%M}"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Boxy"),
    ])
    usage = dict(content_commitment=False, data_encipherment=False, key_agreement=False, encipher_only=False, decipher_only=False)
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=False, key_cert_sign=True, crl_sign=True, **usage), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    addrs = [ipaddress.ip_address(ip) for ip in ips]
    for net in sorted({ipaddress.ip_network(ip + "/24", strict=False) for ip in ips}, key=str):
        for a in net.hosts():
            if a not in addrs:
                addrs.append(a)
    names = [x509.DNSName("localhost")]
    if host:
        names = [x509.DNSName(host), x509.DNSName(host + ".local")] + names
    san = names + [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))] + [x509.IPAddress(a) for a in addrs]
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ips[0] if ips else "localhost")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=820))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=True, key_cert_sign=False, crl_sign=False, **usage), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    (folder / "leaf.key").write_bytes(key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    (folder / "leaf.pem").write_bytes(leaf.public_bytes(pem) + ca.public_bytes(pem))
    (folder / "ca.crt").write_bytes(ca.public_bytes(serialization.Encoding.DER))
    (folder / "ca.pem").write_bytes(ca.public_bytes(pem))
    (folder / "meta.json").write_text(json.dumps({"ips": ips, "made": now.isoformat(), "ca": ca_name.rfc4514_string()}), encoding="utf-8")


def certs_ok(folder, ip):
    from cryptography import x509

    try:
        leaf = x509.load_pem_x509_certificate((folder / "leaf.pem").read_bytes())
        if not (folder / "leaf.key").exists() or not (folder / "ca.crt").exists():
            return False
        if leaf.not_valid_after_utc < datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30):
            return False
        san = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        return ip is None or ipaddress.ip_address(ip) in san.get_values_for_type(x509.IPAddress)
    except Exception:
        return False


def ca_label(folder):
    try:
        return json.loads((folder / "meta.json").read_text(encoding="utf-8"))["ca"].split(",")[0].split("=", 1)[1]
    except Exception:
        return "Boxy"


GUIDE = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Боксы: подключение</title><style>
body{{margin:0;background:#000;color:#eee;font:16px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;padding:20px 18px 40px}}
h1{{font-size:20px;color:#39ff7a;margin:0 0 14px}}ol{{padding-left:22px}}li{{margin:0 0 14px}}b{{color:#fff}}
a.btn{{display:block;text-align:center;background:#39ff7a;color:#000;text-decoration:none;font-weight:700;border-radius:14px;padding:14px;margin:8px 0 4px}}
a.alt{{background:#222;color:#fff;border:1px solid #555}}small{{opacity:.7}}</style></head><body>
<h1>Боксы: подключение телефона к ПК</h1>
<p>Делается один раз. Нужно, чтобы Safari доверял этому компьютеру и разрешил камеру.</p>
<ol>
<li><a class="btn" href="/ca.crt">Скачать сертификат</a>Safari спросит про профиль конфигурации: нажмите <b>«Разрешить»</b>, затем <b>«Закрыть»</b>.</li>
<li>Откройте <b>Настройки</b>. Вверху появится <b>«Профиль загружен»</b>: нажмите его, затем <b>«Установить»</b>, введите код телефона и ещё раз <b>«Установить»</b>.<br><small>Если пункта нет: Настройки → Основные → VPN и управление устройством → {ca}.</small></li>
<li><b>Настройки → Основные → Об этом устройстве</b>, в самом низу <b>«Доверие сертификатам»</b>: включите переключатель у <b>{ca}</b> и подтвердите.</li>
<li><a class="btn alt" href="{url}">Открыть камеру</a><small>Адрес камеры: {url}. Его можно добавить на экран «Домой» через «Поделиться».</small></li>
</ol>
<p><small>Если ПК написал «новый сертификат», повторите шаги 1-3. Старый профиль {old} можно удалить в «VPN и управление устройством».</small></p>
</body></html>"""


class Stats:
    def __init__(self):
        self.reset()
        self.clients = 0

    def reset(self):
        self.n = 0
        self.skip = 0
        self.dec = 0.0
        self.net = 0.0
        self.trk = 0.0
        self.total = 0.0
        self.bytes = 0
        self.t0 = time.perf_counter()

    def line(self):
        el = time.perf_counter() - self.t0
        if not self.n:
            return ""
        n = self.n
        return (f"телефонов {self.clients} · кадров/с {n / el:.1f} · декод {self.dec / n:.1f} мс · нейросеть {self.net / n:.1f} мс"
                f" · трекинг {self.trk / n:.1f} мс · всего на ПК {self.total / n:.1f} мс · {self.bytes / n / 1024:.0f} КБ/кадр" + (f" · пропущено {self.skip}" if self.skip else ""))


def build_app(det, folder):
    from aiohttp import WSMsgType, web

    stats = Stats()
    active = set()
    sessions = {}
    gpu = ThreadPoolExecutor(1, thread_name_prefix="gpu")
    dec = ThreadPoolExecutor(4, thread_name_prefix="jpeg")
    trk = ThreadPoolExecutor(4, thread_name_prefix="track")
    hello = json.dumps({"hello": 1, "model": det.name, "device": det.device_name, "half": det.half, "max": MAX_SIDE, "track": "botsort"}, ensure_ascii=False)

    async def process(ws, q, sess):
        loop = asyncio.get_running_loop()
        while True:
            data = await q.get()
            while not q.empty():
                old = data
                data = q.get_nowait()
                stats.skip += 1
                await ws.send_str('{"id":%d,"skip":1}' % int.from_bytes(old[:4], "little"))
            fid = int.from_bytes(data[:4], "little")
            t0 = time.perf_counter()
            img = await loop.run_in_executor(dec, decode, data)
            t1 = time.perf_counter()
            if img is None:
                await ws.send_str('{"id":%d,"skip":1}' % fid)
                continue
            run = await loop.run_in_executor(gpu, det.run, img)
            t2 = time.perf_counter()
            tracks, info = await loop.run_in_executor(trk, sess.step, *run)
            t3 = time.perf_counter()
            total = (t3 - t0) * 1000
            await ws.send_str(json.dumps({"id": fid, "t": tracks, "s": info, "ms": round(total, 1)}, separators=(",", ":"), ensure_ascii=False))
            stats.n += 1
            stats.dec += (t1 - t0) * 1000
            stats.net += (t2 - t1) * 1000
            stats.trk += (t3 - t2) * 1000
            stats.total += total
            stats.bytes += len(data) - 8

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10, compress=False, max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        who = request.remote or "?"
        now = time.monotonic()
        for k in [k for k, v in sessions.items() if now - v.used > 900]:
            sessions.pop(k, None)
        sess = sessions.get(who)
        if sess is None:
            sess = sessions[who] = Session(det.labels)
        active.add(ws)
        stats.clients += 1
        say(f"Подключился {who}")
        await ws.send_str(hello)
        q = asyncio.Queue()
        task = asyncio.create_task(process(ws, q, sess))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY and len(msg.data) > 8:
                    q.put_nowait(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        sess.command(json.loads(msg.data))
                    except Exception:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            task.cancel()
            active.discard(ws)
            stats.clients -= 1
            say(f"Отключился {who}")
        return ws

    def page(request):
        return web.FileResponse(HERE / "web" / "index.html", headers={"Cache-Control": "no-store"})

    def ca_file(request):
        return web.Response(body=(folder / "ca.crt").read_bytes(), content_type="application/x-x509-ca-cert",
                            headers={"Content-Disposition": 'attachment; filename="boxy-ca.crt"', "Cache-Control": "no-store"})

    def guide(request):
        url = f"https://{request.url.host}:{HTTPS_PORT}/"
        label = ca_label(folder)
        return web.Response(text=GUIDE.format(url=url, ca=label, old="Boxy"), content_type="text/html", charset="utf-8",
                            headers={"Cache-Control": "no-store"})

    secure = web.Application()
    secure.add_routes([web.get("/", page), web.get("/index.html", page), web.get("/ws", ws_handler), web.get("/ca.crt", ca_file), web.get("/setup", guide)])
    plain = web.Application()
    plain.add_routes([web.get("/", guide), web.get("/ca.crt", ca_file)])
    async def close_all():
        for ws in list(active):
            try:
                await ws.close()
            except Exception:
                pass

    return secure, plain, stats, close_all


def quiet(loop, context):
    if isinstance(context.get("exception"), (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return
    loop.default_exception_handler(context)


async def serve(det, folder, ips):
    from aiohttp import web

    asyncio.get_running_loop().set_exception_handler(quiet)

    secure, plain, stats, close_all = build_app(det, folder)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(folder / "leaf.pem", folder / "leaf.key")
    runners = []
    r = web.AppRunner(secure, access_log=None)
    await r.setup()
    runners.append(r)
    try:
        await web.TCPSite(r, "0.0.0.0", HTTPS_PORT, ssl_context=ctx, shutdown_timeout=1.0).start()
    except OSError as e:
        say(f"Порт {HTTPS_PORT} занят ({e}). Скорее всего сервер уже открыт в другом окне: закройте его и запустите снова.")
        raise SystemExit(1)
    http_port = free_port((HTTP_PORT, 8001, 8080, 8081, 8888))
    if http_port:
        r2 = web.AppRunner(plain, access_log=None)
        await r2.setup()
        runners.append(r2)
        await web.TCPSite(r2, "0.0.0.0", http_port, shutdown_timeout=1.0).start()
    ip = ips[0] if ips else "127.0.0.1"
    line = "=" * 66
    say(line)
    say("  БОКСЫ: сервер готов")
    say(f"  {det.device_name} · {det.name}" + (" · fp16" if det.half else " · процессор") + " · номера и подсчёт на ПК")
    if http_port:
        say(f"  Первый раз на iPhone откройте в Safari:  http://{ip}:{http_port}")
    else:
        say(f"  Первый раз на iPhone откройте в Safari:  https://{ip}:{HTTPS_PORT}/setup")
    say(f"  Камера:                                  https://{ip}:{HTTPS_PORT}")
    for other in ips[1:]:
        say(f"  другой адрес этого ПК:                   https://{other}:{HTTPS_PORT}")
    say("  Остановить: закройте это окно")
    say(line)
    try:
        while True:
            await asyncio.sleep(5)
            text = stats.line()
            if text:
                say(text)
            stats.reset()
    finally:
        await close_all()
        for r in runners:
            await r.cleanup()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.environ.setdefault("YOLO_VERBOSE", "False")
    folder = DATA / "certs"
    ips = private_ips()
    if not ips:
        say("Не нашёл адрес ПК в локальной сети. Подключите ПК к тому же Wi-Fi или роутеру, что и телефон.")
    if not certs_ok(folder, ips[0] if ips else None):
        old = (folder / "leaf.pem").exists()
        make_certs(folder, ips)
        say("")
        say("НОВЫЙ СЕРТИФИКАТ: установите его на iPhone (шаги по адресу :8000 ниже)." if old else "Создан сертификат для iPhone (один раз, шаги по адресу :8000 ниже).")
        say("")
    weights = Path(MODEL) if Path(MODEL).is_absolute() else DATA / "models" / MODEL
    say(f"Загружаю нейросеть {weights.name} (первый запуск скачивает её из интернета)…")
    det = Detector(weights, DEVICE)
    if not det.half:
        say("ВНИМАНИЕ: видеокарта NVIDIA не найдена (CUDA недоступна). Работаю на процессоре, будет медленно.")
        say("Обновите драйвер NVIDIA и запустите start.bat снова.")
    det.warmup()
    Session(det.labels).step(*det.run(np.full((544, 960, 3), 114, np.uint8)))
    try:
        asyncio.run(serve(det, folder, ips))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
