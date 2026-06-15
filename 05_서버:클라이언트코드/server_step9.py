"""
server_step9_keypoint_only.py — RGB 도킹 서버 (Step 9: 키포인트 기반 접근 + yaw 정면정렬 + open-loop 진입)
AprilTag 의존성을 제거한 버전입니다.
역할 분담:
  - best.pt 키포인트 + solvePnP  → 거리 z 계산
  - best.pt 키포인트 중심        → 좌우정렬 errx 계산
  - solvePnP rvec                → 정면각 yaw 계산 (AprilTag 대체)
핵심:
  - 10cm(ALIGN_Z) 도달 시 errx(좌우중심) + yaw(정면각)가 모두 맞아야 FINAL_APPROACH 진입.
  - 정면이 안 맞으면 후진하며 yaw를 0으로 맞춘 뒤 재접근(REALIGN).
  - 정면이 맞으면 open-loop 직진으로 곧장 도킹.
실행: python3 server_step9_keypoint_only.py
      python3 server_step9_keypoint_only.py --dry-run
확인: http://GPU서버IP:8081
"""
import argparse, json, time, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zmq, cv2, numpy as np
from ultralytics import YOLO
# ── 경로 ────────────────────────────────────────────────────────
PORT, VIEW_PORT = 5556, 8081
ASSET = Path.home() / "server_test" / "step7_assets"
MODEL_PATH = ASSET / "best.pt"
K_PATH, D_PATH, M3D_PATH = ASSET / "rgb_K.npy", ASSET / "rgb_D.npy", ASSET / "pallet_3d_model_mockup.npy"
# ── 거리/정렬 목표 ───────────────────────────────────────────────
ALIGN_Z    = 0.10
TOL_Z      = 0.015
TOL_XPX    = 0.04
TOL_YAW    = 0.07          # 정면각 허용오차(rad, 약 4도)
DOCK_GAP_M = ALIGN_Z - 0.01
# ── 제어 게인 ────────────────────────────────────────────────────
KP_LIN   = 0.40
KP_ANG   = 1.20
KP_YAW   = 1.50           # yaw 정렬 회전 게인
MAX_LIN  = 0.10
MAX_ANG  = 0.50
FINAL_LIN    = 0.03
REALIGN_BACK = -0.04
REALIGN_Z    = 0.15
MAX_REALIGN  = 4
# ── 검출 파라미터 ───────────────────────────────────────────────
YOLO_CONF = 0.20
KP_CONF   = 0.15
MIN_KP    = 5
Z_BIAS    = 0.03
# ── 충돌 가드 (비활성) ──────────────────────────────────────────
ENABLE_COLLISION = False
COLLISION_MIN_M  = 0.08
COLLISION_X_HALF = 0.15
PC_SCALE = 1.0
latest_jpeg = None
RGB_K = RGB_D = M3D = None
STATE = {"final_t0": None, "realign": False, "attempts": 0}
DTYPE_LE = {1:"i1",2:"u1",3:"<i2",4:"<u2",5:"<i4",6:"<u4",7:"<f4",8:"<f8"}
DTYPE_BE = {1:"i1",2:"u1",3:">i2",4:">u2",5:">i4",6:">u4",7:">f4",8:">f8"}
def clamp(v, lo, hi):
    return max(lo, min(hi, v))
# ── PointCloud2 raw → xyz (충돌가드용) ──────────────────────────
def build_dtype(fields, point_step, big=False):
    dm = DTYPE_BE if big else DTYPE_LE
    names, fmts, offs = [], [], []
    for f in fields:
        dt = int(f["datatype"])
        if dt not in dm:
            continue
        fmt = dm[dt]
        if int(f.get("count", 1)) > 1:
            fmt = (fmt, int(f["count"]))
        names.append(f["name"])
        fmts.append(fmt)
        offs.append(int(f["offset"]))
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"no xyz fields: {names}")
    return np.dtype({
        "names": names,
        "formats": fmts,
        "offsets": offs,
        "itemsize": int(point_step)
    })
def cloud_to_xyz(meta, buf):
    c = meta.get("cloud", {})
    w, h = int(c.get("width", 0)), int(c.get("height", 0))
    ps, rs = int(c.get("point_step", 0)), int(c.get("row_step", 0))
    fields = c.get("fields", [])
    if w <= 0 or h <= 0 or ps <= 0 or not fields or isinstance(fields[0], str):
        return np.empty((0, 3), np.float32)
    dt = build_dtype(fields, ps, bool(c.get("is_bigendian", False)))
    if len(buf) < rs * h:
        return np.empty((0, 3), np.float32)
    if rs == w * ps:
        arr = np.frombuffer(buf[:w * h * ps], dtype=dt, count=w * h)
    else:
        rows = [
            np.frombuffer(buf[r * rs:r * rs + w * ps], dtype=dt, count=w)
            for r in range(h)
        ]
        arr = np.concatenate(rows)
    xyz = np.column_stack([
        np.asarray(arr["x"], np.float32),
        np.asarray(arr["y"], np.float32),
        np.asarray(arr["z"], np.float32)
    ])
    return xyz[np.isfinite(xyz).all(axis=1)] * PC_SCALE
def front_min_dist(xyz):
    if len(xyz) == 0:
        return None
    x, z = xyz[:, 0], xyz[:, 2]
    m = (np.abs(x) <= COLLISION_X_HALF) & (z > 0.15) & (z < 2.0)
    if not np.any(m):
        return None
    return float(np.min(z[m]))
# ── rvec → yaw (팔레트 정면 대비 좌우 회전각, rad) ──────────────
def rvec_to_yaw(rvec):
    R, _ = cv2.Rodrigues(rvec)
    # 카메라 좌표계에서 팔레트 평면의 법선(모델 +Z)이 카메라를 향하는 정도로 yaw 추정.
    # yaw = atan2(R[0,2], R[2,2]) → 좌우로 비스듬히 본 각도.
    return float(np.arctan2(R[0, 2], R[2, 2]))
# ── pose: best.pt → 거리 z + 좌우정렬 errx + 정면각 yaw ─────────
def estimate(img, model):
    r = model(img, conf=YOLO_CONF, verbose=False)[0]
    if r.keypoints is None or r.keypoints.xy.shape[0] == 0:
        return {"kp": None, "valid": None, "z": None, "errx": None, "yaw": None}
    kp = r.keypoints.xy.cpu().numpy()[0].astype(np.float64)
    if r.keypoints.conf is not None:
        cf = r.keypoints.conf.cpu().numpy()[0]
    else:
        cf = np.ones(len(kp))
    valid = cf >= KP_CONF
    if int(valid.sum()) < MIN_KP or kp.shape[0] != M3D.shape[0]:
        return {"kp": kp, "valid": valid, "z": None, "errx": None, "yaw": None}
    ok, rvec, tvec = cv2.solvePnP(
        M3D[valid],
        kp[valid],
        RGB_K,
        RGB_D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return {"kp": kp, "valid": valid, "z": None, "errx": None, "yaw": None}
    z = float(tvec.ravel()[2]) / 1000.0 - Z_BIAS
    yaw = rvec_to_yaw(rvec)
    W = img.shape[1]
    errx = (float(np.mean(kp[valid][:, 0])) - W / 2.0) / W
    return {"kp": kp, "valid": valid, "z": z, "errx": errx, "yaw": yaw}
# ── 상태머신 + 제어 ─────────────────────────────────────────────
def reset_state():
    STATE["final_t0"] = None
    STATE["realign"] = False
    STATE["attempts"] = 0
    STATE["stopped"] = False     # 10cm 도달해 한번 정지했는지

def compute(pose, front_d, dry):
    if ENABLE_COLLISION and front_d is not None and front_d < COLLISION_MIN_M:
        reset_state()
        return 0.0, 0.0, "COLLISION_STOP"

    # ── FINAL_APPROACH: open-loop 직진. 키포인트/PnP 완전 무시 ──
    # (근접 시 키포인트가 하단으로 쏠려도 영향받지 않도록 타이머 직진만 수행)
    if STATE["final_t0"] is not None:
        if time.time() - STATE["final_t0"] < DOCK_GAP_M / FINAL_LIN:
            return (0.0 if dry else FINAL_LIN), 0.0, "FINAL_APPROACH"
        return 0.0, 0.0, "DOCKED"

    # SEARCH — 팔레트 거리 없음
    if pose is None or pose["z"] is None:
        STATE["stopped"] = False
        return 0.0, 0.0, "SEARCH"

    z, errx, yaw = pose["z"], pose["errx"], pose["yaw"]

    # ── 10cm 도달 판정 ──
    at_align = z <= ALIGN_Z + TOL_Z

    # ── ① 10cm 도달 시 우선 한번 완전 정지 ──
    if at_align and not STATE["stopped"]:
        STATE["stopped"] = True
        return 0.0, 0.0, "STOP_AT_ALIGN"

    # ── ② 정지 후: 제자리 회전으로 정면(yaw) + 좌우중심(errx) 정렬 ──
    if STATE["stopped"]:
        centered = abs(errx) < TOL_XPX
        facing   = (yaw is not None) and (abs(yaw) < TOL_YAW)

        # 너무 멀어졌으면(팔레트 놓침/뒤로 밀림) 정지상태 해제하고 재접근
        if z > ALIGN_Z + TOL_Z * 2:
            STATE["stopped"] = False
            return 0.0, 0.0, "SEARCH"

        if centered and facing:
            # 정면 정렬 완료 → open-loop 직진 도킹 시작
            STATE["final_t0"] = time.time()
            return (0.0 if dry else FINAL_LIN), 0.0, "FINAL_APPROACH"

        # yaw를 우선 정렬, yaw가 맞으면 errx 미세 정렬 (모두 제자리 회전)
        if not facing:
            steer = clamp(-KP_YAW * yaw, -MAX_ANG, MAX_ANG)
            tag = "YAW_ALIGN"
        else:
            steer = clamp(-KP_ANG * errx, -MAX_ANG, MAX_ANG)
            tag = "X_ALIGN"
        if dry:
            steer = 0.0
        return 0.0, round(steer, 3), tag

    # ── APPROACH: 10cm까지 거리 줄이며 접근 ──
    steer = clamp(-KP_ANG * errx, -MAX_ANG, MAX_ANG)
    lin = clamp(KP_LIN * (z - ALIGN_Z), 0.0, MAX_LIN)
    if abs(errx) > TOL_XPX * 2:
        lin *= 0.4
    if dry:
        lin, steer = 0.0, 0.0
    return round(lin, 3), round(steer, 3), "APPROACH"
# ── 시각화 ──────────────────────────────────────────────────────
def draw(img, pose, lin, ang, state, front_d):
    out = img.copy()
    c_ok, c_bad = (0, 255, 0), (0, 0, 255)
    if pose and pose.get("kp") is not None:
        for pt, v in zip(pose["kp"], pose["valid"]):
            if not np.isfinite(pt).all():
                continue
            cv2.circle(
                out,
                (int(pt[0]), int(pt[1])),
                4 if v else 2,
                c_ok if v else (0, 255, 255),
                -1
            )
    z = f"{pose['z']:.3f}" if pose and pose.get("z") is not None else "--"
    ex = f"{pose['errx']:+.3f}" if pose and pose.get("errx") is not None else "--"
    yw = f"{np.degrees(pose['yaw']):+.1f}" if pose and pose.get("yaw") is not None else "--"
    good = state in ("APPROACH", "FINAL_APPROACH", "DOCKED")
    cv2.putText(
        out,
        f"{state}  z={z}m errx={ex} yaw={yw}deg",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        c_ok if good else c_bad,
        2
    )
    cv2.putText(
        out,
        f"lin={lin:+.3f} ang={ang:+.3f}",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )
    return out
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        return
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            f = latest_jpeg
            if f is None:
                time.sleep(0.03)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(f)))
                self.end_headers()
                self.wfile.write(f)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
            except Exception:
                break
def main():
    global RGB_K, RGB_D, M3D, latest_jpeg
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    RGB_K = np.load(K_PATH).astype(np.float64)
    RGB_D = np.load(D_PATH).astype(np.float64).reshape(1, -1)
    M3D = np.load(M3D_PATH).astype(np.float64)
    model = YOLO(str(MODEL_PATH))
    reset_state()
    print(f"[도킹서버] best.pt 키포인트 전용 로드. {'DRY_RUN' if dry else 'DRIVE'}  포트 {PORT}")
    print(f"  ALIGN_Z={ALIGN_Z}m TOL_XPX={TOL_XPX} TOL_YAW={TOL_YAW}rad")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{PORT}")
    threading.Thread(
        target=lambda: ThreadingHTTPServer(("0.0.0.0", VIEW_PORT), Handler).serve_forever(),
        daemon=True
    ).start()
    print(f"[뷰어] http://GPU서버IP:{VIEW_PORT}")
    while True:
        try:
            parts = sock.recv_multipart()
            if len(parts) != 3:
                sock.send_string(json.dumps({
                    "state": "BAD_PACKET",
                    "lin": 0.0,
                    "ang": 0.0,
                    "dist_m": None
                }))
                continue
            meta = json.loads(parts[0].decode())
            img = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                sock.send_string(json.dumps({
                    "state": "BAD_RGB",
                    "lin": 0.0,
                    "ang": 0.0,
                    "dist_m": None
                }))
                continue
            front_d = None
            if meta.get("cloud", {}).get("present") and len(parts[2]) > 0:
                try:
                    front_d = front_min_dist(cloud_to_xyz(meta, parts[2]))
                except Exception:
                    front_d = None
            pose = estimate(img, model)
            lin, ang, state = compute(pose, front_d, dry)
            vis = draw(img, pose, lin, ang, state, front_d)
            ok, enc = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                latest_jpeg = enc.tobytes()
            sock.send_string(json.dumps({
                "state": state,
                "lin": lin,
                "ang": ang,
                "dist_m": round(pose["z"], 3) if pose and pose.get("z") is not None else None,
                "errx": round(pose["errx"], 4) if pose and pose.get("errx") is not None else None,
                "yaw": round(pose["yaw"], 4) if pose and pose.get("yaw") is not None else None
            }))
            print(
                f"[{state:14}] lin={lin:+.3f} ang={ang:+.3f} "
                f"z={pose['z'] if pose else None} "
                f"errx={pose.get('errx') if pose else None} "
                f"yaw={pose.get('yaw') if pose else None}"
            )
        except Exception as e:
            print("[도킹서버] 예외:", e)
            try:
                sock.send_string(json.dumps({
                    "state": "ERR",
                    "lin": 0.0,
                    "ang": 0.0,
                    "dist_m": None
                }))
            except Exception:
                pass
if __name__ == "__main__":
    main()
