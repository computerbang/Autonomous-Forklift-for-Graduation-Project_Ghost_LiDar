# 🚜 Autonomous Forklift (Ghost LiDAR)

**TurtleBot3 기반 마커리스(marker-free) 파레트 자율 도킹 — 미니어처 자율 지게차 PoC**
2026 졸업프로젝트 · Team 고스트라이다(Ghost LiDAR)

---

## 📌 개요

AprilTag 같은 별도 마커 없이, **RGB 카메라 + YOLOv8-pose 키포인트**로 파레트를 검출하고
**solvePnP**로 거리·정면각(yaw)을 추정해 파레트에 자율 도킹하는 시스템입니다.
주행·환경 인식은 **2D 라이다(LDS-02) 기반 SLAM**으로 맵을 그리고 자기 위치를 추정합니다.

- **카메라(로컬)** → 파레트 검출 + 정밀 도킹
- **라이다(글로벌)** → 맵 작성 + 자기 위치 추정 + 장애물 회피

센서마다 잘하는 일을 나눠 맡는 표준 AMR 구조를 따릅니다.
<img width="807" height="632" alt="화면 캡처 2026-06-09 135142" src="https://github.com/user-attachments/assets/42ca2681-8b93-47d6-bf15-90cb2b8c9228" />


---

## 🧱 시스템 아키텍처

```
            ┌──────────────────────────┐
            │  Sipeed MaixSense A075V   │  (RGB + ToF)
            └────────────┬─────────────┘
                         │ USB (RNDIS)
            ┌────────────▼─────────────┐        ZMQ multipart (port 5556)
            │  TurtleBot3 / RaspberryPi │  ───────────────────────────────┐
            │  ROS 2 Humble             │   [meta JSON | RGB JPEG | Cloud] │
            │  client_step6.py          │ ◀───────────── cmd_vel 응답 ─────┤
            │  + OpenCR(모터) + LDS-02  │                                  │
            └────────────┬─────────────┘                                  │
                         │ ROS 2 (/scan)                       ┌──────────▼──────────┐
            ┌────────────▼─────────────┐                       │  GPU 서버 (.0.204)   │
            │  slam_toolbox / Cartographer │                   │  server_step9.py     │
            │  → /map, TF (map→odom→base)  │                   │  YOLOv8-pose + PnP   │
            └────────────┬─────────────┘                       │  GTX 1060 / no ROS   │
                         │ WebSocket :8765                      └─────────────────────┘
            ┌────────────▼─────────────┐
            │  Foxglove (PC/Mac)        │  ← /scan, /map 시각화
            └──────────────────────────┘
```

- **카메라 추론 경로**: 로봇 ↔ GPU 서버는 ROS 없이 **ZeroMQ**로 통신 (서버에 ROS 미설치)
- **라이다·SLAM 경로**: 라즈베리파이 안에서 **ROS 2**로 처리, Foxglove로 원격 시각화

---

## 🔩 하드웨어

| 구성 | 사양 |
|------|------|
| 로봇 플랫폼 | TurtleBot3 Burger + OpenCR |
| 메인 보드 | Raspberry Pi 4 (Ubuntu 22.04, ROS 2 Humble) |
| 카메라 | Sipeed MaixSense A075V (RGB + ToF) |
| 라이다 | LDS-02 (2D 360° LiDAR) |
| GPU 서버 | NVIDIA GTX 1060 6GB (Ubuntu 24.04, ROS 미설치) |

## 💻 소프트웨어 스택

- **ROS 2 Humble** — 로봇 제어 / 센서 / SLAM
- **YOLOv8-pose** (Ultralytics) — 파레트 12-keypoint 검출
- **OpenCV** — `solvePnP` 거리·자세 추정
- **ZeroMQ (pyzmq)** — 서버 ↔ 로봇 영상/제어 통신
- **slam_toolbox / Cartographer** — 2D SLAM
- **Foxglove** — 라이다·맵 원격 시각화

---

## 📂 레포 구조

```
01_자료조사/                  # 참고 자료
02_발표자료/                  # 발표 PPT (4월 계획서 / 5월 중간)
03_결과 데이터/               # 캘리브 확인, Isaac Sim 영상, 라이다 캡쳐, 검증 예측 이미지
04_DJI 카메라 왜곡 보정/       # camera_matrix.npy, dist_coeffs.npy
05_서버:클라이언트코드/
   ├─ server_step9.py         # GPU 서버: 키포인트 검출 + solvePnP + 도킹 상태머신
   ├─ client_step6.py         # TB3 클라이언트: /rgb·/cloud 전송 + cmd_vel 수신
   └─ 서버에서 사용해야되는 파일/
        ├─ best.pt                     # YOLOv8-pose 학습 모델 (12 keypoint)
        ├─ rgb_K.npy / rgb_D.npy       # RGB 카메라 내부 파라미터 / 왜곡계수
        └─ pallet_3d_model_mockup.npy  # 파레트 3D 모델 (PnP용 12점)
06_모형 데이터 셋/
   └─ dataset_selected/        # angle_0 / angle_15 / angle_30, 각 dist_20~60cm
```

---

## ⚙️ 동작 원리 — 도킹 파이프라인

`server_step9.py` (키포인트 기반, AprilTag 의존성 제거)

1. **검출** — `best.pt`로 파레트 12개 키포인트 추출
2. **거리(z)** — 키포인트 + `solvePnP` (PnP의 z가 ~3cm 멀게 읽혀 `Z_BIAS=0.03` 보정)
3. **좌우정렬(errx)** — 키포인트 중심을 영상 중심에 맞춤
4. **정면각(yaw)** — `solvePnP` rvec로 정면 틀어짐 계산
5. **상태머신**
   - 접근 중 **10cm(`ALIGN_Z`)** 도달 → `errx`와 `yaw`가 **모두** 허용오차 안이면 직진 도킹(open-loop)
   - 정면이 안 맞으면 **후진하며 yaw를 0으로 맞춘 뒤 재접근(REALIGN, 최대 4회)**

주요 파라미터: `ALIGN_Z=0.10`, `TOL_Z=0.015`, `TOL_XPX=0.04`, `TOL_YAW=0.07(≈4°)`,
`KP_LIN=0.40`, `KP_ANG=1.20`, `KP_YAW=1.50`, `MAX_LIN=0.10`, `MAX_ANG=0.50`

---

## 📊 데이터셋

- **대상**: 1/14 스케일 모형 파레트, A075V RGB 촬영
- **구성**: 각도 `0° / 15° / 30°`, 거리 `20cm ~ 60cm`
- **라벨**: 12-keypoint (Roboflow)
- **학습**: YOLOv8-pose, **mAP 0.995**

---

## 📈 결과

- `03_결과 데이터/`에 캘리브레이션 확인, 검증 예측 이미지(`val_batch0_pred.jpg`), Isaac Sim 영상, 라이다 캡쳐 포함
- 근거리(≤30cm) 0/15/30° 검출 정상, RGB 키포인트 + solvePnP 기반 거리·정면각 추정 동작 확인

---

## 🗺️ 향후 계획

- RGB 정밀 도킹 고도화 (포크홀 중심 정렬, 진입 깊이 제어)
- Nav2 통합 — 자율 주행 + 장애물 회피
- 파레트 맵 등록(semantic mapping): 카메라 검출 → TF → 맵 좌표 마킹
- 추가 데이터셋 / 실제 파레트 스케일 검증

---

## 👥 팀

**고스트라이다 (Ghost LiDAR)** · 호서대학교 졸업프로젝트
