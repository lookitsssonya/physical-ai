import os
import json
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    import mujoco
    HAVE_MUJOCO = True
except ImportError:
    mujoco = None
    HAVE_MUJOCO = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[INFO] Скрипт лежит в: {SCRIPT_DIR}")
print(f"[INFO] MuJoCo доступен: {HAVE_MUJOCO}")

MODEL_XML = os.path.join(SCRIPT_DIR, "quadrotor2d.xml")

# Физические параметры
M = 0.5                    # масса рамы, кг
L = 0.15                   # плечо ротора, м
I = 0.005                  # момент инерции относительно оси Y, кг*м^2
G = 9.81                   # ускорение свободного падения, м/с^2
DT = 0.002                 # шаг физики, с (500 Гц)
F_MAX = 8.0                # ctrlrange ротора, Н
HOVER_F = M * G / 2.0      # тяга висения на один ротор
GUST_AMP = 0.02            # неучтённое возмущение (порыв), доля от веса

# Коэффициенты PD
KP_Z, KD_Z = 40.0, 12.0       # контур высоты
KP_X, KD_X = 2.0, 2.4         # контур горизонтали (внешний)
KP_TH, KD_TH = 4900.0, 140.0  # контур тангажа (внутренний)
TH_MAX = 0.35                 # ограничение задания по тангажу, рад

# Объект управления
class Quad2D:
    """Плоский квадрокоптер: состояние (x, z, theta, vx, vz, omega)."""

    def __init__(self, x=0.0, z=1.0, theta=0.0):
        self.s = np.array([x, z, theta, 0.0, 0.0, 0.0], dtype=float)

    @property
    def qpos(self):
        return self.s[:3]

    @property
    def qvel(self):
        return self.s[3:]

    def step(self, f_left, f_right, wind=0.0, gust=0.0, dt=DT):
        f_left = float(np.clip(f_left, 0.0, F_MAX))
        f_right = float(np.clip(f_right, 0.0, F_MAX))
        x, z, th, vx, vz, om = self.s
        T = f_left + f_right
        ax = T * np.sin(th) / M + wind / M
        az = T * np.cos(th) / M - G + gust / M
        al = L * (f_left - f_right) / I

        vx += ax * dt
        vz += az * dt
        om += al * dt
        x += vx * dt
        z += vz * dt
        th += om * dt
        self.s = np.array([x, z, th, vx, vz, om])
        return self.s

# Perception
def noisy(value, sigma, rng):
    return value + rng.normal(0.0, sigma)


class DelayedSensor:
    """Буфер задержки: возвращает измерение delay_steps шагов назад."""

    def __init__(self, delay_steps, init_value=0.0, dim=1):
        n = max(1, int(delay_steps))
        self.buffer = deque([np.full(dim, init_value, dtype=float)] * n,
                            maxlen=n)

    def read(self, value):
        self.buffer.append(np.asarray(value, dtype=float).copy())
        return self.buffer[0].copy()


class FailureDetector:
    """Детектор отказа сенсора высоты по скользящей дисперсии."""

    def __init__(self, window=25, var_hi=0.02, var_lo=1e-6):
        self.window = window
        self.var_hi = var_hi
        self.var_lo = var_lo
        self.hist = deque(maxlen=window)
        self.failed = False
        self.mode = 'ok'
        self.healthy_streak = 0

    def update(self, measurement):
        self.hist.append(float(measurement))
        if len(self.hist) == self.window:
            var = float(np.var(self.hist))
            if var > self.var_hi:
                now = 'noisy'
            elif var < self.var_lo:
                now = 'stuck'
            else:
                now = 'ok'

            if now != 'ok':
                self.mode = now
                self.healthy_streak = 0
            elif self.mode != 'ok':
                self.healthy_streak += 1
                if self.healthy_streak >= self.window:
                    self.mode = 'ok'

        self.failed = self.mode != 'ok'
        return self.mode

# Planning
def planning(state_est, target, dt_ctrl):
    """Каскадный PD: высота -> тяга, горизонт -> тангаж -> момент."""
    x, z, th, vx, vz, om = state_est
    xd, zd = target
    a_z = KP_Z * (zd - z) + KD_Z * (0.0 - vz)
    T = M * (G + a_z) / max(np.cos(np.clip(th, -1.2, 1.2)), 0.5)
    a_x = KP_X * (xd - x) + KD_X * (0.0 - vx)
    th_d = float(np.clip(a_x / G, -TH_MAX, TH_MAX))
    tau = I * (KP_TH * (th_d - th) + KD_TH * (0.0 - om))
    f_l = T / 2.0 + tau / (2.0 * L)
    f_r = T / 2.0 - tau / (2.0 * L)
    return f_l, f_r, th_d

# Control
def crash_reason(s):
    x, z, th = s[0], s[1], s[2]
    if z <= 0.05:
        return "касание земли"
    if abs(th) > np.pi / 2:
        return "опрокидывание |θ| > 90°"
    if abs(x) > 5.0:
        return "уход по горизонтали |x| > 5 м"
    if z > 6.0:
        return "уход по высоте z > 6 м"
    return None


def is_crashed(s):
    return crash_reason(s) is not None


def run(freq_hz=500.0, delay_ms=0.0, sigma=0.0, sim_time=10.0,
        target=(0.0, 1.0), start=(0.6, 0.4, 0.0), wind=0.0,
        use_detector=False, seed=0, fail_start=None, fail_sigma=0.5,
        fail_mode='noise', filter_tau=None):
    """Один замкнутый эксперимент. Возвращает словарь логов."""
    rng = np.random.default_rng(seed)
    q = make_plant(*start)

    sim_dt = DT
    steps_per_ctrl = max(1, int(round((1.0 / freq_hz) / sim_dt)))
    dt_ctrl = steps_per_ctrl * sim_dt
    delay_steps = max(1, int(round((delay_ms / 1000.0) / sim_dt)))

    sensor = DelayedSensor(delay_steps, dim=6)
    sensor.buffer = deque([q.s.copy()] * delay_steps, maxlen=delay_steps)
    det = FailureDetector(window=25, var_hi=0.02, var_lo=3e-4)

    t_log, z_log, x_log, th_log, meas_log, mode_log = [], [], [], [], [], []
    est_log = []
    f_l = f_r = HOVER_F
    est_prev = q.s.copy()
    n = int(sim_time / sim_dt)
    crash_t = None
    crash_why = None
    detector_on_steps = 0
    stuck_value = None

    for step in range(n):
        t = step * sim_dt

        if step % steps_per_ctrl == 0:
            # Perception
            sig = sigma
            broken = (fail_start is not None) and (t >= fail_start)
            if broken and fail_mode == 'noise':
                sig = fail_sigma

            meas = q.s.copy()
            if sig > 0:
                meas[0] = noisy(meas[0], sig, rng)
                meas[1] = noisy(meas[1], sig, rng)
                meas[2] = noisy(meas[2], sig * 0.2, rng)
                meas[3] = noisy(meas[3], sig * 2, rng)
                meas[4] = noisy(meas[4], sig * 2, rng)
                meas[5] = noisy(meas[5], sig, rng)

            if broken and fail_mode == 'stuck':
                if stuck_value is None:
                    stuck_value = float(meas[1])
                meas[1] = stuck_value
                meas[4] = 0.0

            meas = sensor.read(meas)

            # Прогноз по модели
            T_prev = f_l + f_r
            M_mod = M
            pred = est_prev.copy()
            pred[3] += T_prev * np.sin(pred[2]) / M_mod * dt_ctrl
            pred[4] += (T_prev * np.cos(pred[2]) / M_mod - G) * dt_ctrl
            pred[5] += L * (f_l - f_r) / I * dt_ctrl
            pred[0] += pred[3] * dt_ctrl
            pred[1] += pred[4] * dt_ctrl
            pred[2] += pred[5] * dt_ctrl

            # Фузия
            est = meas.copy()
            if filter_tau is not None:
                K = 1.0 - np.exp(-dt_ctrl / filter_tau)
                mode = det.update(meas[1]) if use_detector else 'ok'
                if mode == 'stuck':
                    K = 0.0
                    detector_on_steps += 1
                elif mode == 'noisy':
                    K = K / 2.0
                    detector_on_steps += 1
                for i in (0, 1, 3, 4):
                    est[i] = pred[i] + K * (meas[i] - pred[i])
            est_prev = est.copy()

            # Planning
            f_l, f_r, _ = planning(est, target, dt_ctrl)
            f_l = float(np.clip(f_l, 0.0, F_MAX))
            f_r = float(np.clip(f_r, 0.0, F_MAX))

            mode_log.append(1.0 if (use_detector and det.failed) else 0.0)
            meas_log.append(meas[1])
            est_log.append(est[1])

        gust = GUST_AMP * M * G * np.sin(2 * np.pi * 0.25 * t)
        q.step(f_l, f_r, wind=wind, gust=gust)

        t_log.append(t)
        x_log.append(q.s[0])
        z_log.append(q.s[1])
        th_log.append(q.s[2])

        reason = crash_reason(q.s)
        if reason is not None:
            crash_t = t
            crash_why = reason
            break

    t_log = np.array(t_log)
    z_log = np.array(z_log)
    x_log = np.array(x_log)
    th_log = np.array(th_log)

    if crash_t is None and len(t_log) > 10:
        k = int(0.7 * len(t_log))
        err = np.hypot(x_log[k:] - target[0], z_log[k:] - target[1])
        rms = float(np.sqrt(np.mean(err ** 2)))
        max_err = float(np.max(err))
    else:
        rms = float('nan')
        max_err = float('nan')

    return dict(t=t_log, x=x_log, z=z_log, th=th_log,
                meas=np.array(meas_log), mode=np.array(mode_log),
                est=np.array(est_log),
                crash_t=crash_t, crash_why=crash_why,
                rms=rms, max_err=max_err,
                freq=freq_hz, delay_ms=delay_ms, sigma=sigma,
                detector=detector_on_steps)


class QuadMuJoCo:
    def __init__(self, x=0.0, z=1.0, theta=0.0, xml_path=MODEL_XML):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.reset(x, z, theta)

    def reset(self, x, z, theta):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = [x, z - 1.0, theta]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def s(self):
        q, v = self.data.qpos, self.data.qvel
        return np.array([q[0], q[1] + 1.0, q[2], v[0], v[1], v[2]])

    def step(self, f_left, f_right, wind=0.0, gust=0.0, dt=DT):
        self.data.ctrl[0] = float(np.clip(f_left, 0.0, F_MAX))
        self.data.ctrl[1] = float(np.clip(f_right, 0.0, F_MAX))
        self.data.qfrc_applied[0] = wind
        self.data.qfrc_applied[1] = gust
        mujoco.mj_step(self.model, self.data)
        return self.s


def make_plant(x=0.0, z=1.0, theta=0.0):
    if HAVE_MUJOCO and os.path.exists(MODEL_XML):
        return QuadMuJoCo(x, z, theta)
    return Quad2D(x, z, theta)


def main():
    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.autolayout": True,
    })

    OUT = {}

    print("ЭКСПЕРИМЕНТ 1: baseline без контроллера")
    print("="*60)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))

    d = make_plant(0.0, 1.0, 0.0)
    t, z, th = [], [], []
    for i in range(2000):
        d.step(0.0, 0.0)
        t.append(i * DT); z.append(d.s[1]); th.append(d.s[2])
        if d.s[1] <= 0.05:
            break
    t_fall_free = t[-1]
    print(f"  Тяга = 0: падение за {t_fall_free:.3f} с")
    ax[0].plot(t, z, lw=1.8, color="#c0392b")
    ax[0].axhline(0.05, ls="--", lw=1, color="k")
    ax[0].set_title(f"Тяга = 0: свободное падение\n(касание земли при t = {t_fall_free:.2f} с)")
    ax[0].set_xlabel("Время, с"); ax[0].set_ylabel("Высота z, м")

    d = make_plant(0.0, 1.0, 0.02)
    t, z, th, x = [], [], [], []
    for i in range(4000):
        d.step(HOVER_F, HOVER_F)
        t.append(i * DT); z.append(d.s[1]); th.append(d.s[2]); x.append(d.s[0])
        if is_crashed(d.s):
            break
    t_fall_ol = t[-1]
    print(f"  Тяга = mg: потеря устойчивости за {t_fall_ol:.3f} с")
    ax[1].plot(t, th, lw=1.8, color="#2c3e50", label="Тангаж θ, рад")
    ax[1].plot(t, x, lw=1.5, color="#6e7e22", label="Смещение x, м")
    ax[1].plot(t, z, lw=1.5, color="#2980b9", label="Высота z, м")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"Постоянная тяга mg без обратной связи\n(потеря устойчивости при t = {t_fall_ol:.2f} с)")
    ax[1].set_xlabel("Время, с"); ax[1].set_ylabel("θ [рад], x, z [м]")

    fig.savefig(os.path.join(SCRIPT_DIR, "fig1_baseline.png"), dpi=130)
    plt.close(fig)
    OUT["t_fall_free"] = t_fall_free
    OUT["t_fall_openloop"] = t_fall_ol

    print("ЭКСПЕРИМЕНТ 2: Perception")
    print("="*60)
    rng = np.random.default_rng(3)
    r = run(freq_hz=500, sim_time=3.0, sigma=0.0)
    t = r["t"]; z_true = r["z"]
    z_noisy = z_true + rng.normal(0, 0.05, size=z_true.shape)
    delay_steps = int(0.030 / DT)
    z_delay = np.concatenate([np.full(delay_steps, z_true[0]), z_true])[:len(z_true)]
    z_both = z_delay + rng.normal(0, 0.05, size=z_true.shape)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, z_noisy, lw=0.7, color="#bdc3c7", label="Зашумлённый (σ = 0.05 м)")
    ax.plot(t, z_both, lw=0.7, color="#f39c12", alpha=0.8, label="Шум + задержка 30 мс")
    ax.plot(t, z_true, lw=2.0, color="#c0392b", label="Истинная высота")
    ax.plot(t, z_delay, lw=1.6, color="#2980b9", ls="--", label="Задержанный на 30 мс")
    ax.set_xlabel("Время, с"); ax.set_ylabel("Высота z, м")
    ax.set_title("Слой Perception: истина vs шум vs задержка")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 2.0)
    fig.savefig(os.path.join(SCRIPT_DIR, "fig2_perception.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.0))
    k_cam = int((1 / 30) / DT)
    ax.plot(t, z_true, lw=1.6, color="#c0392b", label="IMU 500 Гц (истина)")
    ax.step(t[:k_cam], z_true[:k_cam], where="post", lw=1.4, color="#16a085", label="Камера 30 fps (ZOH)")
    ax.plot(t[:k_cam], z_true[:k_cam], "o", ms=3, color="#16a085")
    ax.set_xlim(0.3, 1.2); ax.set_xlabel("Время, с"); ax.set_ylabel("Высота z, м")
    ax.set_title("Рассинхронизация потоков: камера 30 fps против IMU 500 Гц")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(SCRIPT_DIR, "fig3_sync.png"), dpi=130)
    plt.close(fig)
    print("  [OK] fig2_perception.png, fig3_sync.png")

    print("ЭКСПЕРИМЕНТ 3: Частота управления")
    print("="*60)
    FREQS = [1000, 500, 200, 150, 100, 70, 50, 40, 30, 25, 20]
    freq_rows = []
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(FREQS)))
    for c, f in zip(colors, FREQS):
        r = run(freq_hz=f, sim_time=10.0, sigma=0.01)
        freq_rows.append(dict(f=f, crash=r["crash_t"], rms=r["rms"], maxerr=r["max_err"]))
        if f in (1000, 500, 200, 100, 50, 30, 20):
            lab = (f"{f} Гц – падение {r['crash_t']:.2f} с" if r["crash_t"]
                   else f"{f} Гц – RMS  {r['rms']*1000:.0f} мм")
            ax.plot(r["t"], r["z"], lw=1.4, color=c, label=lab)
        cr = f"{r['crash_t']:.2f}с" if r["crash_t"] else "—"
        rms = f"{r['rms']*1000:.0f}мм" if not np.isnan(r["rms"]) else "—"
        print(f"  {f:>5} Гц: падение={cr:<8} RMS={rms}")
    ax.axhline(1.0, ls="--", lw=1, color="k", alpha=0.6)
    ax.set_ylim(0, 2.0); ax.set_xlabel("Время, с"); ax.set_ylabel("Высота z, м")
    ax.set_title("Влияние частоты контура управления на удержание точки (цель z = 1.0 м)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(SCRIPT_DIR, "fig4_freq.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ok = [(row["f"], row["rms"]) for row in freq_rows if row["crash"] is None]
    bad = [(row["f"], 1.0) for row in freq_rows if row["crash"] is not None]
    ax.semilogx([a for a, _ in ok], [b * 1000 for _, b in ok], "o-", color="#2980b9", label="RMS ошибки висения")
    if bad:
        ax.semilogx([a for a, _ in bad], [1000] * len(bad), "X", ms=10, color="#c0392b")
        ax.semilogx([], [], "X", color="#c0392b", label="Падение аппарата")
    ax.set_yscale("log")
    ax.set_xlabel("Частота управления, Гц"); ax.set_ylabel("RMS ошибки, мм")
    ax.set_title("Частота → точность/устойчивость")
    ax.axvspan(20, 30, color="#c0392b", alpha=0.12)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(SCRIPT_DIR, "fig5_freq_rms.png"), dpi=130)
    plt.close(fig)
    OUT["freq_rows"] = freq_rows

    print("ЭКСПЕРИМЕНТ 4: Задержка измерений")
    print("="*60)
    DELAYS = [0, 2, 5, 8, 10, 12, 15, 18, 20, 25, 30, 50, 100, 200]
    delay_rows = []
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    colors = plt.cm.plasma(np.linspace(0, 0.85, 7))
    show = [0, 5, 10, 15, 20, 30, 50]
    ci = 0
    for dms in DELAYS:
        r = run(freq_hz=500, delay_ms=dms, sim_time=10.0, sigma=0.01)
        delay_rows.append(dict(d=dms, crash=r["crash_t"], rms=r["rms"], maxerr=r["max_err"]))
        if dms in show:
            lab = (f"{dms} мс – падение {r['crash_t']:.2f} с" if r["crash_t"]
                   else f"{dms} мс – RMS  {r['rms']*1000:.0f} мм")
            ax.plot(r["t"], r["z"], lw=1.4, color=colors[ci], label=lab)
            ci += 1
        cr = f"{r['crash_t']:.2f}с" if r["crash_t"] else "—"
        rms = f"{r['rms']*1000:.0f}мм" if not np.isnan(r["rms"]) else "—"
        print(f"  {dms:>4} мс: падение={cr:<8} RMS={rms}")
    ax.axhline(1.0, ls="-", lw=1, color="k", alpha=0.6)
    ax.set_ylim(0, 2.2); ax.set_xlabel("Время, с"); ax.set_ylabel("Высота z, м")
    ax.set_title("Влияние задержки измерений на устойчивость (контур 500 Гц)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(SCRIPT_DIR, "fig6_delay.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ok = [(row["d"], row["rms"]) for row in delay_rows if row["crash"] is None]
    bad = [row["d"] for row in delay_rows if row["crash"] is not None]
    ax.plot([a for a, _ in ok], [b * 1000 for _, b in ok], "o-", color="#8e44ad", label="RMS ошибки висения")
    if bad:
        ax.plot(bad, [1000] * len(bad), "X", ms=10, color="#c0392b", label="Падение")
    ax.set_yscale("log"); ax.set_xscale("symlog", linthresh=10)
    ax.axvspan(20, 25, color="#c0392b", alpha=0.12)
    ax.set_xlabel("Задержка измерения, мс"); ax.set_ylabel("RMS ошибки, мм")
    ax.set_title("Задержка → точность/устойчивость")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(SCRIPT_DIR, "fig7_delay_rms.png"), dpi=130)
    plt.close(fig)
    OUT["delay_rows"] = delay_rows

    print("ЭКСПЕРИМЕНТ 5: Fail-Operational (залипание высотомера)")
    print("="*60)
    failop = {}
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for j, det in enumerate([False, True]):
        crashes, rmsss = [], []
        for s in range(5):
            r = run(freq_hz=500, delay_ms=4, sigma=0.05, sim_time=12.0,
                    use_detector=det, seed=s, fail_start=3.0,
                    fail_mode="stuck", filter_tau=0.15)
            crashes.append(r["crash_t"]); rmsss.append(r["rms"])
            ax[j].plot(r["t"], r["z"], lw=1.0, alpha=0.85)
        ax[j].axvline(3.0, ls="-", color="k", lw=1)
        ax[j].axhline(1.0, ls=":", color="k", lw=1, alpha=0.6)
        ax[j].set_xlabel("Время, с")
        ax[j].set_title("Без детектора (Fail-Safe)" if not det
                        else "С детектором (Fail-Operational)", fontsize=9)
        ax[j].set_ylim(0, 2.2)
        failop["with" if det else "without"] = dict(
            crashes=crashes,
            rms=[None if x != x else x for x in rmsss])
        crs = [f"{c:.2f}" if c else "—" for c in crashes]
        rms_avg = np.nanmean(rmsss) if any(x == x for x in rmsss) else float('nan')
        print(f"  {'С' if det else 'Без'} детектором: падения={crs}, RMS~{rms_avg:.3f} м")
    ax[0].set_ylabel("Высота z, м")
    ax[0].text(3.1, 2.0, "отказ высотомера", fontsize=7)
    fig.savefig(os.path.join(SCRIPT_DIR, "fig8_failop.png"), dpi=130)
    plt.close(fig)
    OUT["failop_stuck"] = failop

    r = run(freq_hz=500, delay_ms=4, sigma=0.05, sim_time=12.0,
            use_detector=True, seed=0, fail_start=3.0,
            fail_mode="stuck", filter_tau=0.15)
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(r["t"], r["z"], lw=1.6, color="#27ae60", label="Истинная высота")
    tc = np.linspace(0, r["t"][-1], len(r["meas"]))
    ax.plot(tc, r["meas"], lw=0.6, color="#bdc3c7", label="Показания высотомера")
    ax.plot(tc, r["est"], lw=1.0, color="#2980b9", label="Оценка состояния")
    if len(r["mode"]) > 0:
        m = r["mode"] > 0.5
        ax.fill_between(tc, 0, 2.2, where=m, color="#e74c3c", alpha=0.12,
                        label="Детектор: отказ")
    ax.set_ylim(0, 2.2); ax.set_xlabel("Время, с"); ax.set_ylabel("Высота z, м")
    ax.set_title("Fail-Operational: залипание высотомера на 3-й секунде")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(os.path.join(SCRIPT_DIR, "fig9_detector.png"), dpi=130)
    plt.close(fig)

    print(f"Папка с графиками: {SCRIPT_DIR}")
    print("Файлы:")
    for fname in sorted(os.listdir(SCRIPT_DIR)):
        if fname.startswith("fig") and fname.endswith(".png"):
            print(f"  {fname}  ({os.path.getsize(os.path.join(SCRIPT_DIR, fname))} байт)")


if __name__ == "__main__":
    main()
