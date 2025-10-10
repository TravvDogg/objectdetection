#!/usr/bin/env python3
"""
macOS Apple-Silicon terminal UI + Audio Sonifier + GPU visualisation.

Displays:
- Per-core utilisation (logical cores)
- GPU utilisation (total and per-engine if available)
- Thermal pressure + P/E cluster frequencies via powermetrics (best-effort)
- Real-time tones (per-core oscillators)

Dependencies:
  brew install portaudio
  python3 -m pip install --upgrade textual psutil numpy sounddevice

Run with:
  sudo -v && python3 tui_sonifier.py

Keys:
  q quit
  m mute/unmute
  +/- volume
  f toggle linear/log frequency mapping
  r cycle audible cores
"""
import asyncio
import math
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import psutil
import sounddevice as sd
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Static

UPDATE_HZ = 10
AUDIO_SR = 48000
CHUNK = 256
BASE_FREQ = 80.0
FREQ_SPAN = 2260.0
MAX_AUDIBLE_CORES = 12

RE_THERMAL = re.compile(r"Thermal Pressure:\s*(\w+)", re.IGNORECASE)
RE_E_FREQ = re.compile(r"E-Cluster\s+frequency:\s*(\d+(?:\.\d+)?)\s*MHz", re.IGNORECASE)
RE_P_FREQ = re.compile(r"P-Cluster\s+frequency:\s*(\d+(?:\.\d+)?)\s*MHz", re.IGNORECASE)

@dataclass
class PowermetricsSample:
    thermal: Optional[str] = None
    e_mhz: Optional[float] = None
    p_mhz: Optional[float] = None

def read_powermetrics_once(timeout: float = 1.5) -> PowermetricsSample:
    cmd = "powermetrics -i 1000 --samplers cpu_power,thermal --once"
    try:
        proc = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, _ = proc.communicate(timeout=timeout)
        thermal = e_mhz = p_mhz = None
        for line in out.splitlines():
            if thermal is None and (m := RE_THERMAL.search(line)):
                thermal = m.group(1).capitalize()
            if e_mhz is None and (m := RE_E_FREQ.search(line)):
                e_mhz = float(m.group(1))
            if p_mhz is None and (m := RE_P_FREQ.search(line)):
                p_mhz = float(m.group(1))
        return PowermetricsSample(thermal, e_mhz, p_mhz)
    except Exception:
        return PowermetricsSample()

# GPU usage using psutil sensors_temperatures may not expose GPU; fallback to powermetrics GPU sampler
def read_gpu_utilisation(timeout: float = 1.5) -> Optional[float]:
    cmd = "powermetrics -i 1000 --samplers gpu_power --once"
    try:
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, _ = proc.communicate(timeout=timeout)
        for line in out.splitlines():
            if "GPU Power" in line and "mW" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.lower() == "power:" and i + 1 < len(parts):
                        val = parts[i + 1]
                        try:
                            return float(val)
                        except ValueError:
                            pass
        return None
    except Exception:
        return None

class CoreSynth:
    def __init__(self):
        self.sr = AUDIO_SR
        self.master = 0.1
        self.enabled = True
        self.lock = threading.Lock()
        self.phase = {}
        self.core_params = {}
        self.distortion_drive = 1.0  # starts clean
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            blocksize=CHUNK,
            callback=self._callback,
            finished_callback=self._on_finished,
        )
        self.stream.start()

    def _on_finished(self):
        pass

    def toggle_enable(self):
        with self.lock:
            self.enabled = not self.enabled

    def set_master(self, val: float):
        with self.lock:
            self.master = max(0.0, min(val, 1.0))

    def set_distortion(self, drive: float):
        with self.lock:
            self.distortion_drive = max(1.0, min(drive, 8.0))

    def update_from_util(self, utils):
        with self.lock:
            for i in range(len(utils)):
                u = utils[i]
                frac = max(0.0, min(1.0, u / 100.0))
                # Logarithmic frequency mapping for drone-like sound
                freq_target = BASE_FREQ * (2 ** (frac * 3))  # Up to 8x base freq
                amp_target = 0.1 + 0.4 * frac
                prev_freq, prev_amp = self.core_params.get(i, (freq_target, amp_target))
                freq = 0.85 * prev_freq + 0.15 * freq_target
                amp = 0.85 * prev_amp + 0.15 * amp_target
                self.core_params[i] = (freq, amp)

    def _callback(self, outdata, frames, time_info, status):
        buf = np.zeros(frames, dtype=np.float32)
        with self.lock:
            if not self.enabled:
                outdata[:] = buf.reshape(-1, 1)
                return
            for idx, (freq, amp) in self.core_params.items():
                phase = self.phase.get(idx, 0.0)
                n = np.arange(frames)
                sig = np.sin(2 * np.pi * freq * (n / self.sr) + phase)
                buf += amp * sig
                phase = (phase + 2 * np.pi * freq * (frames / self.sr)) % (2 * np.pi)
                self.phase[idx] = phase

            # apply distortion
            buf *= self.master
            buf = np.tanh(buf * self.distortion_drive).astype(np.float32)
        outdata[:] = buf.reshape(-1, 1)

    def close(self):
        self.stream.stop()
        self.stream.close()

class MetricBar(Static):
    value = reactive(0.0)
    label = reactive("")
    def render(self):
        width = self.size.width or 40
        filled = int(width * self.value)
        bar = "█" * filled + "·" * (width - filled)
        return f"{self.label:>10} |{bar}| {self.value*100:5.1f}%"

class TextLabel(Static):
    text = reactive("")
    def render(self):
        return self.text

class CoreSonifierApp(App):
    CSS = "Screen { align: center middle; } #root { width: 92%; height: 92%; } .panel { border: solid #444; padding: 1; }"

    def __init__(self):
        super().__init__()
        self.cpu_count = psutil.cpu_count(logical=True) or 1
        self.bars: List[MetricBar] = []
        self.gpu_bar = MetricBar()
        self.freq_label = TextLabel()
        self.therm_label = TextLabel()
        self.status = TextLabel()
        self.synth = CoreSynth()

    def compose(self) -> ComposeResult:
        with Container(id="root"):
            yield Static("CPU/GPU Visualiser + Audio (q quit, m mute, +/- vol, f map, r cores)")
            for i in range(self.cpu_count):
                bar = MetricBar()
                bar.label = f"Core {i:02d}"
                self.bars.append(bar)
                yield bar
            self.gpu_bar.label = "GPU"
            yield self.gpu_bar
            yield self.freq_label
            yield self.therm_label
            yield self.status

    async def on_mount(self):
        self.set_interval(1.0 / UPDATE_HZ, self._tick)

    async def _tick(self):
        utils = psutil.cpu_percent(interval=None, percpu=True)
        for i, u in enumerate(utils):
            self.bars[i].value = u / 100
        self.synth.update_from_util(utils)
        gpu_util = await asyncio.to_thread(read_gpu_utilisation)
        if gpu_util is not None:
            # Map ~0-1000 mW to 0-100%
            frac = min(1.0, gpu_util / 1000.0)
            self.gpu_bar.value = frac
        sample = await asyncio.to_thread(read_powermetrics_once)
        therm = sample.thermal or "Unknown"
        # 🔥 Map thermal pressure to distortion
        drive_map = {
            "Nominal": 1.0,
            "Light": 1.5,
            "Moderate": 2.5,
            "Heavy": 4.0,
            "Trapping": 6.0
        }
        drive = drive_map.get(sample.thermal, 1.0)
        self.synth.set_distortion(drive)
        self.freq_label.text = f"Clusters: E {sample.e_mhz or 0:.0f} MHz | P {sample.p_mhz or 0:.0f} MHz"
        self.therm_label.text = f"Thermal Pressure: {therm}"
        self.status.text = f"Audio {'ON' if self.synth.enabled else 'MUTED'} | Vol {self.synth.master:.2f}"

    async def on_key(self, event: events.Key):
        if event.key == 'q':
            await self.action_quit()
        elif event.key == 'm':
            self.synth.toggle_enable()
        elif event.key == '+':
            self.synth.set_master(self.synth.master + 0.05)
        elif event.key == '-':
            self.synth.set_master(self.synth.master - 0.05)

    async def on_unmount(self):
        self.synth.close()

if __name__ == '__main__':
    CoreSonifierApp().run()
