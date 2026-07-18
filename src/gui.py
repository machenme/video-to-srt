#!/usr/bin/env python
"""
ASR Pipeline GUI — Tkinter-based graphical interface.
Drag & drop video files, click start, get subtitles.

Usage:
    uv run python -m src.gui
    uv run src/gui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as both `python -m src.gui` and `python src/gui.py`
if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import os
import queue
import subprocess
import threading
import time
from tkinter import filedialog, messagebox, ttk

# Use tkinterdnd2 for drag-and-drop support (must import before tkinter)
try:
    from tkinterdnd2 import TkinterDnD
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False
    from tkinter import Tk as _Tk

import tkinter as tk

from src.config import PipelineConfig, detect_optimal_workers
from src.main import run_one_video
from src.monitor import GpuMonitor
from src.translator import EdgeTranslator, translate_srt, TranslationError
from src.utils import scan_video_files


# ---------------------------------------------------------------------------
# Log-to-queue bridge
# ---------------------------------------------------------------------------

class QueueHandler(logging.Handler):
    """Pushes log records into a thread-safe queue for GUI consumption."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self._queue = log_queue
        self.setFormatter(logging.Formatter(
            "%(asctime)s  %(message)s", datefmt="%H:%M:%S"
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put(self.format(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------

class AsrGui:
    """Tkinter GUI wrapping the ASR pipeline."""

    def __init__(self):
        # --- root window ---
        if _DND_AVAILABLE:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("ASR Pipeline — 视频转字幕")
        self.root.geometry("900x700")
        self.root.minsize(700, 550)

        # --- state ---
        self._video_paths: list[Path] = []
        self._running = threading.Event()
        self._cancel = threading.Event()
        self._log_queue: queue.Queue = queue.Queue()
        self._progress_queue: queue.Queue = queue.Queue()
        self._gpu_monitor: GpuMonitor | None = None
        self._config: PipelineConfig | None = None

        # --- build UI ---
        self._build_config()
        self._build_output_row()
        self._build_video_list()
        self._build_advanced()
        self._build_gpu_status()
        self._build_progress()
        self._build_buttons()
        self._build_log()
        self._build_statusbar()

        # --- logging bridge ---
        self._setup_logging()

        # --- drag & drop ---
        if _DND_AVAILABLE:
            self.root.drop_target_register("DND_Files")
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self._log("⚠ tkinterdnd2 未安装，拖放功能不可用。使用 [添加视频] 按钮。")

        # --- periodic polling ---
        self._poll_logs()
        self._poll_gpu()

        self._log("ASR Pipeline GUI 启动完成。拖入视频或点击 [添加视频] 开始。")

    # ------------------------------------------------------------------
    # Build config
    # ------------------------------------------------------------------

    def _build_config(self) -> None:
        """Load config.yaml and build PipelineConfig."""
        try:
            config_path = Path("./config.yaml")
            raw = PipelineConfig.from_yaml(config_path)
            self._config = PipelineConfig.build({"config": "./config.yaml"})
        except Exception:
            self._config = PipelineConfig.build({"input_dir": ".", "output_dir": "./output"})
        # Detect optimal worker count from GPU VRAM
        self._optimal_workers = detect_optimal_workers()

    # ------------------------------------------------------------------
    # Output directory row
    # ------------------------------------------------------------------

    def _build_output_row(self) -> None:
        f = ttk.Frame(self.root)
        f.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(f, text="输出目录:").pack(side="left")
        self._out_dir_var = tk.StringVar(value=str(self._config.output_dir))
        ttk.Entry(f, textvariable=self._out_dir_var, width=50).pack(side="left", padx=5)
        ttk.Button(f, text="浏览...", command=self._browse_output).pack(side="left")

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="选择输出目录", initialdir=self._out_dir_var.get())
        if d:
            self._out_dir_var.set(d)

    # ------------------------------------------------------------------
    # Video list (drop zone)
    # ------------------------------------------------------------------

    def _build_video_list(self) -> None:
        frame = ttk.LabelFrame(self.root, text="视频列表（拖拽文件到此处或点击下方按钮添加）", padding=5)
        frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # Treeview
        cols = ("file", "duration", "status")
        self._tree = ttk.Treeview(frame, columns=cols, show="headings", height=3, selectmode="extended")
        self._tree.heading("file", text="文件名")
        self._tree.heading("duration", text="时长")
        self._tree.heading("status", text="状态")
        self._tree.column("file", width=400)
        self._tree.column("duration", width=100, anchor="center")
        self._tree.column("status", width=120, anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._tree.bind("<Double-1>", self._on_double_click_video)
        self._tree.bind("<Delete>", lambda e: self._remove_selected())

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(btn_frame, text="➕ 添加视频", command=self._add_videos).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="✖ 移除选中", command=self._remove_selected).pack(side="left", padx=2)

        btn_row2 = ttk.Frame(frame)
        btn_row2.pack(fill="x", pady=(2, 0))
        ttk.Button(btn_row2, text="🗑 清空列表", command=self._clear_videos).pack(side="left", padx=2)

    def _add_videos(self) -> None:
        files = filedialog.askopenfilenames(
            title="选择视频文件",
            filetypes=[
                ("视频/字幕", "*.mp4 *.mkv *.mov *.avi *.flv *.wmv *.srt"),
                ("所有文件", "*.*"),
            ],
        )
        for f in files:
            self._add_video(Path(f))

    def _add_video(self, p: Path) -> None:
        if p in self._video_paths:
            return
        # First video added → default output dir to video's folder
        if not self._video_paths:
            self._out_dir_var.set(str(p.parent))
        self._video_paths.append(p)
        dur_str = self._get_duration_str(p)
        self._tree.insert("", "end", iid=str(p), values=(p.name, dur_str, "等待中"))

    def _remove_selected(self) -> None:
        for sel in self._tree.selection():
            self._tree.delete(sel)
            p = Path(sel)
            if p in self._video_paths:
                self._video_paths.remove(p)

    def _clear_videos(self) -> None:
        self._tree.delete(*self._tree.get_children())
        self._video_paths.clear()

    def _on_drop(self, event) -> None:
        """Handle drag-and-drop of files."""
        # event.data contains file paths, one per line on Windows
        raw = event.data
        # Strip braces from Windows DnD paths
        paths_str = raw.strip()
        for line in paths_str.splitlines():
            for part in line.split():
                p = part.strip("{}")
                path = Path(p)
                if path.suffix.lower().lstrip(".") in {"mp4", "mkv", "mov", "avi", "flv", "wmv", "srt"}:
                    self._add_video(path)

    def _on_double_click_video(self, event) -> None:
        sel = self._tree.selection()
        if sel:
            path = Path(sel[0])
            if path.exists():
                os.startfile(str(path))

    def _get_duration_str(self, p: Path) -> str:
        try:
            import subprocess as sp
            r = sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(p)],
                capture_output=True, text=True, timeout=15,
            )
            secs = float(r.stdout.strip())
            h, m = divmod(int(secs), 3600)
            m, s = divmod(m, 60)
            if h > 0:
                return f"{h}h{m:02d}m"
            return f"{m}m{s:02d}s"
        except Exception:
            return "?"

    # ------------------------------------------------------------------
    # Advanced settings
    # ------------------------------------------------------------------

    def _build_advanced(self) -> None:
        self._adv_frame = ttk.LabelFrame(self.root, text="高级设置", padding=5)
        self._adv_frame.pack(fill="x", padx=10, pady=(2, 0))

        row1 = ttk.Frame(self._adv_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="模型:").pack(side="left")
        self._model_var = tk.StringVar(value=self._config.model_size)
        ttk.Combobox(row1, textvariable=self._model_var,
                     values=["large-v3-turbo", "large-v3", "medium"],
                     width=18, state="readonly").pack(side="left", padx=5)

        ttk.Label(row1, text="并发:").pack(side="left", padx=(20, 0))
        self._workers_var = tk.IntVar(value=self._optimal_workers)
        ttk.Spinbox(row1, textvariable=self._workers_var, from_=1, to=8, width=4).pack(side="left", padx=5)

        row2 = ttk.Frame(self._adv_frame)
        row2.pack(fill="x", pady=2)
        # Language display name → ISO code mapping
        self._lang_map = {
            "自动检测": "auto",
            "日语 (ja)": "ja",
            "中文 (zh)": "zh",
            "英语 (en)": "en",
            "韩语 (ko)": "ko",
            "法语 (fr)": "fr",
            "德语 (de)": "de",
            "西班牙语 (es)": "es",
            "葡萄牙语 (pt)": "pt",
            "意大利语 (it)": "it",
            "俄语 (ru)": "ru",
            "阿拉伯语 (ar)": "ar",
            "泰语 (th)": "th",
            "越南语 (vi)": "vi",
        }
        ttk.Label(row2, text="语言:").pack(side="left")
        self._lang_var = tk.StringVar(value="自动检测")
        ttk.Combobox(row2, textvariable=self._lang_var,
                     values=list(self._lang_map.keys()),
                     width=12, state="readonly").pack(side="left", padx=5)

        ttk.Label(row2, text="Beam:").pack(side="left", padx=(20, 0))
        self._beam_var = tk.IntVar(value=self._config.beam_size)
        ttk.Spinbox(row2, textvariable=self._beam_var, from_=1, to=10, width=4).pack(side="left", padx=5)

        ttk.Label(row2, text="切割(秒):").pack(side="left", padx=(20, 0))
        self._chunk_var = tk.StringVar(value="自动")
        self._chunk_combo = ttk.Combobox(row2, textvariable=self._chunk_var,
                     values=["自动", "300", "600", "900", "1200", "1800", "3600"],
                     width=6)
        self._chunk_combo.pack(side="left", padx=5)

        row3 = ttk.Frame(self._adv_frame)
        row3.pack(fill="x", pady=2)
        self._vad_var = tk.BooleanVar(value=self._config.vad_filter)
        ttk.Checkbutton(row3, text="VAD 语音检测", variable=self._vad_var).pack(side="left")
        self._srt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="SRT", variable=self._srt_var).pack(side="left", padx=(10, 0))
        self._txt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text="TXT", variable=self._txt_var).pack(side="left", padx=(10, 0))
        self._md_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text="MD", variable=self._md_var).pack(side="left", padx=(10, 0))

        row4 = ttk.Frame(self._adv_frame)
        row4.pack(fill="x", pady=2)
        # Translation target language
        self._translate_label_map = {
            "不翻译": "",
            "中文 (zh)": "zh",
            "英语 (en)": "en",
            "韩语 (ko)": "ko",
            "日语 (ja)": "ja",
            "法语 (fr)": "fr",
            "德语 (de)": "de",
            "俄语 (ru)": "ru",
            "西班牙语 (es)": "es",
            "葡萄牙语 (pt)": "pt",
        }
        ttk.Label(row4, text="翻译为:").pack(side="left")
        self._translate_var = tk.StringVar(value="不翻译")
        ttk.Combobox(row4, textvariable=self._translate_var,
                     values=list(self._translate_label_map.keys()),
                     width=11, state="readonly").pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # GPU status
    # ------------------------------------------------------------------

    def _build_gpu_status(self) -> None:
        self._gpu_label = ttk.Label(self.root, text="GPU: 检测中...", anchor="w")
        self._gpu_label.pack(fill="x", padx=10, pady=(5, 0))
        self._gpu_monitor = GpuMonitor(gpu_index=0, interval=1.0)
        self._gpu_monitor.start()

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------

    def _build_progress(self) -> None:
        f = ttk.Frame(self.root)
        f.pack(fill="x", padx=10, pady=(2, 0))
        self._progress = ttk.Progressbar(f, mode="determinate", length=400, maximum=100)
        self._progress.pack(side="left", fill="x", expand=True)
        self._file_counter_label = ttk.Label(f, text="", width=8, anchor="e")
        self._file_counter_label.pack(side="right", padx=(0, 5))
        self._progress_label = ttk.Label(f, text="就绪", width=12, anchor="e")
        self._progress_label.pack(side="right", padx=(0, 0))

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _build_buttons(self) -> None:
        f = ttk.Frame(self.root)
        f.pack(fill="x", padx=10, pady=(5, 0))
        self._start_btn = ttk.Button(f, text="▶ 开始转写", command=self._start)
        self._start_btn.pack(side="left", padx=2)
        self._stop_btn = ttk.Button(f, text="■ 停止", command=self._stop, state="disabled")
        self._stop_btn.pack(side="left", padx=2)

    # ------------------------------------------------------------------
    # Log output
    # ------------------------------------------------------------------

    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self.root, text="实时日志", padding=5)
        frame.pack(fill="both", expand=True, padx=10, pady=(5, 0))

        self._log_text = tk.Text(frame, height=6, wrap="word", state="disabled",
                                  font=("Consolas", 9))
        sb = ttk.Scrollbar(frame, orient="vertical", command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Color tags
        self._log_text.tag_config("error", foreground="red")
        self._log_text.tag_config("success", foreground="green")
        self._log_text.tag_config("warn", foreground="orange")

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_statusbar(self) -> None:
        self._status_var = tk.StringVar(value="就绪")
        self._statusbar = ttk.Label(self.root, textvariable=self._status_var,
                                     relief="sunken", anchor="w", padding=(5, 2))
        self._statusbar.pack(fill="x", padx=10, pady=(2, 10))

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        handler = QueueHandler(self._log_queue)
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Periodic polling
    # ------------------------------------------------------------------

    def _poll_logs(self) -> None:
        while not self._log_queue.empty():
            try:
                msg = self._log_queue.get_nowait()
                self._append_log(msg)
            except queue.Empty:
                break
        self.root.after(100, self._poll_logs)

    def _poll_gpu(self) -> None:
        if self._gpu_monitor and self._gpu_monitor.available:
            snap = self._gpu_monitor.latest()
            if snap:
                self._gpu_label.configure(
                    text=f"GPU: {snap.utilization_pct}% | "
                         f"{snap.memory_used_mb / 1024:.1f}/{snap.memory_total_mb / 1024:.1f} GB | "
                         f"{snap.temperature_c}°C"
                )
        self.root.after(1000, self._poll_gpu)

    def _poll_progress(self) -> None:
        """Poll progress queue during pipeline run."""
        while not self._progress_queue.empty():
            try:
                pct, file_idx, total_files = self._progress_queue.get_nowait()
                pct = max(0.0, min(100.0, pct))
                self._progress["value"] = pct
                self._progress_label.configure(text=f"{pct:.2f}%")
                if total_files > 1:
                    self._file_counter_label.configure(text=f"({file_idx + 1}/{total_files})")
            except queue.Empty:
                break
        self.root.after(100, self._poll_progress)

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Direct log insertion (for GUI-level messages, not from pipeline)."""
        self._append_log(msg)

    def _append_log(self, msg: str) -> None:
        self._log_text.configure(state="normal")
        tag = None
        if "FAILED" in msg or "ERROR" in msg or "error" in msg.lower():
            tag = "error"
        elif "✓" in msg or "完成" in msg or "success" in msg.lower():
            tag = "success"
        elif "⚠" in msg or "warn" in msg.lower():
            tag = "warn"
        self._log_text.insert("end", msg + "\n", tag or ())
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------

    def _start(self) -> None:
        if not self._video_paths:
            messagebox.showwarning("无视频", "请先添加视频文件（拖入或点击 [添加视频]）。")
            return

        # Collect output formats
        formats = []
        if self._srt_var.get():
            formats.append("srt")
        if self._txt_var.get():
            formats.append("txt")
        if self._md_var.get():
            formats.append("md")
        if not formats:
            messagebox.showwarning("无输出格式", "请至少选择一种输出格式（SRT/TXT/MD）。")
            return

        # SRT files require a translation target
        has_srt = any(p.suffix.lower() == ".srt" for p in self._video_paths)
        translate_to = self._translate_label_map[self._translate_var.get()]
        if has_srt and not translate_to:
            messagebox.showwarning("需要翻译语言", "列表中有 SRT 文件，请选择翻译目标语言。")
            return

        # Update config with GUI values
        try:
            config = PipelineConfig.build({
                "config": "./config.yaml",
                "input_dir": str(Path(self._video_paths[0]).parent),
                "output_dir": self._out_dir_var.get(),
                "model": self._model_var.get(),
                "workers": self._workers_var.get(),
                "language": self._lang_map[self._lang_var.get()],
                "beam_size": self._beam_var.get(),
                "vad_filter": self._vad_var.get(),
                "chunk_duration": 0 if self._chunk_var.get() == "自动" else int(self._chunk_var.get()),
                "output_formats": formats,
                "translate_to": self._translate_label_map[self._translate_var.get()],
            })
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return

        # Disable UI
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._cancel.clear()
        self._running.set()

        # Reset progress
        self._progress["value"] = 0
        self._progress_label.configure(text="0.00%")
        total = len(self._video_paths)
        self._file_counter_label.configure(text=f"(0/{total})" if total > 1 else "")
        self._status_var.set("转写中...")

        # Start pipeline in background thread
        t = threading.Thread(target=self._run_all, args=(config,), daemon=True)
        t.start()

        # Start progress polling
        self._poll_progress()

    def _stop(self) -> None:
        self._cancel.set()
        self._log("⚠ 用户请求停止，等待当前任务完成...")
        self._status_var.set("正在停止...")
        self._stop_btn.configure(state="disabled")

    def _run_all(self, config: PipelineConfig) -> None:
        """Process all videos and SRT files in the list (runs in background thread)."""
        total = len(self._video_paths)
        done_count = 0
        start_time = time.time()

        for i, file_path in enumerate(self._video_paths):
            if self._cancel.is_set():
                break

            # Reset progress for this file
            self._progress_queue.put((0.0, i, total))

            is_srt = file_path.suffix.lower() == ".srt"

            if is_srt:
                # --- SRT-only: skip ASR, translate directly ---
                self.root.after(0, lambda p=file_path: self._set_video_status(p, "翻译中..."))
                try:
                    provider = EdgeTranslator()
                    translate_srt(
                        file_path, config.translate_to,
                        provider=provider,
                        source_lang="auto",
                    )
                    self.root.after(0, lambda p=file_path: self._set_video_status(p, "✅ 已翻译"))
                    done_count += 1
                    self._progress_queue.put((100.0, i, total))
                except Exception as exc:
                    self.root.after(0, lambda p=file_path, e=exc: self._set_video_status(p, f"❌ {str(e)[:30]}"))
                    self._log(f"⚠ SRT 翻译失败: {file_path.name} — {exc}")
            else:
                # --- Video: full ASR pipeline ---
                self.root.after(0, lambda p=file_path: self._set_video_status(p, "处理中..."))

                ok, seg_count, err = run_one_video(
                    config, file_path,
                    progress_callback=lambda stage, cur, tot, idx=i, t=total: (
                        self._progress_queue.put(((cur / max(tot, 1)) * 100, idx, t))
                    ),
                    cancel_event=self._cancel,
                )

                if ok:
                    done_count += 1
                    self.root.after(0, lambda p=file_path, c=seg_count: self._set_video_status(p, f"✅ {c}段"))
                else:
                    self.root.after(0, lambda p=file_path, e=err: self._set_video_status(p, f"❌ {e[:30]}"))

        elapsed = time.time() - start_time

        # Re-enable UI
        self.root.after(0, self._pipeline_done, done_count, total, elapsed)

    def _set_video_status(self, video_path: Path, status: str) -> None:
        try:
            self._tree.set(str(video_path), "status", status)
        except Exception:
            pass

    def _pipeline_done(self, done: int, total: int, elapsed: float) -> None:
        self._running.clear()
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._progress["value"] = 100
        self._progress_label.configure(text="100.00%")
        self._file_counter_label.configure(text="")
        self._status_var.set(f"✅ 完成: {done}/{total}  (耗时 {elapsed:.0f}s)")

        # Open output folder
        out_dir = self._out_dir_var.get()
        if Path(out_dir).exists():
            self._log(f"输出目录: {out_dir}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = AsrGui()
    app.run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
