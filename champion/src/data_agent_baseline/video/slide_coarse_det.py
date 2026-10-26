"""Coarse slide-frame detection from H.264 bitrate rhythm (no full decode).

Idea (validated on the phase2 briefing videos): an H.264 P-frame encodes only
the *residual* against its reference, so a held-still slide costs ~50 bytes per
frame while a slide change -- new content with no usable reference -- explodes to
thousands of bytes for the ~0.5 s the transition/animation lasts. So the byte
size of each coded packet, read straight from the container with NO decoding,
traces a clean "static (small) <-> burst (big)" rhythm:

  * a run of CONSECUTIVE BIG packets  = a transition or an on-slide animation;
  * a run of small packets            = a settled, unchanging slide.

This is strictly more reliable than keying on I-frames: x264 inserts a forced
I-frame every keyint (=250 frames here) even when nothing changed -- a "fake"
keyframe surrounded by tiny packets. Because we trigger on byte BURSTS, those
fake keyframes fall inside a quiet stretch and produce no candidate, while a real
change that x264 coded as a plain P-frame (inside min-keyint) is still caught.

We deliberately favour RECALL over precision: every burst yields a candidate
frame, so a multi-step build/animation may emit several frames for one slide.
That is fine -- the candidates are meant to be fed to a multimodal LLM that does
the final same-slide dedup. ``min_quiet_s`` optionally folds bursts separated by
a short quiet gap (on-slide animation steps) into one slide; see its default.

Representative frame: for each (merged) burst we take the frame settled JUST
AFTER it -- "what the slide looks like once this change finished" -- and record
both that frame's real timestamp and the originating burst window.

Self-contained: ffprobe (bundled by the repo's ``static-ffmpeg`` dep) reads
packet sizes without decoding; PyAV decodes only the handful of frames actually
exported. This module does not depend on ``video.frames`` (which may be retired).

Library::

    from video.slide_coarse_det import detect_slides, extract_slides_coarse
    slides = extract_slides_coarse(video_path, out_dir)   # -> [SlideFrame]
    # or analysis only, no PNG I/O:
    cands, diag = detect_slides(video_path)

CLI::

    uv run python -m video.slide_coarse_det <video.mp4>
    uv run python -m video.slide_coarse_det --task-dir <task_dir> --out-dir frames/
"""
from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import av
from static_ffmpeg import run as _sff_run

__all__ = [
    "SlideFrame",
    "Burst",
    "detect_slides",
    "extract_slides_coarse",
    "SPIKE_K",
    "MERGE_GAP_S",
    "SETTLE_MARGIN_S",
    "MIN_QUIET_S",
]

# ---- tunables (all measured on the phase2 briefing set; biased toward recall) ----
SPIKE_K = 6.0          # a packet is "big" if size > SPIKE_K * median packet size.
#                        6x cleanly clears the periodic ~2x micro-flicker (cursor/clock)
#                        while keeping every real transition/animation packet.
MERGE_GAP_S = 0.4      # big packets <= this far apart belong to ONE burst (the burst
#                        is jagged: B-frames alternate big/small during a transition).
SETTLE_MARGIN_S = 0.2  # take the representative frame this long AFTER the burst ends,
#                        so we sample the settled slide, not a mid-animation blend.
MIN_QUIET_S = 1.5      # merge a burst into the previous slide if the QUIET gap before
#                        it is shorter than this. The phase2 quiet-gap histogram is
#                        bimodal -- on-slide animation steps cluster at 0.5-1.5 s, real
#                        slide changes at 4-8 s, with a clear valley at ~2.5-3 s. 1.5 s
#                        folds the animation cluster while keeping every real change
#                        (knee of the candidate-count curve). Set 0 for pure max-recall
#                        (one frame per burst); raise toward 2.5 for fewer, settled frames.

_, _FFPROBE = _sff_run.get_or_fetch_platform_executables_else_raise()


@dataclass
class Burst:
    """A run of consecutive big packets = one transition or on-slide animation."""

    t_start: float        # first big packet time (s) -- when the change began
    t_end: float          # last big packet time (s) -- when it settled
    peak_bytes: int       # largest packet in the burst (transition magnitude)
    n_big: int            # number of big packets in the burst
    n_kf_big: int = 0     # how many of those big packets are I-frames

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class SlideFrame:
    """One exported candidate slide frame (settled just after a burst)."""

    index: int            # candidate index (0-based, chronological)
    t: float              # REAL timestamp (s) of the exported frame
    path: str             # absolute path to the written PNG ("" if not exported)
    burst_start: float    # originating burst window start (s)
    burst_end: float      # originating burst window end (s)

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _read_packets(video_path: str) -> list[tuple[float, int, bool]]:
    """(pts_time, size_bytes, is_keyframe) for every video packet, time-sorted.

    Read straight from the container via ffprobe -- no decoding.
    """
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,size,flags", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout
    pk: list[tuple[float, int, bool]] = []
    for ln in out.splitlines():
        c = ln.split(",")
        if len(c) >= 3:
            try:
                pk.append((float(c[0]), int(c[1]), "K" in c[2]))
            except ValueError:
                pass
    return sorted(pk)


def _find_bursts(packets: list[tuple[float, int, bool]], spike_thr: float,
                 merge_gap_s: float) -> list[Burst]:
    """Collapse runs of big packets (> spike_thr, <= merge_gap_s apart) into bursts."""
    bursts: list[Burst] = []
    cur_start = cur_end = None
    cur_peak = 0
    cur_n = 0
    cur_kf = 0
    for t, s, is_kf in packets:
        if s <= spike_thr:
            continue
        if cur_start is not None and (t - cur_end) <= merge_gap_s:
            cur_end = t
            cur_peak = max(cur_peak, s)
            cur_n += 1
            cur_kf += is_kf
        else:
            if cur_start is not None:
                bursts.append(Burst(cur_start, cur_end, cur_peak, cur_n, cur_kf))
            cur_start = cur_end = t
            cur_peak = s
            cur_n = 1
            cur_kf = int(is_kf)
    if cur_start is not None:
        bursts.append(Burst(cur_start, cur_end, cur_peak, cur_n, cur_kf))
    return bursts


def _drop_forced_keyframes(bursts: list[Burst]) -> list[Burst]:
    """Drop bursts that are a single I-frame big packet on an otherwise-still slide.

    x264 inserts a forced I-frame every keyint (=250 frames / 8.333 s here) even
    when the picture has not changed. Such an I-frame is self-contained, so it is
    huge (~600x the median P-frame) and trips the spike threshold ALL BY ITSELF --
    a lone big packet (n_big == 1) that is a keyframe, with no neighbouring big
    packets. A real slide change instead produces a RUN of big packets (the
    transition/animation residual), so it never looks like this. Emitting a frame
    for a forced keyframe yields a pixel-identical duplicate of the previous slide,
    which is exactly the duplicate-frame bug; dropping them removes it at the
    source. A genuine hard cut that happens to land on the keyint boundary still
    carries P-frame residual around it (n_big > 1), so it is preserved.
    """
    return [b for b in bursts if not (b.n_big == 1 and b.n_kf_big == 1)]


def _merge_short_quiet(bursts: list[Burst], min_quiet_s: float) -> list[Burst]:
    """Fold a burst into the previous slide if the quiet gap before it < min_quiet_s.

    Merged bursts span from the first burst's start to the last's end; peak/n_big
    accumulate. This turns a slide's animation steps into a single slide event.
    """
    if min_quiet_s <= 0 or not bursts:
        return bursts
    merged: list[Burst] = [bursts[0]]
    for b in bursts[1:]:
        prev = merged[-1]
        if (b.t_start - prev.t_end) < min_quiet_s:
            merged[-1] = Burst(prev.t_start, b.t_end,
                               max(prev.peak_bytes, b.peak_bytes),
                               prev.n_big + b.n_big, prev.n_kf_big + b.n_kf_big)
        else:
            merged.append(b)
    return merged


def detect_slides(
    video_path: str | Path,
    *,
    spike_k: float = SPIKE_K,
    merge_gap_s: float = MERGE_GAP_S,
    settle_margin_s: float = SETTLE_MARGIN_S,
    min_quiet_s: float = MIN_QUIET_S,
    drop_forced_keyframes: bool = True,
) -> tuple[list[tuple[float, Burst]], dict]:
    """Detect candidate slide timestamps from packet-size bursts. Pure analysis.

    Returns:
        (candidates, diagnostics) where
          candidates: [(target_time_s, originating_burst)], chronological. Each
            target is ``min(burst.t_end + settle_margin_s, next_burst.t_start - eps)``
            -- the settled frame just AFTER the burst. The opening slide at t~0 is
            always included as the first candidate (its burst is synthesized).
          diagnostics: dict with median packet size, spike threshold, the raw and
            merged burst lists, and per-keyframe real/fake labels (a keyframe is
            "fake" if no burst overlaps it -- a forced keyint I-frame on a still slide).

    drop_forced_keyframes: drop lone-I-frame bursts (n_big == 1 and that packet is a
        keyframe) BEFORE merging. These are x264 forced keyint I-frames on a still
        slide; emitting them yields a pixel-identical duplicate of the previous
        slide. On by default; set False to keep every burst (pure max-recall).
    """
    packets = _read_packets(str(video_path))
    if not packets:
        return [], {"median_bytes": 0, "spike_thr": 0, "bursts": [], "keyframes": []}

    sizes = [s for _, s, _ in packets]
    median = statistics.median(sizes)
    spike_thr = median * spike_k
    duration = packets[-1][0]

    raw_bursts = _find_bursts(packets, spike_thr, merge_gap_s)
    kept = _drop_forced_keyframes(raw_bursts) if drop_forced_keyframes else raw_bursts
    bursts = _merge_short_quiet(kept, min_quiet_s)

    # representative time = settled frame just after each burst, clamped before the
    # next burst so we never sample into the following transition.
    eps = 1e-3
    starts = [b.t_start for b in bursts]
    candidates: list[tuple[float, Burst]] = []

    # opening slide: the very first frame, before any change.
    first_change = bursts[0].t_start if bursts else duration
    if first_change > settle_margin_s:
        candidates.append((0.0, Burst(0.0, 0.0, packets[0][1], 0)))

    for i, b in enumerate(bursts):
        t = b.t_end + settle_margin_s
        nxt = starts[i + 1] if i + 1 < len(starts) else duration
        t = min(t, max(b.t_end, nxt - eps))
        t = min(t, duration)
        candidates.append((round(t, 3), b))

    keyframes = [
        {"t": round(t, 3), "real": any(b.t_start - eps <= t <= b.t_end + eps for b in bursts)}
        for t, _, k in packets if k
    ]
    diagnostics = {
        "median_bytes": round(median, 1),
        "spike_thr": round(spike_thr, 1),
        "duration": round(duration, 3),
        "n_raw_bursts": len(raw_bursts),
        "n_dropped_forced_kf": len(raw_bursts) - len(kept),
        "n_merged_bursts": len(bursts),
        "bursts": [b.to_dict() for b in bursts],
        "keyframes": keyframes,
    }
    return candidates, diagnostics


def _encode_png(frame: "av.VideoFrame", path: str) -> None:
    """Write a decoded frame to a full-resolution, lossless PNG (no resize)."""
    oc = av.open(path, mode="w")
    st = oc.add_stream("png")
    st.width, st.height, st.pix_fmt = frame.width, frame.height, "rgb24"
    for pkt in st.encode(frame.reformat(format="rgb24")):
        oc.mux(pkt)
    for pkt in st.encode(None):
        oc.mux(pkt)
    oc.close()


def extract_slides_coarse(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    spike_k: float = SPIKE_K,
    merge_gap_s: float = MERGE_GAP_S,
    settle_margin_s: float = SETTLE_MARGIN_S,
    min_quiet_s: float = MIN_QUIET_S,
    drop_forced_keyframes: bool = True,
) -> list[SlideFrame]:
    """Detect candidate slides and write each as a full-res PNG.

    Two passes: (1) ``detect_slides`` finds target timestamps from packet bursts
    (no decode); (2) seek + decode the exact frame at each target and PNG it at
    native resolution.

    Returns:
        list[SlideFrame] in chronological order, each carrying its real exported
        timestamp and the originating burst window.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates, _ = detect_slides(
        video_path, spike_k=spike_k, merge_gap_s=merge_gap_s,
        settle_margin_s=settle_margin_s, min_quiet_s=min_quiet_s,
        drop_forced_keyframes=drop_forced_keyframes,
    )

    c = av.open(str(video_path))
    s = c.streams.video[0]
    tb = float(s.time_base)
    results: list[SlideFrame] = []
    for k, (tt, burst) in enumerate(candidates):
        c.seek(int(tt / tb), stream=s)
        got = None
        for frame in c.decode(s):
            if float(frame.pts * frame.time_base) >= tt:
                got = frame
                break
            got = frame
        if got is None:
            continue
        real_t = round(float(got.pts * got.time_base), 3)
        # filename carries: frame id (k), real frame timestamp (t), burst window (b)
        path = str(out / f"slide_{k:02d}_t{real_t:06.2f}"
                         f"_b{burst.t_start:06.2f}-{burst.t_end:06.2f}.png")
        _encode_png(got, path)
        results.append(SlideFrame(
            index=k, t=real_t, path=path,
            burst_start=round(burst.t_start, 3), burst_end=round(burst.t_end, 3),
        ))
    c.close()
    return results


def _resolve_video(args) -> Path:
    if args.task_dir:
        return Path(args.video) if args.video else Path(args.task_dir) / "context/video/briefing.mp4"
    return Path(args.video)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Coarse slide detection from H.264 bitrate bursts (no full decode).")
    ap.add_argument("video", nargs="?", default=None, help="video path (briefing.mp4)")
    ap.add_argument("--task-dir", default=None,
                    help="task dir; uses <task-dir>/context/video/briefing.mp4")
    ap.add_argument("--out-dir", default=None, help="write candidate PNGs here (else analysis only)")
    ap.add_argument("--spike-k", type=float, default=SPIKE_K)
    ap.add_argument("--merge-gap-s", type=float, default=MERGE_GAP_S)
    ap.add_argument("--settle-margin-s", type=float, default=SETTLE_MARGIN_S)
    ap.add_argument("--min-quiet-s", type=float, default=MIN_QUIET_S)
    ap.add_argument("--keep-forced-keyframes", action="store_true",
                    help="keep lone forced-keyint I-frames (else dropped to avoid duplicate slides)")
    ap.add_argument("--json", action="store_true", help="dump full diagnostics as JSON")
    args = ap.parse_args()
    if not args.video and not args.task_dir:
        ap.error("need a video path or --task-dir")

    video = _resolve_video(args)
    kw = dict(spike_k=args.spike_k, merge_gap_s=args.merge_gap_s,
              settle_margin_s=args.settle_margin_s, min_quiet_s=args.min_quiet_s,
              drop_forced_keyframes=not args.keep_forced_keyframes)
    candidates, diag = detect_slides(video, **kw)

    if args.json:
        print(json.dumps({"candidates": [
            {"t": t, "burst": b.to_dict()} for t, b in candidates], "diagnostics": diag},
            ensure_ascii=False, indent=2))
        return

    n_fake = sum(1 for kf in diag["keyframes"] if not kf["real"])
    print(f"[slide_coarse_det] {video}")
    print(f"  median packet={diag['median_bytes']}B  spike>{diag['spike_thr']}B (x{args.spike_k:g})  "
          f"dur={diag['duration']}s")
    print(f"  bursts: {diag['n_raw_bursts']} raw -> drop {diag['n_dropped_forced_kf']} forced-kf "
          f"-> {diag['n_merged_bursts']} merged (min_quiet={args.min_quiet_s}s)")
    print(f"  keyframes: {len(diag['keyframes'])} total, {n_fake} fake (forced keyint, no burst)")
    print(f"  {len(candidates)} candidate slide(s):")
    for k, (t, b) in enumerate(candidates):
        print(f"    #{k:<2} t={t:7.3f}s   burst=[{b.t_start:7.3f},{b.t_end:7.3f}]s "
              f"peak={b.peak_bytes}B")

    if args.out_dir:
        slides = extract_slides_coarse(video, args.out_dir, **kw)
        print(f"  wrote {len(slides)} PNG(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
