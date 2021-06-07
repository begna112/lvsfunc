"""
Clip rendering helpers.
"""
import vapoursynth as vs

from threading import Condition
from typing import BinaryIO, Callable, Dict, List, Optional, TextIO, Union
from concurrent.futures import Future
from functools import partial

from .progress import Progress, BarColumn, FPSColumn, TextColumn, TimeRemainingColumn
from .util import get_prop

core = vs.core

RenderCallback = Callable[[int, vs.VideoFrame], None]


class RenderContext:
    """
    Contains info on the current render operation.
    """
    clip: vs.VideoNode
    queued: int
    frames: Dict[int, vs.VideoFrame]
    frames_rendered: int
    timecodes: List[float]
    condition: Condition

    def __init__(self, clip: vs.VideoNode, queued: int) -> None:
        self.clip = clip
        self.queued = queued
        self.frames = {}
        self.frames_rendered = 0
        self.timecodes = [0.0]
        self.condition = Condition()


def finish_frame(outfile: Optional[BinaryIO], timecodes: Optional[TextIO], ctx: RenderContext) -> None:
    """
    Output a frame.

    :param outfile:   Output IO handle for Y4MPEG
    :param timecodes: Output IO handle for timecodesv2
    :param ctx:       Rendering context
    """
    if timecodes:
        timecodes.write(f"{round(ctx.timecodes[ctx.frames_rendered]*1000):d}\n")
    if outfile is None:
        return

    f: vs.VideoFrame = ctx.frames[ctx.frames_rendered]

    outfile.write("FRAME\n".encode("utf-8"))

    for i, p in enumerate(f.planes()):
        if f.get_stride(i) != p.width * f.format.bytes_per_sample:
            outfile.write(bytes(p))  # type: ignore
        else:
            outfile.write(p)  # type: ignore


def clip_async_render(clip: vs.VideoNode,
                      outfile: Optional[BinaryIO] = None,
                      timecodes: Optional[TextIO] = None,
                      callback: Union[RenderCallback, List[RenderCallback], None] = None) -> List[float]:
    """
    Render a clip by requesting frames asynchronously using clip.get_frame_async,
    providing for callback with frame number and frame object.

    This is mostly a re-implementation of VideoNode.output, but a little bit slower since it's pure python.
    You only really need this when you want to render a clip while operating on each frame in order
    or you want timecodes without using vspipe.

    :param clip:      Clip to render.
    :param outfile:   Y4MPEG render output BinaryIO handle. If None, no Y4M output is performed.
                      Use ``sys.stdout.buffer`` for stdout. (Default: None)
    :param timecodes: Timecode v2 file TextIO handle. If None, timecodes will not be written.
    :param callback:  Single or list of callbacks to be preformed. The callbacks are called
                      when each sequential frame is output, not when each frame is done.
                      Must have signature ``Callable[[int, vs.VideoNode], None]``
                      See :py:func:`lvsfunc.comparison.diff` for a use case (Default: None).

    :return:          List of timecodes from rendered clip.
    """
    cbl = [] if callback is None else callback if isinstance(callback, list) else [callback]

    ctx = RenderContext(clip, core.num_threads)

    def cb(f: Future[vs.VideoFrame], n: int) -> None:
        ctx.frames[n] = f.result()
        nn = ctx.queued

        while ctx.frames_rendered in ctx.frames:
            frame = ctx.frames[ctx.frames_rendered]
            ctx.timecodes.append(ctx.timecodes[-1]
                                 + get_prop(frame, "_DurationNum", int)
                                 / get_prop(frame, "_DurationDen", int))
            finish_frame(outfile, timecodes, ctx)
            [cb(ctx.frames_rendered, ctx.frames[ctx.frames_rendered]) for cb in cbl]
            del ctx.frames[ctx.frames_rendered]  # tfw no infinite memory
            ctx.frames_rendered += 1

        # enqueue a new frame
        if nn < clip.num_frames:
            ctx.queued += 1
            cbp = partial(cb, n=nn)
            clip.get_frame_async(nn).add_done_callback(cbp)  # type: ignore

        ctx.condition.acquire()
        ctx.condition.notify()
        ctx.condition.release()

    if outfile:
        if clip.format is None:
            raise ValueError("Cannot render a variable format clip to y4m!")
        if clip.format.color_family not in (vs.YUV, vs.GRAY):
            raise ValueError("Can only render YUV and GRAY clips to y4m!")
        if clip.format.color_family == vs.GRAY:
            y4mformat = "mono"
        else:
            ss = (clip.format.subsampling_w, clip.format.subsampling_h)
            if ss == (1, 1):
                y4mformat = "420"
            elif ss == (1, 0):
                y4mformat = "422"
            elif ss == (0, 0):
                y4mformat = "444"
            elif ss == (2, 2):
                y4mformat = "410"
            elif ss == (2, 0):
                y4mformat = "411"
            elif ss == (0, 1):
                y4mformat = "440"
            else:
                raise ValueError("What have you done")

        y4mformat = f"{y4mformat}p{clip.format.bits_per_sample}" if clip.format.bits_per_sample > 8 else y4mformat

        header = f"YUV4MPEG2 C{y4mformat} W{clip.width} H{clip.height} F{clip.fps_num}:{clip.fps_den} Ip A0:0\n"
        outfile.write(header.encode("utf-8"))

    if timecodes:
        timecodes.write("# timestamp format v2\n")

    ctx.condition.acquire()

    # seed threads
    for n in range(min(clip.num_frames, core.num_threads)):
        cbp = partial(cb, n=n)  # lambda won't bind the int immediately
        clip.get_frame_async(n).add_done_callback(cbp)  # type: ignore

    while ctx.frames_rendered != clip.num_frames:
        ctx.condition.wait()

    return ctx.timecodes  # might as well


def find_scene_changes(clip: vs.VideoNode, scxvid: bool = False) -> List[int]:
    """
    Generate a list of scene changes (keyframes).

    Dependencies:
    * vapoursynth-wwxd
    * vapoursynth-scxvid (Optional: scxvid mode)

    :param clip:   Clip to search for scene changes. Will be rendered in its entirety.
    :param scxvid: Use scxvid instead of wwxd.

    :return:       List of scene changes.
    """
    progress = Progress(TextColumn("{task.description}"),
                        BarColumn(),
                        TextColumn("{task.completed}/{task.total}"),
                        TextColumn("{task.percentage:>3.02f}%"),
                        FPSColumn(),
                        TimeRemainingColumn())
    frames = []
    clip = clip.resize.Bilinear(640, 360, format=vs.YUV420P8)
    clip = clip.scxvid.Scxvid() if scxvid else clip.wwxd.WWXD()
    with progress:
        task = progress.add_task("Detecting scene changes...", total=clip.num_frames)

        def _cb(n: int, f: vs.VideoFrame) -> None:
            progress.update(task, advance=1)
            p = get_prop(f, "_SceneChangePrev", int) if scxvid else get_prop(f, "Scenechange", int)
            if p == 1:
                frames.append(n)

        clip_async_render(clip, callback=_cb)
    return sorted(frames)
