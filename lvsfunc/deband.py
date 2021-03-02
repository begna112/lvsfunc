"""
    Various functions used for debanding.
    This used to be the `debandshit` module written by Z4ST1N,
    with some functions that were rarely (if ever) used removed because I can't reasonably maintain them.

    You can view the original python module here:
    https://github.com/LightArrowsEXE/debandshit
"""
import math
from functools import partial
from typing import Optional

import vapoursynth as vs
from vsutil import depth, get_depth, get_subsampling, get_y, join, plane, split

core = vs.core


def Dither_bilateral(clip: vs.VideoNode, ref: Optional[vs.VideoNode] = None,
                     radius: Optional[float] = None,
                     thr: float = 0.35, wmin: float = 1.0,
                     subspl: float = 0,
                     legacy_rounding: bool = False) -> vs.VideoNode:
    """
    Dither's Gradfun3 mode 2, for Vapoursynth.
    Not much different from using core.avsw.Eval with Dither_bilateral_multistage,
    just with normal value rounding and without Dither_limit_dif16.
    Default setting of 'radius' changes to reflect the resolution.

    Basic usage: flt = db.Dither_bilateral(clip, thr=1/3)
                 flt = mvf.LimitFilter(flt, clip, thr=1/3)

    For a more thorough explanation on the parameters, check out the AviSynth page:
    http://avisynth.nl/index.php/Dither_tools#Dither_bilateral16

    :param clip:                Input clip
    :param ref:                 When computing the value weighting, pixels of this clip are taken for reference
    :param radius:              Size of the averaged square. If `None`, auto-adjusted based resolution:
                                    480p  ->  9
                                    720p  -> 12
                                    810p  -> 13
                                    900p  -> 14
                                    1080p -> 16
                                    2K    -> 20
                                    4K    -> 32
    :param thr:                 Pixels whose difference of value with the center pixel is over thr will be discarded
    :param wmin:                The partial or missing pixels are completed with the center pixel
                                to match the wmin fraction of the filter window at full weight
    :param subspl:              Subsampling rate
    :param legacy_rounding:     Use legacy rounding

    :return:                    Debanded clip
    """
    if radius is None:
        radius = max((clip.width - 1280) / 160 + 12, (clip.height - 720) / 90 + 12)

    if get_subsampling(clip) == "vs.GRAY":
        planes = [clip, clip, clip]
    else:
        planes = split(clip)

    if legacy_rounding:
        thr_1 = max(thr * 4.5, 1.25)
        thr_2 = max(thr * 9, 5)
        subspl_2 = subspl if subspl in (0, 1) else subspl / 2
        r4 = math.floor(max(radius * 4 / 3, 4))
        r2 = math.floor(max(radius * 2 / 3, 3))
        r1 = math.floor(max(radius / 3, 2))
    else:
        thr_1 = round(max(thr * 4.5, 1.25), 1)
        thr_2 = round(max(thr * 9, 5), 1)
        subspl_2 = subspl if subspl in (0, 1) else subspl / 2
        r4 = round(max(radius * 4 / 3, 4))
        r2 = round(max(radius * 2 / 3, 3))
        r1 = round(max(radius / 3, 2))

    clips = [clip.fmtc.nativetostack16()]
    clip_names = ["c"]
    ref_t = "c"
    if ref is not None:
        ref_s = ref.fmtc.nativetostack16()
        clips += [ref_s]
        clip_names += ["ref"]
        ref_t = "ref"

    avs_stuff = f"c.Dither_bilateral16(radius={r4}, thr={thr_1}, flat=0.75, "
    avs_stuff += f"wmin={wmin}, ref={ref_t}, subspl={subspl}, "
    avs_stuff += f"y={planes[0]}, u={planes[1]}, v={planes[2]})"
    avs_stuff += f".Dither_bilateral16(radius={r2}, thr={thr_2}, flat=0.25, "
    avs_stuff += f"wmin={wmin}, ref={ref_t}, subspl={subspl_2} "
    avs_stuff += f"y={planes[0]}, u={planes[1]}, v={planes[2]})"
    avs_stuff += f".Dither_bilateral16(radius={r1}, thr={thr_2}, flat=0.50, "
    avs_stuff += f"wmin={wmin}, ref={ref_t}, subspl={subspl_2} "
    avs_stuff += f"y={planes[0]}, u={planes[1]}, v={planes[2]})"

    return core.avsw.Eval(avs_stuff, clips=clips, clip_names=clip_names).fmtc.stack16tonative()
# TODO: rewrite to take arrays for most parameters (should at least save the extra stack16 conversions)


def f3kbilateral(clip: vs.VideoNode,
                 range: Optional[int] = None,
                 y: int = 50, c: int = 0,
                 thr: float = 0.6, thrc: Optional[float] = None,
                 elast: float = 3.) -> vs.VideoNode:
    """
    f3kbilateral: f3kdb multistage bilateral-esque filter.

    This thing is more of a last resort for extreme banding.
    With that in mind, 40~60 is probably an effective range for y & c strengths.
    I did use range=20, y=160 to scene-filter some horrendous fades, though.

    :param clip:    Input clip
    :param range:   Banding detection range
    :param y:       Banding detection threshold for luma
    :param c:       Banding detection threshold for chroma
    :param thr:     Threshold (on an 8-bit scale) to limit filtering diff for the luma
    :param thrc:    Threshold (on an 8-bit scale) to limit filtering diff for the chroma
    :param elast:   Elasticity of the soft threshold

    :return:        Debanded clip
    """
    try:
        from mvsfunc import LimitFilter
    except ModuleNotFoundError:
        raise ModuleNotFoundError("f3kbilateral: missing dependency 'mvsfunc'")

    range = (12 if clip.width < 1800 and clip.height < 1000 else 16) if range is None else range
    r1 = round(range*4/3)
    r2 = round(range*2/3)
    r3 = round(range/3)
    y1 = y // 2
    y2, y3 = y, y
    c1 = c // 2
    c2, c3 = c, c

    if c == 0:
        flt0 = get_y(clip)
    flt0 = depth(clip, 16)

    flt1 = f3kdb_mod(flt0, r1, y1, c1)
    flt2 = f3kdb_mod(flt1, r2, y2, c2)
    flt3 = f3kdb_mod(flt2, r3, y3, c3)

    flt = LimitFilter(flt3, flt2, ref=flt0, thr=thr, elast=elast, thrc=thrc)
    flt = depth(flt, get_depth(clip))

    if c == 0 and (get_subsampling(clip) is not None):
        return join([flt, plane(clip, 1), plane(clip, 2)])
    return flt


def f3kpf(clip: vs.VideoNode,
          range: int = None,
          y: int = 40, cb: int = 40, cr: Optional[int] = None,
          thr: float = 0.3, thrc: Optional[float] = None,
          elast: float = 2.5) -> vs.VideoNode:
    """
    f3kdb with a simple prefilter by mawen1250 - https://www.nmm-hd.org/newbbs/viewtopic.php?f=7&t=1495#p12163.

    Since the prefilter is a straight gaussian+average blur, f3kdb's effect becomes very strong, very fast.
    Functions more or less like gradfun3 without the detail mask.

    :param clip:    Input clip
    :param range:   Banding detection range
    :param y:       Banding detection threshold for luma
    :param cb:      Banding detection threshold for first chroma plane
    :param cr:      Banding detection threshold for second chroma plane.
                    Same value as `cb` if `None`
    :param thr:     Threshold (on an 8-bit scale) to limit filtering diff for the luma
    :param thrc:    Threshold (on an 8-bit scale) to limit filtering diff for the chroma
    :param elast:   Elasticity of the soft threshold

    :return:        Debanded clip
    """
    try:
        from mvsfunc import LimitFilter
    except ModuleNotFoundError:
        raise ModuleNotFoundError("f3kpf: missing dependency 'mvsfunc'")

    range = (12 if clip.width < 1800 and clip.height < 1000 else 16) if range is None else range
    cr = cb if cr is None else cr

    if cr == 0 and cb == 0:
        clp = get_y(clip)
        clp = depth(clp, 32)
    else:
        clp = depth(clip, 32)

    blur32 = clp.std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1] * 9, planes=0)
    blur16 = depth(blur32, 16)
    diff = clp.std.MakeDiff(blur32)
    f3k = f3kdb_mod(blur16, range, y, cb, cr)
    f3k = LimitFilter(f3k, blur16, thr=thr, elast=elast, thrc=thrc)
    f3k = depth(f3k, 32)
    out = f3k.std.MergeDiff(diff)
    out = depth(out, get_depth(clip))
    if cr == 0 and cb == 0 and not clip.format.color_family == vs.GRAY:
        out = join([out, plane(clip, 1), plane(clip, 2)])
    return out


def lfdeband(clip: vs.VideoNode) -> vs.VideoNode:
    """
    Ported from an AviSynth script.

    :param clip:    Input clip

    :return:        Debanded clip
    """
    wss = 1 << clip.format.subsampling_w
    hss = 1 << clip.format.subsampling_h
    w = clip.width
    h = clip.height
    dw = round(w / 2)
    dh = round(h / 2)

    clp = depth(clip, 32)
    dsc = clp.fmtc.resample(dw - dw % wss, dh - dh % hss, kernel='spline64')
    ddb = f3kdb_mod(dsc, range=30, y=80, cb=80, cr=80, grainy=0, grainc=0)
    ddif = ddb.std.MakeDiff(dsc)
    dif = ddif.fmtc.resample(w, h, kernel='spline64')
    out = clp.std.MergeDiff(dif)
    return depth(out, get_depth(clip))


def f3kdb_mod(clip: vs.VideoNode,
              range: Optional[int] = None, y: int = 40,
              cb: Optional[int] = None, cr: Optional[int] = None,
              grainy: int = 0, grainc: int = 0) -> vs.VideoNode:
    """
    f3kdb wrapper function.
    Allows 32 bit in/out, and sets most parameters automatically; otherwise it's just f3kdb.

    :param clip:        Input clip
    :param range:       Banding detection range
    :param y:           Banding detection threshold for luma
    :param cb:          Banding detection threshold for first chroma plane.
                        Half of `y` if `None` and clip is an RGB clip
    :param cr:          Banding detection threshold for second chroma plane.
                        Same value as `cb` if `None`
    :param grainy:      Grain added to the luma plane
    :param grainc:      Grain added to the chroma planes

    :return:            Debanded clip
    """
    f = clip.format
    cf = f.color_family
    bits = f.bits_per_sample
    tv_range = cf == vs.GRAY or cf == vs.YUV

    range = (12 if clip.width < 1800 and clip.height < 1000 else 16) if range is None else range
    cb = (y if get_subsampling(clip) is None else y // 2) if cb is None else cb
    cr = cb if cr is None else cr

    if bits > 16:
        clip = depth(clip, 16)

    try:
        f3kdb = core.neo_f3kdb.Deband
    except AttributeError:
        f3kdb = core.f3kdb.Deband

    deb = f3kdb(clip, range, y, cb, cr, grainy, grainc, keep_tv_range=tv_range, output_depth=min(get_depth(clip), 16))

    if get_depth(clip) == 16:
        return depth(deb, get_depth(clip))
