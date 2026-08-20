"""Run the whole chain on ONE photo and say exactly where it breaks.

Every stage of this system has been verified alone. Nothing has ever run
detector -> tag ruler -> keypoints -> measurement on a single photograph, and
that is where the failures nobody predicted live: coordinate frames that do not
agree, a crop offset creeping back in, a scale in the wrong units. The ear_tag
box that came back at x1 = -50.9 was exactly that class of bug, and it cost
two days.

So this test does not just check that each stage returns something. It checks
that the stages agree with each other:

  * the tag box must lie INSIDE the animal box. A tag floating outside the
    animal means one of the two is in the wrong coordinate frame - which is
    the crop-offset bug, and no single-stage test can see it.
  * keypoints must land inside the animal box, in full-image pixels.
  * a measurement in centimetres must be biologically possible. A withers
    height of 4 cm or 40 m means the scale is wrong by orders of magnitude,
    and that is far more likely than a real animal that shape.

Stages are independent: if the detector will not import, pass boxes with
--animal-bbox / --tag-bbox and the rest of the chain still gets tested. That
matters because it lets the backend side run this without the ML repo.

  python chain_test.py --image cow.jpg --pose <ckpt.pt>
  python chain_test.py --image cow.jpg --pose <ckpt.pt> \
      --animal-bbox 332,804,3532,4784 --tag-bbox 486,1974,944,2501
"""
import argparse
import math
import sys
from pathlib import Path

FAIL, WARN, OK = "FAIL", "WARN", "PASS"
_results = []


def check(level, name, detail=""):
    _results.append((level, name, detail))
    mark = {OK: "PASS", WARN: "WARN", FAIL: "FAIL"}[level]
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def parse_bbox(s):
    if not s:
        return None
    v = [float(x) for x in s.replace(" ", "").split(",")]
    if len(v) != 4:
        raise SystemExit(f"--bbox needs 4 numbers, got {s}")
    return v


def inside(inner, outer, tol=0.02):
    """Is `inner` box within `outer`, allowing a small tolerance?"""
    ow, oh = outer[2] - outer[0], outer[3] - outer[1]
    return (inner[0] >= outer[0] - tol * ow
            and inner[1] >= outer[1] - tol * oh
            and inner[2] <= outer[2] + tol * ow
            and inner[3] <= outer[3] + tol * oh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pose", default="", help="pose checkpoint (.pt)")
    ap.add_argument("--repo", default="",
                    help="repo root, so ml.detection.detector can be imported")
    ap.add_argument("--animal-bbox", default="")
    ap.add_argument("--tag-bbox", default="")
    ap.add_argument("--annotate", default="chain_annotated.jpg")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"no such image: {img_path}")
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    print(f"\nimage {img_path.name}  {W} x {H}")

    animal = parse_bbox(args.animal_bbox)
    tag = parse_bbox(args.tag_bbox)

    # ---------------------------------------------------- 1. detection
    print("\n[1] DETECTION")
    if animal is None or tag is None:
        if args.repo and args.repo not in sys.path:
            sys.path.insert(0, args.repo)
        try:
            # The real signatures, read off ml-dev rather than assumed:
            #   detect_animal(image_path: str, device="auto")
            #   detect_ear_tag(image_path: str, animal_bbox, device="auto")
            # Both take a PATH and load with cv2.imread themselves; neither
            # takes an array. Both return None when nothing is found.
            from ml.detection.detector import detect_animal, detect_ear_tag
            a = detect_animal(str(img_path))
            animal = list(a.bbox) if a else None
            t = detect_ear_tag(str(img_path), tuple(animal) if animal
                               else None)
            tag = list(t.bbox) if t else None
            check(OK if animal else FAIL, "detect_animal returned a box",
                  f"{[round(v) for v in animal]}" if animal else "None")
            check(OK if tag else FAIL, "detect_ear_tag returned a box",
                  f"{[round(v) for v in tag]}" if tag else "None")
        except Exception as e:
            check(WARN, "detector not importable",
                  f"{type(e).__name__}: {str(e)[:90]} "
                  f"- pass --animal-bbox/--tag-bbox to test the rest")
    else:
        check(OK, "boxes supplied on the command line", "detector skipped")

    if animal:
        check(OK if inside(animal, [0, 0, W, H]) else FAIL,
              "animal box inside the image",
              f"{[round(v) for v in animal]} vs {W}x{H}")
        # clip before measuring, or a box that overruns the frame reports a
        # nonsensical share of it and hides the real problem
        cx1, cy1 = max(0.0, animal[0]), max(0.0, animal[1])
        cx2, cy2 = min(float(W), animal[2]), min(float(H), animal[3])
        frac = (max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)) / float(W * H)
        check(OK if frac > 0.05 else WARN, "animal fills a usable share",
              f"{100 * frac:.0f}% of frame (box clipped to image)")
    if tag:
        check(OK if inside(tag, [0, 0, W, H]) else FAIL,
              "tag box inside the image", f"{[round(v) for v in tag]}")
        if animal:
            # THE coordinate-frame test. A tag outside the animal means one
            # box is in a crop frame and the other in the full frame.
            check(OK if inside(tag, animal, tol=0.05) else FAIL,
                  "tag box lies INSIDE the animal box",
                  "if this fails, one box is in the wrong coordinate frame "
                  "- that is the crop-offset bug")
            ta = (tag[2] - tag[0]) * (tag[3] - tag[1])
            aa = (animal[2] - animal[0]) * (animal[3] - animal[1])
            check(OK if ta < 0.05 * aa else FAIL, "tag is tag-sized",
                  f"tag is {100 * ta / aa:.1f}% of the animal's area")

    # -------------------------------------------------- 2. tag ruler
    print("\n[2] SCALE FROM THE EAR TAG")
    scale = None
    if tag is None:
        check(WARN, "skipped - no tag box")
    else:
        try:
            from tag_ruler import (estimate_scale, ScaleResult,
                                   scale_error_fraction)
            bgr = np.asarray(pil)[:, :, ::-1].copy()
            res = estimate_scale(bgr, tag)
            if isinstance(res, ScaleResult):
                scale = res
                check(OK, "scale measured",
                      f"{res.cm_per_px:.5f} cm/px, button "
                      f"{res.button_axes_px[0]:.1f}px, conf {res.confidence}")
                err = scale_error_fraction(res)
                check(OK if err < 0.15 else WARN, "scale error is usable",
                      f"+/-{100 * err:.1f}%")
                if res.note:
                    check(WARN, "ruler note", res.note[:100])
            else:
                check(WARN, "ruler refused (this is a designed outcome)",
                      res.reason[:100])
        except ImportError as e:
            check(WARN, "tag_ruler not importable", str(e)[:90])
        except Exception as e:
            check(FAIL, "tag_ruler crashed", f"{type(e).__name__}: {e}")

    # -------------------------------------------------- 3. keypoints
    print("\n[3] KEYPOINTS")
    kps = None
    if not args.pose or animal is None:
        check(WARN, "skipped", "need --pose and an animal box")
    else:
        try:
            from bovine_pose_infer import BovinePoseModel
            pm = BovinePoseModel(args.pose)
            kps = pm.extract(pil, animal)
            trained = {k: v for k, v in kps.items() if v.confidence > 0}
            check(OK if trained else FAIL, "joints returned",
                  f"{len(trained)} usable of {len(kps)}")
            oob = [k for k, v in trained.items()
                   if not (animal[0] - 0.15 * (animal[2] - animal[0])
                           <= v.x <=
                           animal[2] + 0.15 * (animal[2] - animal[0]))]
            check(OK if not oob else FAIL,
                  "joints land in full-image coords near the animal",
                  f"{len(oob)} outside: {oob[:5]}" if oob
                  else "all inside the padded animal box")
            good = sorted(trained.items(), key=lambda kv: -kv[1].confidence)
            print("     strongest joints: " + ", ".join(
                f"{k}({v.confidence:.2f})" for k, v in good[:5]))
        except ImportError as e:
            check(WARN, "bovine_pose_infer not importable", str(e)[:90])
        except Exception as e:
            check(FAIL, "pose model crashed", f"{type(e).__name__}: {e}")

    # ------------------------------------------------ 4. measurement
    print("\n[4] MEASUREMENT IN CENTIMETRES")
    if kps is None or scale is None:
        check(WARN, "skipped",
              "needs BOTH keypoints and a scale - this is the stage that has "
              "never run end to end")
    else:
        try:
            from tag_ruler import measure_cm
        except ImportError:
            measure_cm = None
        # withers height: the classic class-C trait, and the one that exposes
        # a wrong scale immediately because its plausible range is narrow
        pairs = [("Stature (withers to ground)", "withers", "hoof_left"),
                 ("Body length (shoulder to pin)", "shoulder_left",
                  "pin_left")]
        for label, a_name, b_name in pairs:
            ka, kb = kps.get(a_name), kps.get(b_name)
            if not ka or not kb or ka.confidence <= 0 or kb.confidence <= 0:
                check(WARN, f"{label} - not measurable",
                      f"{a_name} or {b_name} unavailable")
                continue
            dpx = math.hypot(ka.x - kb.x, ka.y - kb.y)
            cm = dpx * scale.cm_per_px
            if measure_cm is not None:
                m = measure_cm(dpx, scale)
                shown = (f"{cm:.1f} cm"
                         + (f"  CI {m[0]:.0f}-{m[1]:.0f} cm"
                            if isinstance(m, (tuple, list)) and len(m) >= 2
                            else ""))
            else:
                shown = f"{cm:.1f} cm"
            plausible = 40.0 <= cm <= 260.0
            check(OK if plausible else FAIL, f"{label}",
                  shown + ("" if plausible else
                           "  <- biologically impossible: the scale is wrong "
                           "by orders of magnitude, not by a few percent"))

    # ------------------------------------------------- 5. annotate
    if animal or tag or kps:
        try:
            from PIL import ImageDraw
            ann = pil.copy()
            d = ImageDraw.Draw(ann)
            lw = max(2, int(min(W, H) * 0.004))
            if animal:
                d.rectangle(animal, outline=(40, 200, 90), width=lw)
            if tag:
                d.rectangle(tag, outline=(240, 60, 60), width=lw)
            if kps:
                r = max(3, int(min(W, H) * 0.005))
                for k, v in kps.items():
                    if v.confidence > 0:
                        d.ellipse([v.x - r, v.y - r, v.x + r, v.y + r],
                                  fill=(70, 140, 255))
            ann.thumbnail((1600, 1600))
            ann.save(args.annotate, quality=90)
            print(f"\nannotated image -> {args.annotate}")
            print("  OPEN IT. Green = animal, red = ear tag, blue = joints.")
            print("  No assertion can tell you the joints are on the right "
                  "anatomy; your eyes can.")
        except Exception as e:
            print(f"\ncould not annotate: {e}")

    # -------------------------------------------------- summary
    n_fail = sum(1 for l, _, _ in _results if l == FAIL)
    n_warn = sum(1 for l, _, _ in _results if l == WARN)
    print("\n" + "=" * 68)
    print(f"{len(_results)} checks: "
          f"{len(_results) - n_fail - n_warn} pass, {n_warn} warn, "
          f"{n_fail} fail")
    if n_fail:
        print("\nFAILURES:")
        for l, n, d in _results:
            if l == FAIL:
                print(f"  - {n}: {d}")
    elif n_warn:
        print("\nNo failures, but stages were skipped. The chain is not "
              "proven until\nsection 4 produces a plausible centimetre "
              "measurement - that is the only\ncheck that exercises every "
              "stage at once.")
    else:
        print("\nThe full chain ran and produced a plausible measurement. "
              "Now look at\nthe annotated image, then repeat on 3-4 more "
              "photos.")
    print("=" * 68)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
