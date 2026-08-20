"""Breed verification at inference time, with abstention built in.

This is what the pipeline calls. It answers the question the result contract
actually asks - "does this photo support the breed on the BPA record?" - and
it is allowed to say no answer.

DESIGN RULES, EACH ONE THERE BECAUSE THE ALTERNATIVE MISLEADS SOMEBODY

1. The confidence threshold is not hardcoded. It is read from the coverage
   table measured during training and stored in the checkpoint. If the model
   turns out weaker after a retrain, the threshold tightens by itself instead
   of silently letting bad answers through.

2. A breed the model was never trained on returns abstain, not "disagree".
   The Catbuf/luke pool covers a few dozen breeds; India has far more. An
   animal whose recorded breed is outside the class list is unverifiable, and
   saying "does not match" about it would be a false accusation against a
   record that is probably correct.

3. The species call is reported separately and trusted more. Cattle versus
   buffalo is a far easier question than Gir versus Sahiwal, it is right much
   more often, and it decides which trait rubric applies downstream - so it
   is worth surfacing on its own even when the breed call abstains.

4. Whatever honest accuracy was measured travels with every answer, in
   'measured_accuracy'. Nothing downstream has to guess how much to trust
   this, and nothing can quote a number the model did not earn.

  from breed_verify_infer import BreedVerifier
  v = BreedVerifier(r"D:\\bovine-pose\\clean\\breed_merged\\breed_verifier.pt")
  out = v.verify(image, bbox=(x1, y1, x2, y2), claimed_breed="Gir")
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Where to sit on the accuracy/coverage curve. 0.80 means: pick the confidence
# cut at which the model answered 80% of validation images, and accept the
# accuracy it had on those. Answering less often buys accuracy.
TARGET_COVERAGE = 0.80
# Never answer at all if the honest margin over the background control was
# smaller than this - at that point the model is reading scenery.
MIN_HONEST_GAP = 0.15
# A head must also be right often enough to be worth consulting. The first
# real run cleared MIN_HONEST_GAP with a +20.4 point margin and still only
# managed 39.8% when it answered, because the gap test says the signal is
# real - not that it is sufficient. Both gates now have to pass.
MIN_EXPECTED_ACC = 0.70
# And abstention has to actually buy something. If tightening the threshold
# from full coverage to 30% gains less than this, confidence carries no
# information and the threshold is decoration.
MIN_ABSTENTION_LIFT = 0.10
# Species is the most reliable head, but it is not infallible: the first run
# called a Tharparkar a buffalo at confidence 1.0. So it abstains too.
MIN_SPECIES_CONF = 0.90


class BreedVerifier:
    def __init__(self, ckpt_path, device=None):
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"no breed verifier at {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.classes = ck["classes"]
        self.species_names = ck["species"]
        self.breed_to_species = ck.get("breed_to_species", {})
        self.T = float(ck.get("temperature", 1.0))
        self.input_size = int(ck["input_size"])
        self.backbone_name = ck["backbone"]
        self._W, self._b = ck["W"], ck["b"]
        self._mu, self._sd = ck["mu"], ck["sd"]
        self._sW, self._sb = ck["sp_W"], ck["sp_b"]
        self._smu, self._ssd = ck["sp_mu"], ck["sp_sd"]

        m = ck.get("metrics", {})
        self.split_used = ck.get("trained_on_split", "unknown")
        self.metrics = m.get(self.split_used, {})
        self.breed_acc = float(self.metrics.get("breed_acc", float("nan")))
        self.background_acc = float(self.metrics.get("background_acc",
                                                    float("nan")))
        self.species_acc = float(self.metrics.get("species_acc",
                                                  float("nan")))
        self.honest_gap = self.breed_acc - self.background_acc

        # threshold and its expected accuracy, straight from the measured
        # coverage curve rather than from an optimistic guess
        cov_rows = self.metrics.get("coverage", [])
        self.threshold, self.expected_acc = 1.01, float("nan")
        for cov, acc, minconf in cov_rows:
            if abs(cov - TARGET_COVERAGE) < 1e-6:
                self.threshold, self.expected_acc = float(minconf), float(acc)
        full = next((a for c, a, _ in cov_rows if abs(c - 1.0) < 1e-6),
                    float("nan"))
        tight = next((a for c, a, _ in cov_rows if abs(c - 0.3) < 1e-6),
                     float("nan"))
        self.abstention_lift = tight - full

        # Three independent gates, all of which must pass. Any one failing
        # means the head stays quiet and the reason is reported.
        self.breed_off_reason = None
        if not (self.honest_gap == self.honest_gap):
            self.breed_off_reason = "no measured background control"
        elif self.honest_gap < MIN_HONEST_GAP:
            self.breed_off_reason = (
                f"margin over the background control is only "
                f"{100 * self.honest_gap:+.1f} points, so its predictions "
                f"reflect where the photo was taken more than which breed "
                f"it shows")
        elif not (self.expected_acc >= MIN_EXPECTED_ACC):
            self.breed_off_reason = (
                f"even at its chosen threshold it is right only "
                f"{100 * self.expected_acc:.1f}% of the time, below the "
                f"{100 * MIN_EXPECTED_ACC:.0f}% needed to be worth acting on")
        elif not (self.abstention_lift >= MIN_ABSTENTION_LIFT):
            self.breed_off_reason = (
                f"abstaining does not help: tightening from full coverage to "
                f"30% moves accuracy only {100 * self.abstention_lift:+.1f} "
                f"points, so its confidence carries no information")
        self.breed_enabled = self.breed_off_reason is None

        # the coarser claim, which the confusion structure says the images can
        # actually support
        self.group_names = ck.get("group_names", [])
        self.breed_to_group = ck.get("breed_to_group", {})
        self._gW, self._gb = ck.get("g_W"), ck.get("g_b")
        self._gmu, self._gsd = ck.get("g_mu"), ck.get("g_sd")
        self._gT = float(ck.get("g_T", 1.0))
        self.group_acc = float(self.metrics.get("group_acc", float("nan")))
        self.group_background = float(self.metrics.get("group_background",
                                                       float("nan")))
        self.group_gap = self.group_acc - self.group_background
        self.group_enabled = (self._gW is not None
                              and self.group_gap == self.group_gap
                              and self.group_gap >= MIN_HONEST_GAP
                              and self.group_acc >= MIN_EXPECTED_ACC)
        # Same trap as the breed head: a source-held-out split may not contain
        # every group, and an untrained group column will still happily win an
        # argmax. dwarf_cattle was absent from the first honest run, so a
        # Vechur would have been confidently filed as something else.
        self.trained_groups = ck.get("trained_groups", self.group_names)
        # Per-group recall, so a group the model handles badly can be reported
        # as unreliable instead of being averaged away by a good overall
        # number. exotic_dairy managed 43% while buffalo managed 98%, and
        # exotic-vs-indigenous decides which rubric applies at all.
        self.group_recall = ck.get("group_recall", {})
        self.weak_groups = sorted(g for g, r in self.group_recall.items()
                                  if r < MIN_EXPECTED_ACC)

        # The group head needs its own abstention cut, from its own measured
        # coverage curve. Without one it answered a held-out Jersey at 0.554
        # confidence, got the group wrong, and reported group_consistent=False
        # - a false accusation against a correctly registered animal. A
        # threshold cannot fix a CONFIDENT error, but it removes the whole
        # class of low-confidence ones, and those are the majority.
        gcov = self.metrics.get("group_coverage", [])
        self.group_threshold, self.group_expected_acc = 0.0, float("nan")
        for cov, acc, minconf in gcov:
            if abs(cov - TARGET_COVERAGE) < 1e-6:
                self.group_threshold = float(minconf)
                self.group_expected_acc = float(acc)

        # Only breeds the probe actually trained on can be predicted or
        # checked. The rest are unverifiable, which is different from wrong.
        self.trained_classes = ck.get("trained_classes", self.classes)

        self._backbone = None

    # -------------------------------------------------------------- private
    def _load(self):
        """Rebuild the backbone at the resolution the probe was fitted on.

        The probe's weights only mean anything against features from the same
        input size. Training may have overridden the backbone's native size
        (518 for DINOv2 is a GPU-only luxury), so that size is stored in the
        checkpoint and reapplied here via dynamic_img_size. Rebuilding at the
        native default instead throws a shape assertion - which is the good
        outcome; the bad one would be silently different features.
        """
        if self._backbone is None:
            import timm
            kw = dict(pretrained=True, num_classes=0)
            probe = timm.create_model(self.backbone_name, pretrained=False,
                                      num_classes=0)
            native = probe.pretrained_cfg.get("input_size",
                                              (3, 224, 224))[-1]
            if int(native) != self.input_size:
                kw.update(img_size=self.input_size, dynamic_img_size=True)
            m = timm.create_model(self.backbone_name, **kw)
            m.eval().to(self.device)
            for q in m.parameters():
                q.requires_grad = False
            self._backbone = m
        return self._backbone

    def _features(self, image, bbox):
        im = image if isinstance(image, Image.Image) else Image.open(image)
        im = im.convert("RGB")
        if bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            pw, ph = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
            im = im.crop((max(0, x1 - pw), max(0, y1 - ph),
                          min(im.width, x2 + pw), min(im.height, y2 + ph)))
        if min(im.size) < 8:
            return None
        a = np.asarray(im.resize((self.input_size, self.input_size),
                                 Image.BILINEAR), dtype=np.float32) / 255.0
        a = ((a - MEAN) / STD).transpose(2, 0, 1)
        x = torch.from_numpy(a[None]).to(self.device)
        with torch.no_grad():
            return self._load()(x).float().cpu()

    @staticmethod
    def _probs(f, W, b, mu, sd, T=1.0):
        fn = (f - torch.as_tensor(mu, dtype=f.dtype)) / \
             torch.as_tensor(sd, dtype=f.dtype)
        return torch.softmax((fn @ W + b) / T, 1)[0].numpy()

    # --------------------------------------------------------------- public
    def verify(self, image, bbox=None, claimed_breed=None):
        """Compare a photo against a recorded breed.

        Returns the contract fields plus enough context to explain itself:
        breed_verified is True, False, or None - and None is a legitimate,
        expected outcome rather than an error.
        """
        out = {"breed_verified": None, "breed_verify_confidence": None,
               "predicted_breed": None, "predicted_species": None,
               "species_confidence": None, "abstained": True,
               "reason": None, "top3": [],
               "measured_accuracy": {
                   "split": self.split_used,
                   "breed": round(self.breed_acc, 4)
                   if self.breed_acc == self.breed_acc else None,
                   "background_control": round(self.background_acc, 4)
                   if self.background_acc == self.background_acc else None,
                   "species": round(self.species_acc, 4)
                   if self.species_acc == self.species_acc else None,
                   "expected_at_this_threshold": round(self.expected_acc, 4)
                   if self.expected_acc == self.expected_acc else None}}

        f = self._features(image, bbox)
        if f is None:
            out["reason"] = "crop too small to assess"
            return out

        sp = self._probs(f, self._sW, self._sb, self._smu, self._ssd)
        si = int(sp.argmax())
        sconf = float(sp[si])
        out["species_confidence"] = round(sconf, 3)
        # Reported only when confident. A wrong species call is worse than
        # none: it selects the wrong trait rubric for everything downstream.
        if sconf >= MIN_SPECIES_CONF:
            out["predicted_species"] = self.species_names[si]
        else:
            out["species_note"] = (
                f"species withheld: {sconf:.2f} is below the "
                f"{MIN_SPECIES_CONF:.2f} needed to pick a trait rubric")

        # the group claim, reported whenever it is trustworthy - it stands on
        # its own even when the exact-breed head is silent
        if self.group_enabled:
            gp = self._probs(f, self._gW, self._gb, self._gmu, self._gsd,
                             self._gT)
            gmask = np.array([g in self.trained_groups
                              for g in self.group_names])
            if gmask.any():
                gp = np.where(gmask, gp, 0.0)
                tot = gp.sum()
                gp = gp / tot if tot > 0 else gp
            gi = int(gp.argmax())
            gname = self.group_names[gi]
            out["predicted_group"] = gname
            out["group_confidence"] = round(float(gp[gi]), 3)
            gconf = float(gp[gi])
            confident = gconf >= self.group_threshold
            if not confident:
                out["group_reliable"] = False
                out["group_note"] = (
                    f"confidence {gconf:.2f} is below the "
                    f"{self.group_threshold:.2f} cut measured for "
                    f"{100 * TARGET_COVERAGE:.0f}% coverage - reported for "
                    f"information, not to be acted on")
            elif gname in self.weak_groups:
                out["group_reliable"] = False
                out["group_note"] = (
                    f"'{gname}' was only {100 * self.group_recall[gname]:.0f}%"
                    f" correct on the held-out source - treat this call as a "
                    f"hint, not a finding")
            else:
                out["group_reliable"] = True
            if claimed_breed is not None:
                want = self.breed_to_group.get(claimed_breed)
                if want is None:
                    out["group_note"] = (
                        f"'{claimed_breed}' has no group mapping")
                elif want not in self.trained_groups:
                    out["group_note"] = (
                        f"the recorded breed's group '{want}' was not in the "
                        f"training split, so it cannot be checked")
                elif not confident:
                    # deliberately leave group_consistent None: contradicting
                    # a record is an accusation, and it needs confidence
                    pass
                else:
                    out["group_consistent"] = bool(gname == want)

        pr = self._probs(f, self._W, self._b, self._mu, self._sd, self.T)
        # Zero out breeds the probe never trained on. Their weights only ever
        # received negative gradient, so any confidence they carry is an
        # artefact - and reporting one as a prediction is a false accusation
        # against a record that is probably correct.
        mask = np.array([c in self.trained_classes for c in self.classes])
        if mask.any():
            pr = np.where(mask, pr, 0.0)
            s = pr.sum()
            pr = pr / s if s > 0 else pr
        order = np.argsort(-pr)[:3]
        out["top3"] = [(self.classes[i], round(float(pr[i]), 3))
                       for i in order]
        top = int(order[0])
        out["predicted_breed"] = self.classes[top]
        conf = float(pr[top])

        if not self.breed_enabled:
            out["reason"] = (f"breed head disabled - {self.breed_off_reason}."
                             + (f" Group ('{out.get('predicted_group')}') and "
                                f"species are still reported."
                                if self.group_enabled else
                                " Species is still reported."))
            return out

        if claimed_breed is not None and \
                claimed_breed not in self.trained_classes:
            out["reason"] = (f"recorded breed '{claimed_breed}' is not among "
                             f"the {len(self.trained_classes)} breeds this "
                             f"model was actually trained on, so it cannot be "
                             f"checked")
            return out

        if conf < self.threshold:
            out["breed_verify_confidence"] = round(conf, 3)
            out["reason"] = (f"confidence {conf:.2f} is below the "
                             f"{self.threshold:.2f} cut measured for "
                             f"{100 * TARGET_COVERAGE:.0f}% coverage")
            return out

        out["abstained"] = False
        out["breed_verify_confidence"] = round(conf, 3)
        if claimed_breed is None:
            out["reason"] = "no recorded breed supplied; prediction only"
            return out

        agree = self.classes[top] == claimed_breed
        out["breed_verified"] = bool(agree)
        out["reason"] = (
            f"image supports the recorded breed '{claimed_breed}'"
            if agree else
            f"image looks like '{self.classes[top]}', not the recorded "
            f"'{claimed_breed}' - worth a human check of the record")
        return out

    def summary(self):
        L = [f"BreedVerifier | split '{self.split_used}' | "
             f"{len(self.trained_classes)} breeds actually trained "
             f"(of {len(self.classes)} in the index)",
             f"  breed  {100 * self.breed_acc:.1f}%  vs background "
             f"{100 * self.background_acc:.1f}%  "
             f"(margin {100 * self.honest_gap:+.1f} pts, "
             f"abstention lift {100 * self.abstention_lift:+.1f} pts)"]
        if self.group_acc == self.group_acc:
            L.append(f"  group  {100 * self.group_acc:.1f}%  vs background "
                     f"{100 * self.group_background:.1f}%  "
                     f"(margin {100 * self.group_gap:+.1f} pts)")
            if self.group_expected_acc == self.group_expected_acc:
                L.append(f"         answers above {self.group_threshold:.2f} "
                         f"-> ~{100 * self.group_expected_acc:.1f}% correct "
                         f"when it answers")
        L.append(f"  species {100 * self.species_acc:.1f}%")
        L.append(f"  breed head {'ENABLED' if self.breed_enabled else 'OFF'}"
                 + (f", threshold {self.threshold:.2f} -> "
                    f"~{100 * self.expected_acc:.1f}% when it answers"
                    if self.breed_enabled
                    else f"\n     because {self.breed_off_reason}"))
        L.append(f"  group head {'ENABLED' if self.group_enabled else 'OFF'}"
                 + (f", {len(self.trained_groups)} groups trained: "
                    f"{', '.join(self.trained_groups)}"
                    if self.group_enabled else ""))
        if self.group_recall:
            for g in sorted(self.group_recall,
                            key=lambda g: -self.group_recall[g]):
                flag = "  <- unreliable" if g in self.weak_groups else ""
                L.append(f"     {g:<15}{100 * self.group_recall[g]:>6.0f}%"
                         f"{flag}")
        if not self.breed_enabled and not self.group_enabled:
            L.append("  -> species only; breed_verified stays null")
        return "\n".join(L)


if __name__ == "__main__":
    import sys
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\bovine-pose\clean\breed_merged\breed_verifier.pt"
    v = BreedVerifier(ck)
    print(v.summary())
    if len(sys.argv) > 2:
        print(v.verify(sys.argv[2],
                       claimed_breed=sys.argv[3] if len(sys.argv) > 3
                       else None))
