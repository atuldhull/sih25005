# App: the breed fields, and the one that must not be rendered as-is

For Person 1. This is the whole change; nothing else in the app needs touching.

There is no Flutter or Dart toolchain on the integration machine, so none of
the Dart below has been compiled. It is written out rather than committed for
exactly that reason — untested code in a working app is worse than no code.
The same panel has been built and tested on the web console
(`server/static/demo.html`, `server/test_demo_ui.js`, 22 checks) so the
behaviour it should have is pinned down and can be copied from there.

---

## 1. The urgent one: `breed_verified` is false on every record

`lib/models/scoring_result.dart:71` reads it:

```dart
breedVerified: j['breed_verified'],
```

It is not rendered anywhere yet, which is the only reason this has not already
caused a problem.

**`false` means NOT CHECKED. It does not mean the breed is wrong.**

The exact-breed model was measured at 38.1% on photographs from a source it had
never seen during training, and its confidence carries no useful signal
(tightening it from answering every time to answering 30% of the time moves
accuracy by 5.6 points). The model therefore disables its own breed head. The
server sends `false` only because the app declares the field non-nullable —
`server/main.py:370` coerces `null` to `false` for that reason, and the
contract records it at `contract/scoring_result.json:85`.

If a screen ever renders this as a red cross or "breed mismatch", it accuses
every correctly registered animal in the district.

Use `breed_verify_status` instead. It is a string, always one of:

| value | meaning | how it should look |
|---|---|---|
| `unverified` | never checked — the head is off | plain text, no warning styling |
| `agree` | the photograph matches the record | plain text |
| `disagree` | the photograph contradicts the record | warning styling, and worth a human check of the record — never an automatic correction |

## 2. What to show instead: the group head

This is the breed signal that actually works: **80.2% source-held-out, against
a 60.7% background-only control** (that control matters — an earlier breed model
scored 97.9% by learning the farm rather than the animal).

Fields already arriving in every `/session` response:

| field | type | notes |
|---|---|---|
| `predicted_species` | String | `cattle` or `buffalo` |
| `species_confidence` | double | |
| `species_consistent` | bool? | against the record |
| `predicted_group` | String? | `red_zebu`, `grey_draught`, `dwarf_cattle`, `exotic_dairy`, `buffalo` |
| `group_confidence` | double? | |
| `group_consistent` | bool? | against `breed_registered` |
| `group_reliable` | bool? | **false = show as a hint, never as a finding** — that group's own measured recall is poor (exotic_dairy is 43%) or confidence fell under the measured threshold |

## 3. Model change

Make the field nullable so the app cannot crash if the server ever stops
coercing, and pick up the additive fields:

```dart
final bool? breedVerified;
final String breedVerifyStatus;
final String? predictedSpecies;
final double? speciesConfidence;
final String? predictedGroup;
final double? groupConfidence;
final bool? groupConsistent;
final bool? groupReliable;
```

```dart
breedVerified: j['breed_verified'] as bool?,
breedVerifyStatus: j['breed_verify_status'] ?? 'unverified',
predictedSpecies: j['predicted_species'],
speciesConfidence: (j['species_confidence'] as num?)?.toDouble(),
predictedGroup: j['predicted_group'],
groupConfidence: (j['group_confidence'] as num?)?.toDouble(),
groupConsistent: j['group_consistent'] as bool?,
groupReliable: j['group_reliable'] as bool?,
```

All of these are additive — a build that ignores them still works, so this can
land whenever it suits.

## 4. Wording that has already been through review

From the tested web panel, if it helps to reuse:

- unverified → *"Registered as Gir. The exact-breed model is switched off —
  measured at 38.1% on photographs from a source it had never seen, which is not
  worth acting on. This is NOT a mismatch: the breed was never checked."*
- group agrees → *"red zebu — 100% confident, consistent with Gir."*
- group disagrees → *"does NOT match Gir, worth checking the record"*
- `group_reliable == false` → *"below this group's own reliability bar, treat as
  a hint"*

## 5. The one change that would matter most: send the tag close-up

`ScanTagScreen` already captures the ear tag, but the app uploads only the tag
NUMBER. If it also uploaded the photograph, the server now accepts it:

```dart
request.files.add(
  await http.MultipartFile.fromPath('tag_photo', tagPhotoPath));
```

`tag_photo` or `tag_image`, either name, optional, same 10 MB cap as the other
photos. Nothing breaks without it.

**What it changes.** Everything measured in centimetres — five of the twenty
traits, heart girth, and the weight — needs a real scale, and the only object
of known size in the frame is the tag. In the side photograph the tag is a
thumbnail and the detector frequently does not find it at all. In a close-up
it fills the frame, needs no detector, and its printed 18 mm digit row gives a
scale directly; the server then carries that scale to the side photograph
using the tag itself as a bridge.

Measured on a real pair, the same session posted twice:

| | engine used | weight |
|---|---|---|
| without `tag_photo` | `baseline` — every score a placeholder | invented |
| with `tag_photo` | **`ml-pipeline`** — real measurement | measured |

That is the difference between the demo showing demonstration data and showing
a measurement. The close-up needs to be of an **NDDB-spec** tag — barcode row,
two digit rows, or the round button — because those are the features of known
size. A handwritten management tag will be refused, with a reason.

## 6. Two smaller things

**Quality.** `quality_passed` can now be `false` on a session that still scored.
Blur was measured against what it actually costs: out to a heavy blur the pose
model's keypoint drift stays inside its own median error, and the blur score
saturates in exactly the range where usable and hopeless separate, so it cannot
be a gate. Soft images are recorded and scored, not refused. If the app shows
quality at all, it should say the session was scored and to weight it lower —
not that it failed.

**Field names on upload.** Already resolved server-side and pinned by
`server/test_app_contract.py`, listed here only so it is not rediscovered: the
app sends `tag_id` and `video`, the server originally required `animal_id` and
`gait_video`. Both aliases are now accepted. `device_session_id` is optional —
if absent the server derives one from the uploaded bytes, so a retry of the same
upload collapses to the same session instead of creating a duplicate. Sending a
real one is still better, since it is the offline queue's idempotency key.
