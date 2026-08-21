# Stage run-sheet

Read this at 6am. Everything below has been run and verified on this laptop.

---

## 1. Twenty minutes before you present

Open a Command Prompt and run **one** command:

    D:\sih25005-integration\server\stage_ready.bat

It does three things in order: resets and reseeds the demo story with fresh
dates, runs the GO / NO-GO preflight, then starts the server on all interfaces.
Leave that window open — closing it stops the server.

**Read the last line it prints.** It gives you the address the app must use, and
it changes every time you join a different network:

    [OK]   GIVE THE APP THIS ADDRESS - http://192.168.1.45:8000

Then, while you still have time to fix things:

| | |
|---|---|
| **Warm the chat model** | Open `http://localhost:8000/chat-ui` and ask one throwaway question. `gemma2:9b` is 5.4 GB and gets paged out — cold, the first answer crawls. Warm, it answers in seconds. Do this **at least 5 minutes before**. |
| **Start the emulator** | Android Studio → Device Manager → ▶ on *Medium Phone API 36.0*. It now uses your laptop webcam as its camera. |
| **Open the app** | It should say **✓ All synced** at the top. |
| **Check Settings** | Connection row must be green: *"Connected. The measurement pipeline is loaded."* If it is not, the server address is wrong — tap it and fix it. |

---

## 2. One thing that changed, and you need to know before you're asked

Preflight will print this, and it is **not a bug**:

    [FAIL] LIVE TRIGGER would NOT fire on stage - engine 'ml-pipeline'
           returned no symptoms

The old script had you score an animal live and watch an outbreak alert appear.
That only ever worked because the baseline engine **invented** a `skin_nodules`
finding at confidence 0.82. That invention has been removed — there is no
trained symptom detector in this build, so the system now reports nothing
rather than something.

**Do not try to fire a live outbreak.** Show the alerts feed as what it is:
seeded demonstration data, marked as such on every card.

If someone asks why, that is your strongest answer of the day:

> "It used to fire. It fired because the server was inventing a symptom when
> the real pipeline had nothing to say. We took that out this week. The alert
> feed you're looking at is demonstration data and it says so on every card —
> because an entry in a vet's feed is a request for a person to drive to a
> farm."

---

## 3. The demo, in order

Roughly eight minutes. Every number below is what it actually produces.

### ① Capture a session — the app (3 min)

**Scan** tab → *Capture Ear Tag* → **Confirm** → type **`356279812346`** →
side photo → *YES* → *YES* → rear photo → *YES* → *YES* → record 8 s →
**CONTINUE**.

Wait for "Scoring…", then:

    3 of 20 traits measured
    Weight 173-274 kg

Tap **VIEW SCORECARD**.

### ② The scorecard is the pitch (2 min)

Scroll slowly. Point at these four, in this order:

1. **MEASURED** banner — *"3 of 20 were measured. The other 17 were refused."*
2. **Breed identity** — *red zebu, 100% confident, consistent with Sahiwal.*
   Then read the line underneath aloud: the exact-breed model is switched off
   because it measured **38.1%** on photographs from a source it had never
   seen. *"It disables itself rather than guess."*
3. **Weight 173–274 kg** — a range, and a second method that **disagrees**.
   *"Two independent routes, and we show you that they don't agree."*
4. **Any refused trait** — tap one and read the reason. e.g.
   *"measured 54.90 cm, outside the calibrated range. The measurement itself is
   tight (±5.23), so the landmarks are more likely wrong than the animal
   unusual."*

> The line to land: **"Seventeen refusals is not the system failing. It's the
> system telling you what it doesn't know."**

### ③ The farmer assistant (1.5 min)

**Assistant** tab → **Sahiwal 356279812346** → tap the Hindi chip
**वज़न कितना है?**

It answers in Hindi, with the range and the disagreement. Then switch language
(translate icon, top right) to **ಕನ್ನಡ** and ask anything — it answers in
Kannada.

Then pick **Gir 356279812345** and ask *"What is her weight?"* — it **refuses**,
because that animal's only sessions are demonstration ones.

> **"Same app, same question, different animal. It answers when it measured,
> and refuses when it didn't."**

### ④ The vet officer's feed (1 min)

**Alerts** tab. Point at the red banner on every card, and the outbreak card:
*skin_nodules in Anand — 3 animals in 14 days.*

Say the paragraph from section 2.

### ⑤ Records (30 s)

**Records** → `356279812345` → *View History*. The weight trend says
**"No measured weights yet"** and each row reads **placeholder / not a
measurement**.

> **"The same numbers a chart would happily have drawn as a rising trend."**

---

## 4. If a judge is technical, do this

This is the strongest thirty seconds you have.

**Settings** → turn **Demo camera mode OFF** (or leave it on and use Gallery) →
**Scan** → **Gallery** → pick *any* photo that is not a cow. A chair, a
whiteboard, their face.

It comes back:

    NOTHING COULD BE MEASURED
    no_animal_detected — 0 of 20, no weight, no alert

> **"Before this week that returned twenty confident scores, a weight of
> 348 kg, and a Lumpy Skin Disease alert in a vet's feed. From a photograph of
> a chair."**

---

## 5. Camera and upload

Both work, in both modes, on every capture screen:

- **Take a photo** — Settings → Demo camera mode **OFF**. The emulator uses your
  laptop webcam, so hold a printed tag or a photo up to it. On a real phone it
  uses the real camera.
- **Upload a photo** — the **Gallery** button, next to the shutter. Works in
  demo mode too. This is the one that always works, so prefer it under pressure.

To get photos onto the emulator, drag and drop the file onto the emulator
window, or:

    adb push D:\sih25005-demo-photos\1-side.jpg /sdcard/Pictures/

---

## 6. Do not do these

- **Do not run the server test suite.** `test_demo.py` calls `demo_seed.main()`,
  which wipes every captured session. If you run it after capturing something
  on stage, that capture is gone — and the app will still say it synced,
  correctly, because it did.
- **Do not close the `stage_ready` window.** That is the server.
- **Do not connect to open venue WiFi.** The API has no auth.
- **Do not promise the weight is accurate.** It is not — see section 7.

---

## 7. The three questions you will be asked

**"Is that weight right?"**
> "No, and we can tell you exactly why. It's derived from the ear tag, and the
> tag we're demonstrating with is a rendering, not the tag that cow wears.
> Weight goes as the cube of that scale error. We measured what a correct scale
> would give: 382 to 429 kg. The machinery is right; the ruler is borrowed."

**"Only 3 out of 20 traits — isn't that a failure?"**
> "Ten of the twenty need udder and teat landmarks nobody has annotated yet.
> Five need a centimetre scale, which needs a conformant tag. We could widen
> the ranges tomorrow and score fifteen — none of them would be more correct.
> The refusals are the system catching its own bad landmarks."

**"How do I know it isn't just making this up?"**
> Show them section 4. Then: "Every response carries an `engine` field.
> `ml-pipeline` means measured, `baseline` means demonstration, and every
> screen in the app renders those differently."

---

## 8. If something breaks

| symptom | fix |
|---|---|
| App says "server could not be reached" | Settings → Server address. Emulator: `http://10.0.2.2:8000`. Real phone: the address `run_server.bat` printed. |
| Chat takes forever | The model was paged out. Ask one question and wait; the second is fast. |
| Scorecard shows red DEMONSTRATION DATA | The tag close-up did not reach the server, or nothing measured. Retake with the tag filling the frame. |
| Emulator camera is black | Demo camera mode OFF requires a working webcam. Switch back to demo, or use **Gallery**. |
| Everything is broken | Close the server window and re-run `stage_ready.bat`. It is safe to re-run at any time. |

---

## 9. What is honestly not finished

Say these before a judge finds them. They land far better volunteered.

- **The weight needs a real tag photographed with its size stated.** Field work,
  not code.
- **10 of 20 traits need an annotation session** for udder and teat landmarks.
- **No trained symptom detector**, so no health screening runs. The app says
  NOT SCREENED rather than "no problems found".
- **The exact-breed model is switched off** at 38.1% source-held-out. The
  breed-group head ships at 80.2%.
- **Nothing has been tested on a physical phone** — emulator only.
