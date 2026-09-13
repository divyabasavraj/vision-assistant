# Demo run sheet — Vision Assistant

Eight minutes, three acts. The point of the order is that Act 1 sets up a limitation
that Act 2 resolves, so the architecture argument makes itself.

## The night before

- [ ] **Phone-test the two unverified changes.** Does `SCORE_THRESHOLD = 0.5` still detect
      curb / pole / crosswalk? Is the stereo cue on the correct side with earbuds in?
      If curbs vanish, put the threshold back to 0.35 and commit. A working demo beats a tuned one.
- [ ] Record the **backup video**: one full successful run, ~90 s, on the phone.
- [ ] Deploy to Vercel, or write down the LAN HTTPS URL and confirm laptop and phone
      share the network the room actually has.
- [ ] Charge to 100%, bring the cable and the earbuds.
- [ ] Pack props: a water bottle (out of vocabulary), a printed page (for "read this").

## Ten minutes before

- [ ] Open the URL and grant camera + mic **once**, so no permission dialog appears on stage.
- [ ] Brightness up, Do Not Disturb on, rotation locked, volume ~70 %.
- [ ] Close every other app — thermal throttling costs frames per second.
- [ ] Speak one phrase to confirm audio.

## Run of show

### 0:00 — Open with one sentence

> "This is a wearable assistant for blind users. The phone watches the path and speaks
> obstacles, and you can ask it questions out loud. Two models do those two jobs, and
> I'll show you why it has to be two."

Don't explain the architecture yet. Show it first.

### 0:20 — Act 1 · The safety loop (the model I trained)

Hold the phone at chest height. Show the screen: live boxes, labels, `wasm`, `5 fps`.

> "Everything on this screen is my model, running inside the browser on this phone.
> Nothing leaves the device."

Walk slowly toward a chair or a door. **Let it speak — don't talk over it.**

Have someone stand in your path. Listen for: `person, ahead, close, go left`

**The moment that lands:** switch on Airplane Mode. Keep walking. It keeps narrating.

> "No network. That's why the safety model is on the device — a blind person doesn't
> stop walking when the signal drops."

### 3:00 — Act 2 · Open vocabulary (the cloud model)

Airplane Mode off. Hand the examiner something you never trained on, and invite it:

> "Pick up anything in the room."

Say: `vision, what is this?` — then wait. Let the 3–4 seconds happen; don't fill the silence.

> "About three and a half seconds. Fine for a question. Fatal for a step — which is
> exactly why the first model exists."

Then `vision, read this` on the printed page.

### 5:00 — Act 3 · The numbers

- Baseline COCO YOLO11n on my test split: **mAP50 0.26** across all 15 classes —
  exactly 0 on stairs, curb, pole, door, crosswalk and auto.
- Fine-tuned on 9,035 merged images: **0.667**. Door 0.95, stairs-up 0.92.
- Weakest: bicycle 0.45, pole 0.49, curb 0.50, crosswalk 0.52.
- Runtime: 5 fps, wasm backend, iPhone 13.

### 6:00 — Volunteer the failure before they find it

Show the camel photo.

> "My detector calls these dogs. It has 15 classes, so an unknown object gets mapped to
> the nearest one it knows. Adding classes doesn't fix that — the world isn't a finite
> list. It's the reason for the second tier: same photo, the cloud model described the
> herd correctly."

### 7:00 — Close on the trade-off, not on a feature

> "A nano model on a phone needs a small fixed vocabulary to stay fast and work offline,
> so I trained it on the 15 things that matter for safety. Open-ended recognition goes to
> a large cloud model where three seconds is acceptable. Different problems, different models."

## If it breaks

| Breaks | Say | Do |
|---|---|---|
| Model won't load | "Let me reload — it caches the model on first open." | Reload once, then the video |
| No sound | *nothing* | Silent switch, volume, re-tap start (iOS needs a gesture to unlock speech) |
| No camera | "Safari only allows the camera over HTTPS." | Use the cert URL, not the bare IP |
| It says something wrong | "Good — that's an out-of-vocabulary case. Watch what the second tier does with it." | Turn it into Act 2 |
| Everything | "I recorded a full run this morning." | Play the backup video and keep narrating |

**Never debug live for more than 20 seconds.** Switch to the video and keep your voice
level. They are grading your understanding, not your Wi-Fi.

## The three questions you will get

1. **"Why not just use Gemini for everything?"** — Latency, quota, offline, privacy, and
   it returns words rather than coordinates. The narrator needs `cx` and `h` to compute
   direction and time-to-contact.
2. **"Is this actually deep learning?"** — The long answer is already written in `viva-prep.md`.
3. **"How do you know how far away something is?"** — Box height scaled by the class's
   real-world size, and approach speed from the growth rate (time-to-contact). No depth sensor.

Full answers: [`docs/viva-prep.md`](./viva-prep.md)
