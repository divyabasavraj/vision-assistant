# Vision Assistant

A web app for blind and low-vision users. A YOLO11n detector, fine-tuned by me on a navigation dataset, runs **inside the phone browser** and speaks obstacles in real time ("person, left, close"). A cloud vision-language model answers spoken questions on demand ("vision, what's this?").

Live URL: _(added on Day 3)_

## Run

```bash
npm install
npm run dev        # http://localhost:3000  (camera/mic need HTTPS on a phone — see plan Task 5)
npm test           # unit tests (Vitest)
```

Training scripts live in `training/` and use a Python 3.12 venv:

```bash
source training/.venv/bin/activate
pytest training/tests -q
```

## Documents

- Design spec: `docs/superpowers/specs/2026-09-11-vision-assistant-design.md`
- Build plan: `docs/superpowers/plans/2026-09-11-vision-assistant-build.md`
- Daily journal: `docs/journal/`
- Viva preparation: `docs/viva-prep.md`
