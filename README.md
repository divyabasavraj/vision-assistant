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

## Testing on a phone over Wi-Fi

Phones only allow camera/mic on HTTPS, and Next's `--experimental-https` needs an admin password to
install its CA. Instead, make a certificate once (no password) and pass it in:

```bash
~/Library/Caches/mkcert/mkcert-v1.4.4-darwin-arm64 -key-file certificates/lan-key.pem -cert-file certificates/lan.pem $(ipconfig getifaddr en0) localhost 127.0.0.1
npm run dev -- --hostname 0.0.0.0 --experimental-https --experimental-https-key ./certificates/lan-key.pem --experimental-https-cert ./certificates/lan.pem
```

Open `https://<mac-ip>:3000` on the phone, accept the certificate warning once. `next.config.ts`
lists private LAN ranges in `allowedDevOrigins`; without it the page renders but nothing is interactive.
