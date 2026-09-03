# Raqam — Data-Handling Policy (v0.7)

One page. Applies to every Raqam deployment. Written to be attached to a pilot MOU.
Raqam is an **independent, review-assist tool — not an official record system.**

## 1. What Raqam touches

A photograph of one handwritten form field (digits), and the digit values read from it.
Depending on the form, those values may be part of sensitive records — CNIC numbers,
health/immunization data, beneficiary lists, exam results.

## 2. Default: nothing leaves the device

- Recognition, the local queue, and human review all run **on-device with zero connectivity**.
- No form image and no field value is transmitted anywhere unless an operator explicitly
  triggers **Sync**, and Sync is enabled for that deployment.
- There is no telemetry, no analytics, no third-party calls. The recognizer runs locally
  (NumPy on a server/Pi, or in-browser on a phone).

## 3. Source images are not retained

- For each digitized record Raqam stores a **SHA-256 fingerprint** of the source image
  (chain-of-custody), **not the image**.
- The operator may keep the photo on the device's normal camera roll; Raqam does not copy,
  upload, or manage it. Deployments handling CNIC/health data should instruct operators to
  delete source photos after the record is reviewed.

## 4. Sync (only if enabled per deployment)

- Opt-in per deployment, off by default.
- Transport: HTTPS to a server the partner controls. At rest: the partner's own encrypted storage.
- Synced payload = form name, field name, digit value, review status, image hash. **Not the image.**
- Every sync is logged (count, timestamp, device).

## 5. Storage on the device

- Local queue: SQLite (server) / IndexedDB (browser), on the operator's device.
- `ponytail:` field-level encryption + key management for shared devices is a known gap
  (tracked for the Android/PWA hardening phase). Until then: use Raqam on
  single-operator devices for sensitive forms, and rely on the device's own lock screen /
  full-disk encryption.

## 6. Human review is mandatory

- Any digit below the confidence threshold is flagged and **must** be confirmed by a person
  before the record is treated as data. Auto-accept is never set to 100%.
- The threshold is tuned per deployment against that deployment's risk tolerance.

## 7. Consent

- For health data: use the **partnering programme's existing consent process** for beneficiary
  data. Do not create a parallel one.
- For any deployment involving CNIC numbers: obtain an explicit legal read on NADRA
  data-handling expectations **before** the pilot.

## 8. Elections

- Raqam is **not** deployed for election result forms (Form 45/47) under this policy.
  Any such use requires separate legal and political review and an institutional partner
  (see the project plan, §07).

## 9. Open source

- The recognition engine and tooling are open-source and auditable. Partners may self-host
  and inspect every line that touches their data.

---
*Contact / revisions: track in the project repository. This policy is a starting point for a
pilot, not legal advice; have counsel review it against current Pakistani data-protection law
before signing.*
