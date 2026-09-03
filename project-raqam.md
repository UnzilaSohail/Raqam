# Project Raqam

**Raqam (رقم)** — Urdu for "digit / figure / number"

*From a weekend "watch it think" neural-network demo to an offline handwriting-digitization toolkit for the paper forms that still run Pakistan — school marksheets, health tally sheets, relief registrations, meter readings, and civic records.*

Prepared: September 2026 · Status: Concept & roadmap, pre-pilot · Core engine: NumPy, offline, no cloud API

---

## 00. Summary

The original build is a genuinely good learning project: a from-scratch neural network with live training visualization, a "dreams" feature, and a hand-drawn digit predictor. Its Phase 5 — an offline, confidence-triaged form digitizer — is the part worth taking seriously. This plan keeps Phases 1–4 as the engine, and rebuilds Phases 5–6 into something scoped, funded, and safe enough to actually put in front of a school clerk, a health worker, or a district government office in Pakistan.

- **6 sectors it can serve** — education, health, disaster relief, local government, utilities, civic transparency
- **$0 marginal cost per form** — no cloud OCR bill, no per-request pricing; runs on a Raspberry Pi or a low-end Android phone
- **92–96% is where the MLP demo stops** — not accurate enough alone for anything consequential; this plan makes confidence triage the actual product

---

## 01. The Problem — Pakistan still runs on paper, in places the cloud doesn't reach

A large share of the country's routine record-keeping — the data that decides who got vaccinated, who passed an exam, who received flood relief, whose electricity bill is correct — is still written by hand on paper, in areas where connectivity is patchy, expensive cloud OCR is a non-starter, and the person filling the form may write digits in Urdu numerals as often as Western ones.

Six pillars, in order of how ready each one is for a pilot:

| Sector | What it is | Fit |
|---|---|---|
| **Health** | Polio & EPI tally sheets — Lady Health Workers and vaccination teams still record house-to-house immunization tallies on paper during campaigns, later compiled by hand. Well-documented part of Pakistan's polio programme. | Digit-only, high volume, forgiving of a human-review step. |
| **Education** | School & board marksheets — government schools and regional education boards (BISE Lahore, Karachi, Peshawar, and others) compile roll numbers and marks from handwritten sheets; delays and re-checks are a recurring complaint at result time. | Strong — printed roll-number boxes match the scanner pipeline exactly. |
| **Disaster relief** | Flood & camp registration — after the 2022 floods, aid agencies registered displaced households on paper in camps with no signal; incomplete registries were cited as a real gap in the response. | Strong, but needs a rugged offline-first app, not just a script. |
| **Local government** | Union council registers — birth, death and Nikah registration still runs through hand-filled registers before reaching NADRA's systems. | Moderate — needs alphanumeric (names), not just digits, for full value. |
| **Utilities** | DISCO meter readings — manually noted readings are a recurring source of inflated or disputed electricity bills, with dedicated complaint portals existing just to contest them. | Strong, and the only pillar with a plausible paying customer. |
| **Civic transparency** | Form 45 result sheets — polling-station result forms are the primary record election-transparency groups cross-check against consolidated results. Genuinely high-stakes, high-sensitivity. | Technically strong, politically the highest-risk pillar by far — see §07. |

---

## 02. Why the Obvious Fix Fails — why nobody just calls a cloud OCR API

- **Cost compounds at scale.** A per-request cloud OCR bill, priced in dollars, gets worse every time the rupee depreciates against it — exactly the wrong incentive for an NGO or school district digitizing millions of fields.
- **Connectivity is the actual bottleneck.** Flood camps, northern districts, and many rural union councils don't have the reliable link a cloud API assumes as a baseline.
- **Western-digit models silently fail on Urdu numerals.** Older form-fillers, and many rural respondents, write ۰۱۲۳۴۵۶۷۸۹ rather than 0123456789 — a model trained only on MNIST will misread or reject these outright.
- **The data is sensitive by default.** CNIC numbers, health records, and beneficiary lists are exactly the categories a school or NGO should not route through an unknown third-party cloud pipeline.
- **Pakistan's data-protection law is still in draft.** A comprehensive Personal Data Protection Act has been proposed for years without full enactment — a reason to build privacy-by-default now, not to wait for a law to force the issue.
- **100% automation is the wrong goal.** A marksheet, a vaccination record, or an election tally sheet is exactly where a silent wrong digit is worse than a flagged one — the product has to be the triage, not just the recognition.

---

## 03. Vision & Principles — what Raqam actually is

An open-source, offline-first toolkit that turns a photo of a handwritten form field into structured data, on hardware as modest as a Raspberry Pi or a low-end Android phone — with every low-confidence digit routed to a human, never silently guessed.

- **Offline by default** — recognition, storage and the review queue all work with zero connectivity; sync is opportunistic, never required.
- **Human-in-the-loop, always** — confidence triage isn't a fallback feature, it's the trust mechanism the whole product depends on.
- **Numerals as actually written** — Western digits and Urdu-Indic numerals (۰–۹) recognized natively, not transliterated after the fact.
- **Cheap hardware, not new hardware** — targets phones and machines field staff already carry; no procurement budget required to pilot.

---

## 04. Roadmap — from weekend build to field pilot

Phases 1–4 of the original plan (network core, live training visualization, "dreams," hand-drawing predictor) are unchanged — they're the engine and the demo, and still worth building first. Everything below replaces the original Phase 5–6 with a scoped path to a real deployment.

| Phase | Window | Goal | Exit criteria |
|---|---|---|---|
| 0–1 *(kept)* | Weekend | Build the core engine: NumPy MLP, live training viz, dream gallery, draw-and-predict canvas. | ≥92% test accuracy; live demo runs end to end. |
| 5 | Month 1 | Confidence-triage digitizer on MNIST-style digit cells; CSV/Excel export; one real paper form scanned end to end. | A stranger can scan a 6-digit field and get a correct, reviewable result with no help. |
| 6 | Months 1–2 | Full scanner pipeline: OpenCV grid/box detection, camera capture, annotated review image, running CSV log. | Works on a photo taken with a mid-range phone camera, not just a webcam. |
| 7 | Months 2–4 | Urdu-Indic numeral support; upgrade recognizer from MLP to a small CNN for production-grade accuracy. | >99% digit accuracy on held-out real-world scans, both numeral systems, before any auto-accept. |
| 8 | Months 3–5 | Offline-first Android app: on-device inference, local queue, opportunistic sync, review UI for field staff. | Runs fully offline on a sub-$100 Android phone; one field worker completes a real day's forms with it. |
| 9 | Months 4–8 | Paid pilot with one willing partner in one sector (education or health first — see §07 on why not elections first). | Signed pilot MOU; 500+ real forms digitized; measured error rate and time saved vs. manual entry. |
| 10 | Months 8–18 | Expand to a second sector, formalize the sustainability model, pursue grant and/or DISCO revenue. | Two active deployments; a funding or revenue path that outlives the founding team's free time. |

---

## 05. Architecture — the pipeline, and where it has to get harder

**Capture → Preprocess → Segment → Recognize → Triage → Export**

1. **Capture** — phone camera or scanner; works from a photo, not a flatbed.
2. **Preprocess** — deskew, adaptive threshold, contrast normalize.
3. **Segment** — detect the form's printed grid/boxes, crop each digit cell.
4. **Recognize** — CNN classifier, Western + Urdu-Indic numerals.
5. **Triage** — below-threshold digits flagged for a human, not guessed.
6. **Export** — CSV/Excel now; queued sync to a district dashboard later.

### What has to change from the hackathon version

- **Model** — the plain NumPy MLP (92–96% accuracy) is fine for a demo and unacceptable for a marksheet or a vaccination record. A compact CNN — still small enough to run on a phone — is a Phase-7 requirement, not a nice-to-have.
- **Numeral system** — MNIST only teaches Western digits. Urdu handwritten-numeral recognition is an active academic area with existing datasets and published deep-learning approaches to build on, rather than a from-scratch research problem.
- **Segmentation** — real forms aren't clean MNIST crops. The OpenCV contour-detection approach from the original Phase 6 is the right starting point, but needs tuning per form template (roll-number boxes look nothing like a meter-reading dial).
- **Storage** — a local, encrypted queue (SQLite is enough) that syncs opportunistically, so a field worker in a flood camp never loses a day's work to a dead signal.

---

## 06. Privacy & Trust — the data this touches is exactly the kind that shouldn't leak

CNIC numbers, health records, and beneficiary lists sit in the most sensitive categories a piece of software can handle. The design has to treat that as the default assumption, not an add-on.

> **Working rule:** No raw form image or extracted field leaves the device by default. Sync, when it happens, is opt-in per deployment, encrypted in transit and at rest, and logged. Every digitized record keeps a hash of its source image for chain-of-custody, without needing to store the image itself long-term.

- Pakistan's Personal Data Protection Bill has been under development for years without full enactment — a reason to build privacy-by-default now rather than assume a law will arrive to require it.
- For health data specifically, align with whatever consent process the partnering health programme already uses for beneficiary data — don't invent a parallel one.
- For any deployment touching CNIC numbers, get an explicit legal read on NADRA data-handling expectations before piloting, not after.

---

## 07. The Election Question — Form 45 is the most tempting pillar and the one to touch last

Polling-station result forms (Form 45) are compiled by presiding officers and are the primary record election-transparency organizations use to cross-check consolidated constituency results (Form 47). Discrepancies between the two have been a genuine, actively disputed issue in recent elections, including court directives ordering Form 45 images to be published. That makes it a real and important digitization target — and also the pillar where a well-intentioned tool can do the most damage if it's wrong, or is perceived as taking a side.

> **Recommendation:** Do not build for elections first, and do not position this as an official or binding tool at any point. If pursued at all, scope it narrowly as an independent civil-society aid for cross-checking already-published Form 45 images against reported constituency totals — never as a system that produces or certifies results — and only after a specific legal and political review, ideally alongside an established election-monitoring body rather than as a standalone product.

Concretely: build and prove the pipeline in education or health first, where a wrong digit costs a re-check rather than a headline. Revisit elections, if at all, once the recognition accuracy, the triage workflow, and an institutional partner are all already proven elsewhere.

---

## 08. Partners & Funding — who to actually call

| Track | Who | Why them |
|---|---|---|
| Seed funding | Ignite – National Technology Fund (under the Ministry of IT & Telecom), via its Pakistan Startup Fund or incubation centers | Pakistan's dedicated public tech-fund vehicle; existing track record funding early-stage local tech. |
| Technical partners | University CS/EE departments (FAST-NUCES, NUST, ITU Punjab, LUMS) | Final-year-project pipeline for the CNN/Urdu-numeral work; low-cost, high-motivation talent. |
| Health pilot | Provincial Emergency Operations Centres / EPI programme, with UNICEF's Pakistan innovation work as a possible technical ally | Already runs the paper tally-sheet workflow this tool targets, at national scale. |
| Education pilot | One regional education board (BISE) or a district education office, starting small | Bounded, well-defined form templates (roll-number boxes); low political sensitivity. |
| Election-transparency track | Established local election-monitoring networks — approached only after §07's caution is satisfied | Already doing this cross-checking manually; wouldn't need convincing of the value, only of the risk controls. |
| Commercial revenue | A single DISCO's billing department, for meter-reading digitization | The one pillar with a direct, quantifiable cost saving a utility would actually pay for. |

---

## 09. Sustainability — how this outlives the first grant

An open-core model: the offline recognition engine and CLI tooling stay open-source, free for schools, NGOs, and government pilots — that's the trust and adoption layer. A managed, paid tier (hosted dashboards, DISCO billing integration, support contracts) funds the team, so the project isn't permanently dependent on the next grant cycle.

- **Free tier** — open-source engine + mobile app, self-hosted, for schools/NGOs/government.
- **Grant-funded** — health and disaster-relief pilots, funded through Ignite and donor/NGO partners.
- **Commercial** — DISCO meter-reading contract cross-subsidizes the free-tier work.

---

## 10. Success Metrics — what "it worked" has to mean

| Metric | Manual baseline | Pilot target |
|---|---|---|
| Time per 6-digit field | ~20–30 sec (manual re-entry) | <5 sec + review only when flagged |
| Digit-level error rate reaching final data | Varies, rarely measured | <0.5% after human review |
| Share auto-accepted (no human touch) | 0% (all manual) | >80%, tuned per deployment's risk tolerance |
| Cost per 1,000 fields vs. cloud OCR | $1.50 (typical cloud OCR) | ~$0 marginal, after one-time setup |
| Works fully offline | n/a | 100% of the time, by design |

---

## 11. Risks — what could go wrong, and the guardrail for each

| Risk | Severity | Mitigation |
|---|---|---|
| A confidently wrong digit reaches final data unflagged | High | Conservative confidence threshold; dual-entry spot-checks during pilot; never ship auto-accept at 100%. |
| Tool used or perceived as an official/authoritative source in a dispute (esp. elections) | High | Explicit non-goals in every deployment agreement; elections pillar deferred and legally reviewed first (§07). |
| Sensitive data (CNIC, health) mishandled or leaked | High | On-device by default, encrypted storage, no raw-image retention beyond a hash, partner-aligned consent. |
| Low trust from field staff (fear of being replaced or blamed for errors) | Medium | Position explicitly as a review-assist tool; involve field staff in pilot design, not just rollout. |
| Funding runs out before commercial track matures | Medium | Keep the education/health pilots small and grant-scoped; don't over-hire ahead of revenue. |

---

## 12. Team & Budget — what a first pilot actually costs

**Core team (part-time is fine to start)**
- 1 ML/backend engineer — recognition model, pipeline
- 1 mobile/frontend engineer — Android app, review UI
- 1 field coordinator — the partner relationship, pilot logistics, training
- Fractional legal/privacy review — one-time, before any pilot with sensitive data

**Rough 6-month pilot budget**

| Item | Cost |
|---|---|
| Engineering (2 people, part-time) | ≈ PKR 3.0–4.5M |
| Field coordination + training | ≈ PKR 0.8–1.2M |
| Devices (phones/Pi for pilot sites) | ≈ PKR 0.3–0.6M |
| Legal/privacy review | ≈ PKR 0.2–0.4M |
| **Total** | **≈ PKR 4.3–6.7M** |

*Rough planning figures for a single-sector 6-month pilot in 2026 terms — validate against actual local salaries and partner in-kind contributions before using this as a funding ask.*

---

## 13. Next 90 Days

1. Finish the Phase 1–4 engine and demo (weekend build, as originally scoped) — it's the credibility artifact for every conversation that follows.
2. Pick one sector to pilot first — education is the lowest-risk starting point given bounded form templates and low political sensitivity — and identify one specific willing partner (a single school board or district office), not a national rollout.
3. Start the CNN + Urdu-numeral upgrade in parallel; the MLP demo is not the model that ships to a pilot.
4. Draft a one-page data-handling policy (on-device default, no raw-image retention, consent alignment) before any conversation involves real people's data.
5. Prepare an Ignite seed-funding application, scoped to the single-sector pilot above, not the full six-pillar vision.

---

### Sources

Pakistan-specific context in this plan drew on: [Ignite – National Technology Fund](https://en.wikipedia.org/wiki/Ignite_National_Technology_Fund), [ignite.org.pk](https://ignite.org.pk/), [Pakistan data protection law status — DLA Piper](https://www.dlapiperdataprotection.com/index.html?t=about&c=PK), [FAFEN — Form 45 explainer](https://fafen.org/explainer-what-is-form-45-and-why-does-it-matter-in-an-election/), [2024 election Form 45/47 discrepancy allegations](https://en.wikipedia.org/wiki/Allegations_of_rigging_in_the_2024_Pakistani_general_election), [handwritten Urdu numeral recognition research](https://doi.org/10.3390/app13031624), [polio vaccination tally sheets](https://polioeradication.org/news/vaccines-vaccinators-and-tally-sheets/), [2022 flood registration gaps](https://en.emranews.com/162244/), and Pakistani electricity-billing complaint/dispute portals. Figures marked as estimates are planning approximations, not verified costings.
