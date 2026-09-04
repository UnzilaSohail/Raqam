# Real form samples — drop them here

This is where you put **real scanned form photos** so Raqam can be measured and
tuned on genuine handwriting instead of synthetic data.

## Where / how to upload

On this machine, put files under a folder named for the form type:

```
D:\claudecode\data\real_forms\
    marksheet\
        labels.csv
        img_001.jpg
        img_002.jpg
        ...
    polio-tally\
        labels.csv
        ...
```

- One folder per form type (`marksheet`, `polio-tally`, `epi-tally`,
  `flood-registration`, `meter-reading`, `union-council-register`).
- Any image format: `.jpg` `.png` `.webp` `.pdf` `.tif`. Phone photos are fine.
- The images are **git-ignored** — they never get committed or uploaded anywhere.
- 20–50 images per form type is enough for a first read; more is better.

If you're on another computer or phone: zip the folder and share it, or drop it in
`Downloads` and tell me the path — I'll move it in.

## labels.csv format

One row per image: the filename, then the correct value a human would type.

```csv
filename,value
img_001.jpg,145072
img_002.jpg,145073
img_003.jpg,09821
```

- `value` = the digits exactly as they should end up in the data (no spaces).
- If a field was blank or unreadable, use an empty value or `SKIP`.
- Use **dummy or consented data only** — no real CNIC / patient data in a test set.

## What happens next

```bash
python -m raqam.evaluate --scans data/real_forms/marksheet
```

gives real accuracy numbers: digit error rate before and after review, auto-accept
rate, box-detection success. From there the segmentation and confidence threshold
get tuned for that specific template.
