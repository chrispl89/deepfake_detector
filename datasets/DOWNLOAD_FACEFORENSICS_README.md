# FaceForensics++ Download Script

**IMPORTANT:** The official FaceForensics++ download script is **NOT included** in this repository due to licensing restrictions.

## Why Not Included?

The `download_faceforensics.py` script is proprietary software owned by the FaceForensics++ team and subject to their Terms of Service. We cannot redistribute it without explicit permission.

## How to Obtain the Official Script

### ⚠️ IMPORTANT: Access Requires Permission

**You CANNOT simply download the script from GitHub!**

FaceForensics++ requires you to request access first.

### Step 1: Fill Out Access Request Form

**Go to the official Google Form:**
```
https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
```

**What you need to provide:**
- Your name and email
- Your affiliation (university/organization)
- Intended use (research/education)
- Agreement to Terms of Service

### Step 2: Wait for Approval

- You will receive an email response (usually within a week)
- The email will contain the download link to the script
- If you don't hear back, check your spam folder

### Step 3: Download the Script

Once you receive the email with the download link:

1. Click the link provided in the email
2. Download the script
3. Save it as `datasets/download_faceforensics.py` in this repository

**Note:** The script link is private and personalized. Do not share it publicly.

### Step 4: Verify Download

After downloading, verify the file exists:
```bash
ls -la datasets/download_faceforensics.py
```

## Usage After Download

Once you have the official script, you can use it:

### Direct Usage:
```bash
python datasets/download_faceforensics.py ./data/datasets/faceforensics \
  --dataset Deepfakes \
  --compression c23 \
  --num_videos 10
```

### Through Our API:
```python
from datasets import download_faceforensicspp
result = download_faceforensicspp()
```

## Requirements

The script requires:
```bash
pip install tqdm
```

## Terms of Service

By downloading and using FaceForensics++ data, you agree to:
- **Research/Educational use only**
- **No commercial use**
- **No redistribution without permission**
- **Cite the original paper**

**Full ToS:** http://kaldir.vc.in.tum.de/faceforensics/

## Citation

If you use FaceForensics++, please cite:

```bibtex
@inproceedings{rossler2019faceforensicspp,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={R{\"o}ssler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={ICCV},
  year={2019}
}
```

## Troubleshooting

### "File not found" error when running our integration

**Solution:** Download the official script first (see steps above).

### "Permission denied" or "Terms of Service" prompt

**Solution:** This is normal. You must accept the ToS to proceed.

### Alternative: Manual Download

If the script doesn't work, you can also:
1. Visit: http://kaldir.vc.in.tum.de/faceforensics/
2. Follow their official download instructions
3. Manually download datasets

## Support

For issues with the FaceForensics++ script or data:
- **Official repo:** https://github.com/ondyari/FaceForensics
- **Issues:** https://github.com/ondyari/FaceForensics/issues
- **Website:** http://kaldir.vc.in.tum.de/faceforensics/

For issues with our deepfake detection system integration:
- See our main README.md
- Check FACEFORENSICS_INTEGRATION.md

---

**Note:** This file is a placeholder. The actual download script must be obtained from the official FaceForensics++ repository.
