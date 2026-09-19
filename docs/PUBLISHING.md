# Publish to GitHub

Review `git status` and the privacy exclusions in `RECOVERY_AUDIT.md`, then run:

```bash
git init -b main
git add .
git commit -m "Initial public release of PSMA PET-CT nnU-Net project"
gh repo create psma-pet-ct-segmentation-nnunet \
  --public \
  --source . \
  --remote origin \
  --push
```

If the GitHub CLI is not authenticated, run `gh auth login` first. The repository
archive does not contain raw scans, private case lists, checkpoints, prediction
volumes, IP addresses, hostnames, or container identifiers.

Before attaching model weights to a GitHub release, confirm that the training
data license and institutional policy permit redistribution. Checkpoints are
ignored by default.
