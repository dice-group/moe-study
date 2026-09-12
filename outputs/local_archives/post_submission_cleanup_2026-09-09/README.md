# Post-submission cleanup: 2026-09-09

Verified snapshot of uncommitted tracked and untracked files in the main repository and both submodules. Ignored outputs/checkpoints are not copied or removed.

manifest.json lists files selected for staging and archive-only files, with SHA-256 hashes or symlink targets. Separate staged/unstaged patches preserve the previous edits.

To restore an archive-only file, extract its exact repository-relative path from uncommitted_snapshot.tar.gz into the repository (check for newer files first). Git history alone cannot recover these formerly untracked files.
