Code scanning outputs live here. Each run creates a timestamped subfolder with raw reports and a `summary.md`.

How to run (Windows PowerShell):

1) One-time prerequisites:
- Install Node.js and npm
- Install Rust (rustup) so `cargo` is available
- Install Python 3

2) Run scans from repo root:

```powershell
pwsh -File scripts/run-codescan.ps1 -InstallTools
```

Options:
- `-InstallTools`: attempt to install Rust cargo tools (cargo-audit, cargo-outdated, cargo-udeps)
- `-SkipNode`, `-SkipRust`, `-SkipPython`: skip specific language scans
- `-OutputSubdir <name>`: use a custom subfolder under `codescan/`

Reports generated:
- Node: npm audit, npm-check-updates, depcheck, ts-unused-exports, unimported, madge (circular)
- Rust: cargo-audit, cargo-outdated, cargo-udeps, clippy
- Python: pip-audit, bandit, ruff, vulture
- Duplication: jscpd (html, json, markdown)

Quick access:
- Latest run summary: `codescan/latest.md`
- Full run summary: `codescan/<timestamp>/summary.md`


