# Provenance and preparation checks

Source archive: `coretemp_sim_v14_final_rev2.zip`.
Release: v14 final, revision 2 (cross-seed stability criterion and corrected perfusion paired numbers).

The five Python files, all saved outputs, and the original report are preserved byte for byte. The original README is now `docs/V14_STUDY_REPORT.md`; the root README is a new project overview. No hardware, human data, older archives, or presentation files were added.

Checks performed during repository preparation:

- All 19 original SHA-256 entries passed, mapping the manifest's `README.md` entry to `docs/V14_STUDY_REPORT.md`.
- All five Python source files passed syntax parsing.
- All JSON files parsed successfully.
- Full simulations were not rerun; saved results are from the supplied archive.

The original manifest is retained unchanged as a historical record. After rerunning scripts, its output checksums may no longer match. Dependencies are unpinned; the original environment versions are recorded in the manifest. No license was added.
