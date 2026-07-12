# Shock and blast Streamlit deployment

Use `blasting/shock_blast_integrated.py` as the **Main file path** in
Streamlit Community Cloud. The `blasting/` directory is a self-contained
deployment bundle so the cloud app does not depend on Python import paths
outside that directory.

The deployment requires all of these repository files:

```text
blasting/shock_blast_integrated.py
blasting/requirements.txt
blasting/models/__init__.py
blasting/models/shock_blast.py
```

Commit the complete `blasting/` directory. Uploading the app file without its
`models/` subdirectory produces a `ModuleNotFoundError` during startup.

Test the cloud entry point locally from the repository root:

```powershell
py -3.13 -m pip install -r blasting/requirements.txt
py -3.13 -m streamlit run blasting/shock_blast_integrated.py
```

For ordinary local development, the equivalent direct command is:

```powershell
py -3.13 -m streamlit run shock_blast_integrated.py
```

The deployment app and model are synchronized copies of the canonical root
files. Run the deployment tests before publishing changes so the two copies do
not drift.
