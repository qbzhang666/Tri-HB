One-time install
Open PowerShell in the project folder and install the Python dependencies:

cd "C:\Users\qianbinz\Monash Uni Enterprise Dropbox\Qianbing Zhang\00-GitHub\Tri-HB"
py -3.13 -m pip install -r requirements.txt
(Substitute py -3.13 with whichever Python launcher you have. The app needs Python ≥ 3.9.)

Launch
Default port (8501):

py -3.13 -m streamlit run tri_hb_integrated.py
A second instance on port 8502 (useful when port 8501 is busy) — either run the included launcher,

.\launch_8502.ps1
or invoke Streamlit directly:

py -3.13 -m streamlit run tri_hb_integrated.py --server.port 8502
Then open the URL Streamlit prints (http://localhost:8501 or :8502) in any modern browser.

Quick sanity checks
py -3.13 --version
py -3.13 -m streamlit --version