"""Génère le rapport en PDF.

rapport.md -> rapport.html (markdown + pygments) -> PDF (Edge, Chrome ou Chromium
en mode headless), puis ajout des numéros de page et des métadonnées (PyMuPDF).

    python rapport/build_pdf.py

Pour choisir un autre navigateur Chromium, donner son chemin dans PDF_BROWSER :
    PowerShell : $env:PDF_BROWSER = "C:\\chemin\\vers\\chrome.exe"
    cmd        : set PDF_BROWSER=C:\\chemin\\vers\\chrome.exe
    bash/zsh   : export PDF_BROWSER="/chemin/vers/chrome"
"""

import datetime
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import markdown
import pymupdf
from pygments.formatters import HtmlFormatter

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "rapport.md"
HTML_OUT = HERE / "rapport.html"
PDF_OUT = HERE / "RUSESCU_BAH_SEDJAI_rapport_TP_Snake.pdf"

TITLE = "TP MDP : Snake"
AUTHORS = "RUSESCU Daniel Alexandru, BAH Sulaiman, SEDJAI Fethi"
FOOTER = "TP MDP : Snake - RUSESCU, BAH, SEDJAI"

MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
    "août", "septembre", "octobre", "novembre", "décembre",
]

BROWSER_ENV = "PDF_BROWSER"   # chemin d'un navigateur Chromium choisi à la main
BROWSERS = [
    # Windows
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    # macOS
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    # Linux, ou tout navigateur présent dans le PATH
    "msedge", "microsoft-edge", "google-chrome", "google-chrome-stable", "chromium",
    "chromium-browser", "chrome",
]

CSS = """
@page { size: A4; margin: 2cm 2cm 2.4cm 2cm; }
html {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.45; color: #1f2328;
}
body { margin: 0; }
a { color: #1d4e89; text-decoration: none; }

/* Page de garde */
.cover {
  height: 252mm; display: flex; flex-direction: column;
  justify-content: space-between; text-align: center; break-after: page;
}
.cover-school { font-size: 12pt; letter-spacing: 0.08em; color: #555; margin-top: 8mm; }
.cover-course { font-size: 30pt; font-weight: 700; line-height: 1.15; color: #1d3557; }
.cover-title { font-size: 22pt; font-weight: 600; margin-top: 14mm; color: #1d4e89; }
.cover-subtitle { font-size: 13pt; margin-top: 3mm; color: #333; }
.cover-scope { font-size: 11pt; margin-top: 8mm; color: #555; font-style: italic; }
.cover-authors { font-size: 13pt; font-weight: 600; line-height: 1.7; }
.cover-meta { font-size: 11pt; line-height: 1.7; }
.cover-work {
  text-align: left; border: 1px solid #c9d3e0; background: #f4f7fb;
  border-radius: 6px; padding: 10px 16px; font-size: 10pt; margin-bottom: 4mm;
}
.cover-work-title { font-weight: 700; color: #1d3557; margin-bottom: 2px; }
.cover-work p { margin: 4px 0; }
.cover-repo { color: #444; }

/* Sommaire */
.toc-title { font-size: 17pt; font-weight: 700; color: #1d3557; margin-bottom: 6mm; }
.toc { break-after: page; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc > ul > li { margin: 8px 0 2px; font-weight: 600; }
.toc ul ul { padding-left: 18px; font-weight: 400; }
.toc ul ul li { margin: 2px 0; }

/* Contenu */
h1, h2, h3 { color: #1d3557; break-after: avoid; }
h2 {
  font-size: 17pt; margin: 0 0 10px; padding-bottom: 4px;
  border-bottom: 2px solid #1d3557; break-before: page;
}
h3 { font-size: 12.5pt; margin: 1.5em 0 0.3em; }
h3 + p em:only-child { color: #57606a; }
h3 + p { break-after: avoid; }
p { margin: 0.45em 0; }
hr { display: none; }
code { font-family: Consolas, "Cascadia Mono", "Courier New", monospace; }
p code, li code, td code {
  background: #eff1f3; padding: 0 3px; border-radius: 3px; font-size: 0.92em;
}
.codehilite {
  background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 4px;
  padding: 6px 10px; margin: 0.6em 0; break-inside: avoid;
}
.codehilite pre { margin: 0; font-size: 8.6pt; line-height: 1.35; white-space: pre-wrap; }
table { border-collapse: collapse; margin: 0.6em 0; break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f0f3f7; }
img { max-width: 100%; break-inside: avoid; }
.fig-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; margin: 0.6em 0; }
.fig-grid figure { margin: 0; break-inside: avoid; }
.fig-grid img { width: 100%; }
.fig-grid figcaption { font-size: 8.5pt; color: #57606a; text-align: center; }
.formula {
  text-align: center; font-size: 12pt; margin: 0.8em 0; padding: 8px;
  background: #f4f7fb; border: 1px solid #c9d3e0; border-radius: 4px;
  font-family: "Cambria Math", Cambria, "Times New Roman", serif;
}
"""


def french_date(day):
    day_str = "1er" if day.day == 1 else str(day.day)
    return f"{day_str} {MONTHS[day.month - 1]} {day.year}"


def find_browser():
    chosen = os.environ.get(BROWSER_ENV, "").strip().strip('"').strip("'")
    if chosen:
        # Le fichier doit exister (et être exécutable avec Python ≥ 3.12 sous Windows) ;
        # sinon le lancement échoue plus loin avec un message clair.
        found = shutil.which(chosen)
        if found:
            return found
        sys.exit(f"{BROWSER_ENV} = {chosen} : navigateur introuvable ou non exécutable.")
    for candidate in BROWSERS:
        path = pathlib.Path(candidate)
        if path.is_file():
            return str(path)
        found = shutil.which(candidate)
        if found:
            return found
    sys.exit("Aucun navigateur Edge/Chrome/Chromium trouvé pour l'impression PDF. "
             f"Indiquez son chemin dans la variable d'environnement {BROWSER_ENV}.")


def build_html():
    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace("{{DATE}}", french_date(datetime.date.today()))
    body = markdown.markdown(
        text,
        extensions=[
            "fenced_code", "codehilite", "tables", "toc",
            "md_in_html", "attr_list", "sane_lists",
        ],
        extension_configs={
            "codehilite": {"guess_lang": False, "css_class": "codehilite"},
            "toc": {"toc_depth": "2-3"},
        },
    )
    pygments_css = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    html = (
        "<!doctype html>\n<html lang=\"fr\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{TITLE}</title>\n<style>{CSS}\n{pygments_css}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    HTML_OUT.write_text(html, encoding="utf-8")


def print_pdf(raw_pdf):
    browser = find_browser()
    # Dossiers temporaires sous rapport/ plutôt que /tmp : Chromium en snap (Ubuntu)
    # n'a pas accès au /tmp du système.
    problem = None
    with tempfile.TemporaryDirectory(dir=HERE) as profile:
        try:
            subprocess.run(
                [
                    browser, "--headless=new", "--disable-gpu", "--no-first-run",
                    "--no-pdf-header-footer", "--print-to-pdf-no-header",
                    f"--user-data-dir={profile}", f"--print-to-pdf={raw_pdf}",
                    HTML_OUT.as_uri(),
                ],
                check=True, timeout=180, capture_output=True, encoding="utf-8", errors="replace",
            )
        except subprocess.CalledProcessError as error:
            last_lines = (error.stderr or "").strip().splitlines()[-3:]
            problem = "\n".join([f"le navigateur s'est arrêté avec le code {error.returncode}",
                                 *last_lines])
        except subprocess.TimeoutExpired:
            problem = "le navigateur n'a pas répondu en 180 s"
        except OSError as error:
            problem = str(error)
    if problem:   # message donné après le nettoyage des dossiers temporaires
        sys.exit(f"Impression PDF impossible avec {browser} : {problem}")
    if not pathlib.Path(raw_pdf).is_file():
        sys.exit("Le navigateur n'a pas produit de PDF.")


def finalize_pdf(raw_pdf):
    """Ajoute les pieds de page et les métadonnées ; renvoie (octets du PDF, nombre de pages)."""
    with pymupdf.open(raw_pdf) as doc:
        n_pages = doc.page_count
        for index, page in enumerate(doc):
            if index == 0:  # pas de pied de page sur la page de garde
                continue
            width, height = page.rect.width, page.rect.height
            box = pymupdf.Rect(57, height - 42, width - 57, height - 26)
            style = {"fontsize": 8, "fontname": "helv", "color": (0.45, 0.45, 0.45)}
            page.insert_textbox(box, FOOTER, align=pymupdf.TEXT_ALIGN_LEFT, **style)
            page.insert_textbox(box, f"{index + 1} / {n_pages}",
                                align=pymupdf.TEXT_ALIGN_RIGHT, **style)
        doc.set_metadata({"title": TITLE, "author": AUTHORS, "subject": "Rapport de TP"})
        data = doc.tobytes(garbage=3, deflate=True)
    return data, n_pages


def main():
    build_html()
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        raw_pdf = str(pathlib.Path(tmp) / "raw.pdf")
        print_pdf(raw_pdf)
        data, n_pages = finalize_pdf(raw_pdf)
    # Écriture après le nettoyage : un PDF ouvert dans une visionneuse fait échouer
    # l'ouverture sans tronquer le fichier existant.
    try:
        PDF_OUT.write_bytes(data)
    except OSError as error:
        sys.exit(f"Impossible d'écrire {PDF_OUT} ({error.strerror}). Fermez le PDF s'il est ouvert.")
    print(f"{PDF_OUT} ({n_pages} pages)")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):   # accents lisibles même si la sortie est redirigée
        if hasattr(stream, "reconfigure"):    # absent sous IDLE, None sans console (pythonw)
            stream.reconfigure(encoding="utf-8")
    main()
