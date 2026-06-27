import re
from pathlib import Path

def main():
    path = Path("main.tex")
    text = path.read_text(encoding="utf-8")
    
    # 1. Update standard figure graphics options (width/height limiting and keepaspectratio)
    # Match standard \includegraphics[width=...]{...} but NOT definition at line 84 or minipages
    # We will use a regex to capture the options and the path.
    
    def repl_std(match):
        opt = match.group(1)
        filepath = match.group(2)
        # Skip definition at line 84 (already has height)
        if "keepaspectratio" in opt and "height" in opt:
            return match.group(0)
        # If it's a minipage image (width=\linewidth or width=\textwidth within minipage)
        if "linewidth" in opt or "linewidth" in opt:
            return f"\\includegraphics[width=\\linewidth,height=0.25\\textheight,keepaspectratio]{{{filepath}}}"
        
        # Standard image: reduce width slightly to avoid overfull warnings, set max height
        # e.g., width=0.85\textwidth
        width_val = "0.85\\textwidth"
        if "0.7" in opt or "0.8" in opt:
            width_val = "0.80\\textwidth"
        
        return f"\\includegraphics[width={width_val},height=0.38\\textheight,keepaspectratio]{{{filepath}}}"

    # Regex for standard graphics
    pattern = r"\\includegraphics\[([^\]]+)\]\{([^\}]+)\}"
    
    # We'll run the replacement
    new_text = re.sub(pattern, repl_std, text)
    
    # Let's check for any other spacing improvements we can make:
    # Add vertical spacing adjustment to float separation parameters or figure environments
    # For example, before \begin{document}, we can add:
    # \setlength{\floatsep}{10pt plus 2pt minus 2pt}
    # \setlength{\textfloatsep}{15pt plus 2pt minus 3pt}
    # \setlength{\intextsep}{12pt plus 2pt minus 2pt}
    
    spacing_cmds = """
% Spacing adjustments for floats/figures
\\setlength{\\floatsep}{8pt plus 2pt minus 2pt}
\\setlength{\\textfloatsep}{12pt plus 2pt minus 3pt}
\\setlength{\\intextsep}{10pt plus 2pt minus 2pt}
\\renewcommand{\\textfraction}{0.15}
\\renewcommand{\\topfraction}{0.85}
\\renewcommand{\\bottomfraction}{0.70}
\\renewcommand{\\floatpagefraction}{0.66}
"""
    
    if "% Spacing adjustments for floats/figures" not in new_text:
        # Insert spacing commands right before \begin{document}
        new_text = new_text.replace(r"\begin{document}", spacing_cmds + "\n\\begin{document}")

    path.write_text(new_text, encoding="utf-8")
    print("Image scaling and spacing adjustments applied to main.tex")

if __name__ == "__main__":
    main()
