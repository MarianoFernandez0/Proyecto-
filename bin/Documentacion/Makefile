#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# $Id: Makefile 900 2011-12-23 18:38:33Z fefo $
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# LaTeX sourcecode
MAIN_SRC = tesis
REF_SRC = referencias

all: $(MAIN_SRC).pdf

$(MAIN_SRC).pdf: $(MAIN_SRC).tex $(REF_SRC).bib
#	 introduction.tex abstract.tex acknowledgments.tex preface.tex
	pdflatex $(MAIN_SRC).tex

full: $(MAIN_SRC).pdf
	pdflatex $(MAIN_SRC).tex
	bibtex $(MAIN_SRC).aux
	pdflatex $(MAIN_SRC).tex
	pdflatex $(MAIN_SRC).tex


# delete the logs
clean:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.blg
	rm -f *.dvi
	rm -f *.glo
	rm -f *.idx
	rm -f *.ist
	rm -f *.lof
	rm -f *.log
	rm -f *.lot
	rm -f *.nav
	rm -f *.out
	rm -f *.ps
	rm -f *.rel
	rm -f *.snm
	rm -f *.toc




