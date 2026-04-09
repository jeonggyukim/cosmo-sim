NOTES = ic_sampling_review
UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	OPEN = evince
else ifeq ($(UNAME), Darwin)
	OPEN = open
endif
TIDY = rm -f *.aux *.log *.blg *.out *.toc

all : notes

notes : $(NOTES).tex
	pdflatex -draftmode -synctex=1 $^
	pdflatex $^
	$(OPEN) $(NOTES).pdf &
	$(TIDY)

clean :
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.synctex.gz
