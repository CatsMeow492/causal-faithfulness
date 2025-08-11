PDF=main.pdf
TEX=main.tex

all: $(PDF)

$(PDF): $(TEX)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	cp build/main.pdf ./main.pdf

clean:
	latexmk -C -outdir=build
	rm -rf build

.PHONY: all clean

