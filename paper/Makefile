filename=root

pdf:
	pdflatex --output-directory build ${filename}.tex
	bibtex build/${filename}.aux||true
	pdflatex --output-directory build ${filename}.tex
	pdflatex --output-directory build ${filename}.tex
	mv build/${filename}.pdf .

read:
	evince build/${filename}.pdf &

clean:
	rm -f build/${filename}.{ps,pdf,log,aux,out,dvi,bbl,blg}

