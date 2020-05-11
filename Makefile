help:
	@echo  "Development makefile"
	@echo
	@echo  "usage: make <target>"
	@echo  "Targets:"
	@echo  "	lint		Reports all linter violations"

lint:
	pylint --rcfile .pylintrc -j 4 ./**/*.py
