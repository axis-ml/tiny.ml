DUNE ?= opam exec -- dune
EXAMPLES := basics training jit extending

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "targets:"
	@echo "  deps       install deps, test deps and dev tools"
	@echo "  build      build everything"
	@echo "  examples   run every example"
	@echo "  list       list the examples"
	@echo "  <name>     run one example: $(EXAMPLES)"
	@echo "  test       run the tests"
	@echo "  promote    accept snapshot changes"
	@echo "  lint       check formatting"
	@echo "  fmt        format the code in place"
	@echo "  ci         lint, build, test"
	@echo "  watch      rebuild on file changes"
	@echo "  doc        build the API docs"
	@echo "  doc-open   build the API docs and open them in a browser"
	@echo "  doc-serve  build the API docs and serve them on localhost:8000"
	@echo "  clean      remove _build"

.PHONY: deps
deps:
	opam install . --deps-only --with-test --with-dev-setup

.PHONY: build
build:
	$(DUNE) build

.PHONY: examples
examples:
	$(DUNE) exec tiny_ml

.PHONY: list
list:
	$(DUNE) exec tiny_ml -- --list

.PHONY: $(EXAMPLES)
$(EXAMPLES):
	$(DUNE) exec tiny_ml -- $@

.PHONY: test
test:
	$(DUNE) test

.PHONY: promote
promote:
	$(DUNE) promote

.PHONY: lint
lint:
	$(DUNE) build @fmt

.PHONY: fmt
fmt:
	$(DUNE) fmt

.PHONY: ci
ci: lint build test

.PHONY: watch
watch:
	$(DUNE) build --watch

DOC_HTML := _build/default/_doc/_html

.PHONY: doc
doc:
	$(DUNE) build @doc
	@echo "docs: $(DOC_HTML)/tiny_ml/index.html"

.PHONY: doc-open
doc-open: doc
	@xdg-open $(DOC_HTML)/tiny_ml/index.html 2>/dev/null \
	  || open $(DOC_HTML)/tiny_ml/index.html

.PHONY: doc-serve
doc-serve: doc
	@echo "http://localhost:8000/tiny_ml/"
	python3 -m http.server 8000 --directory $(DOC_HTML)

.PHONY: clean
clean:
	$(DUNE) clean
