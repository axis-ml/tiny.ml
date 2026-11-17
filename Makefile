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

.PHONY: doc
doc:
	$(DUNE) build @doc

.PHONY: clean
clean:
	$(DUNE) clean
