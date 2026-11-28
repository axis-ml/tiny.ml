(lang dune 3.24)

(name tiny_ml)

(generate_opam_files true)

(formatting
 (enabled_for ocaml dune))

(source
 (github axis-ml/tiny.ml))

(authors "Quoc Nguyen" "tu-hm")

(maintainers "Quoc Nguyen" "tu-hm")

(license MIT)

(package
 (name tiny_ml)
 (synopsis "A tiny JAX-style tracer in OCaml")
 (description
  "Write numeric code once in OCaml, then evaluate it or differentiate it with forward-mode autodiff.")
 (depends
  (ocaml (>= 5.0))
  (ppxlib (and (>= 0.35) (< 0.36)))
  (ocamlformat
   (and
    :with-dev-setup
    (= 0.29.0)))
  (ocaml-lsp-server :with-dev-setup)
  (odoc :with-doc)))
