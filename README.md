# tiny.ml

A small ML framework and compiler for OCaml, in the style of JAX and XLA.
Write numeric code once, then run it, differentiate it, or compile it.

**Status:** early. The first phase, the tracer (eval, autodiff and tracing
into an IR graph), works. IR passes and real compiler backends come next.

```ocaml
open Tracer

let f x = Ops.(-(sin x * scalar 2.) + x)

let () =
  let x = Ops.scalar 3. in
  print_endline (Ops.to_string (f x));                        (* 2.7177599838802657 *)
  print_endline (Ops.to_string (Ad.deriv f x));               (* 2.979984993200891 *)
  print_endline (Ops.to_string (Ad.deriv (Ad.deriv f) x));    (* 0.2822400161197344 *)
  print_endline (Ops.to_string (Compile.compile1 f x))        (* 2.7177599838802657 *)
```

## Features

- **Arrays:** float64 n-d arrays with NumPy-style broadcasting, elementwise
  math, reductions, shape ops and `dot`.
- **Autodiff (`Ad`):** forward mode (`jvp`, `deriv`, `jacfwd`) and reverse
  mode (`vjp`, `grad`, `jacrev`, `hessian`). Derivatives nest, and `f` can
  branch on its inputs.
- **Pytrees (`Tree`):** pass pairs, lists, arrays and records in and out.
- **Compile:** `Compile.jit` traces a function into an `Ir` graph. A
  compiled function is still a normal function, so you can differentiate it.
  `Ir.to_string` and `Ir.to_dot` show the graph.
- **Control flow (`Control`):** `cond` and `while_loop` branch on data and
  still work under `jit` and autodiff (`while_loop`: forward mode only).
- **Syntax (optional):** `ppx_tracer` lets you write `let%ops f x = -.(sin x *. 2.) +. x`
  and derive a `Tree` with `[@@deriving tree]`.
- **Extensible:** add primitives, transformations and backends from outside
  the library, using `Tracer.Expert`. See `examples/extending.ml`.

## Quick start

Requires OCaml 5.0 or later.

```sh
opam install . --deps-only --with-test --with-dev-setup
dune build
dune test                         # run the tests
dune exec tiny_ml                 # run every example
dune exec tiny_ml -- basics       # run one example
```

Use it from your own dune project:

```
(executable
 (name main)
 (libraries tiny_ml.tracer)
 (preprocess (pps tiny_ml.ppx_tracer)))   ; optional
```

## Examples

| Name | What it shows |
| --- | --- |
| `basics` | arrays, `deriv` and `grad`, Newton's method |
| `training` | fitting a small MLP to `sin x` |
| `jit` | the traced graph of a function and its gradient, `Control.cond` |
| `control` | `cond` and `while_loop` on data, under `jit` and autodiff |
| `extending` | a new primitive and a new interpreter, outside the library |

## Layout

```
lib/tracer/      the library (start with tracer.mli)
lib/ppx_tracer/  the optional syntax extension
examples/        the examples
bin/             the example runner
test/            tests
```

## License

MIT. See [LICENSE](LICENSE).
