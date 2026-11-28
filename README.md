# tiny.ml

A small JAX-style ML framework for OCaml: write numeric code once, then run,
differentiate, batch or compile it.

```ocaml
open Tracer

let f x = Ops.(-(sin x * scalar 2.) + x)

let () =
  let x = Ops.scalar 3. in
  print_endline (Ops.to_string (f x));                        (* 2.7177599838802657 *)
  print_endline (Ops.to_string (Ad.deriv f x));               (* 2.979984993200891 *)
  print_endline (Ops.to_string (Ad.deriv (Ad.deriv f) x));    (* 0.2822400161197344 *)
  print_endline (Ops.to_string (Vmap.vmap (Ad.grad f) (Ops.array [| 0.; 3. |])));
                                                     (* [-1.0 2.979984993200891] *)
  print_endline (Ops.to_string (Compile.compile ~backend:Compile.jit f x))
                                                     (* 2.7177599838802657 *)
```

## Features

- **`Ops`**: float64 n-d arrays, NumPy broadcasting, reductions, shape ops, `dot`.
- **`Ad`**: `jvp`, `deriv`, `jacfwd`, `vjp`, `grad`, `jacrev`, `hessian`.
  Reverse mode is linearize + transpose. Nests.
- **`Vmap`**: `vmap` with per-leaf `in_axes` and `out_axis`.
- **`Tree`**: pytrees; every transformation has a `_tree` form.
- **`Ir`, `Compile`**: trace to a graph (`to_string`, `to_dot`); `jit`
  caches it as one `Call` op.
- **`Control`**: `cond` and `while_loop` on data (`while_loop`: forward mode only).
- **Errors**: `Tracer.Error`, or a `result` via `Tracer.catch`.
- **`ppx_tracer`** (optional): `let%ops` and `[@@deriving tree]`.
- **`Tracer.Expert`**: new primitives, interpreters and rules from outside
  the library; `Define.unary`/`binary` for elementwise ops in one call.

All transformations compose: `jit`, `grad`, `jvp` and `vmap` nest freely.

**Status:** early. Next: `Ir` passes (DCE, CSE) and real backends. Missing:
`dot_general`, dtypes beyond float64, `scan`, `custom_jvp`/`custom_vjp`, PRNG.

## Quick start

OCaml 5.0+.

```sh
opam install . --deps-only --with-test --with-dev-setup
dune build && dune test
dune exec tiny_ml                 # all examples
dune exec tiny_ml -- basics       # one: basics training jit control extending
```

```
(executable
 (name main)
 (libraries tiny_ml.tracer)
 (preprocess (pps tiny_ml.ppx_tracer)))   ; optional
```

## For JAX readers

Code lives in `lib/tracer/`; read the `.mli` files in this order.

| File | Modules | JAX |
| --- | --- | --- |
| `tracer.mli` | public API | `jax` |
| `common.mli` | `Ndarray`, `Aval` | `ShapedArray` |
| `lang.mli` | `Prim`, `Interp`, `Builtin`, `Ops`, `Tree` | primitives, `bind`, trace stack, `lax`/`jnp`, pytrees |
| `autodiff.mli` | `Jvp`, `Partial_eval`, `Transpose`, `Vjp`, `Derivatives`, `Ad` | `ad.py`, `partial_eval.py`, `defjvp` |
| `staging.mli` | `Ir`, `Compile`, `Control` | jaxpr, `pjit`, `lax.cond`/`while_loop` |
| `batching.mli` | `Batching`, `Vmap` | `batching.py`, `vmap` |

Then `examples/extending.ml` for a new primitive and interpreter.

## License

MIT. See [LICENSE](LICENSE).
