open Tracer

let s = Ops.scalar
let branchy x = if Float.compare (Ops.to_float x) 0. > 0 then Ops.sin x else Ops.cos x

let () =
  Check.value "grad branch, x = 1" ~expected:(s (cos 1.)) (Ad.grad1 branchy (s 1.));
  Check.value
    "grad branch, x = -1"
    ~expected:(s (-.sin (-1.)))
    (Ad.grad1 branchy (s (-1.)));
  Check.value "deriv branch, x = 1" ~expected:(s (cos 1.)) (Ad.deriv branchy (s 1.));
  Check.value
    "deriv branch, x = -1"
    ~expected:(s (-.sin (-1.)))
    (Ad.deriv branchy (s (-1.)));
  Check.value
    "grad (grad branch), x = 1"
    ~expected:(s (-.sin 1.))
    (Ad.grad1 (Ad.grad1 branchy) (s 1.));
  Check.value
    "grad (grad branch), x = -1"
    ~expected:(s (-.cos (-1.)))
    (Ad.grad1 (Ad.grad1 branchy) (s (-1.)));
  Check.value
    "deriv (grad branch), x = -1"
    ~expected:(s (-.cos (-1.)))
    (Ad.deriv (Ad.grad1 branchy) (s (-1.)));
  Check.value
    "grad (deriv branch), x = 1"
    ~expected:(s (-.sin 1.))
    (Ad.grad1 (Ad.deriv branchy) (s 1.))
;;

let () =
  let grow x =
    let rec go x =
      if Float.compare (Ops.to_float x) 10. > 0 then x else go Ops.(x * scalar 2.)
    in
    go x
  in
  Check.value
    "grad of a data-dependent loop, x = 3"
    ~expected:(s 4.)
    (Ad.grad1 grow (s 3.));
  Check.value
    "deriv of a data-dependent loop, x = 0.5"
    ~expected:(s 32.)
    (Ad.deriv grow (s 0.5))
;;

let () =
  let f x =
    let a = Ops.to_array x in
    Check.bool "to_array under grad" (a = [| 1.; 2. |]);
    Check.bool "allclose under grad" (Ops.allclose x (Ops.array [| 1.; 2. |]));
    Ops.reduce_sum (Ops.sin x)
  in
  Check.value
    "grad after reads"
    ~expected:(Ops.array [| cos 1.; cos 2. |])
    (Ad.grad Tree.value f (Ops.array [| 1.; 2. |]))
;;

let () =
  Check.value
    "eager compile branches"
    ~expected:(s (sin 1.))
    (Compile.compile1 ~backend:Compile.eager branchy (s 1.));
  Check.raises ~matches:Check.is_concretization "jit cannot branch on data" (fun () ->
    Compile.compile1 ~backend:Compile.jit branchy (s 1.));
  Check.raises
    ~matches:Check.is_concretization
    "Ir.make1 cannot branch on data"
    (fun () -> Ir.make1 branchy (s 1.));
  Check.value
    "trace inside deriv reads the outer primal"
    ~expected:(s 2.)
    (Ad.deriv
       (fun y ->
          let k = Ops.to_float y in
          Ir.run1 (Ir.make1 (fun x -> Ops.(x * y * scalar (k /. k))) (s 0.)) (s 2.))
       (s 5.))
;;

let () =
  let leak transform =
    let r = ref (s 0.) in
    ignore
      (transform (fun x ->
         r := x;
         x));
    !r
  in
  Check.raises ~matches:Check.is_escaped "leaked jvp tracer cannot be read" (fun () ->
    Ops.to_float (leak (fun f -> Ad.deriv f (s 1.))));
  Check.raises ~matches:Check.is_escaped "leaked vjp tracer cannot be read" (fun () ->
    Ops.to_float (leak (fun f -> Ad.grad1 f (s 1.))));
  Check.raises ~matches:Check.is_escaped "leaked stage tracer cannot be read" (fun () ->
    Ops.to_float (leak (fun f -> Ir.run1 (Ir.make1 f (s 1.)) (s 1.))))
;;

let () =
  match Ir.make1 branchy (s 1.) with
  | _ -> Check.bool "concretization message" false
  | exception Concretization_error msg ->
    Check.bool
      "concretization message names the stage tracer"
      (String.length msg > 0 && String.sub msg 0 5 = "stage")
;;

let () = Check.finish ()
