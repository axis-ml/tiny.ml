(* Sealed with [module type of Ad]: an untested function of Ad does not compile. *)

open Tracer
open Expert

let s = Ops.scalar

(* f' = 1 - 2 cos x, f'' = 2 sin x *)
let f x = Ops.(-(sin x * scalar 2.) + x)
let df x = Ops.(scalar 1. - (scalar 2. * cos x))

(* g' = (cos x + sin x) e^x *)
let g x = Ops.(sin x * exp x)
let dg x = Ops.((cos x + sin x) * exp x)

(* grad = 3 x^2, hessian = diag (6 x) *)
let h x = Ops.(reduce_sum (x * x * x))
let x = Ops.array [| 0.5; 1.; 2. |]
let diag v = Ops.(eye (size v) * v)
let pair = Tree.(pair value value)

(* k (a, b) = a sin b: dk/da = sin b, dk/db = a cos b *)
let k (a, b) = Ops.(a * sin b)

module _ : module type of Ad = struct
  let () =
    let y, dy = Ad.jvp pair Tree.value k (s 2., s 1.) (s 1., s 0.) in
    Check.value "jvp primal" ~expected:(k (s 2., s 1.)) y;
    Check.value "jvp tangent along a" ~expected:(Ops.sin (s 1.)) dy;
    let _, dy = Ad.jvp pair Tree.value k (s 2., s 1.) (s 0., s 1.) in
    Check.value "jvp tangent along b" ~expected:Ops.(scalar 2. * cos (scalar 1.)) dy;
    Check.raises
      ~matches:Check.is_type_error
      "jvp rejects a tangent of another type"
      (fun () -> Ad.jvp pair Tree.value k (s 2., s 1.) (s 1., Ops.ones [| 2 |]))
  ;;

  let jvp = Ad.jvp

  let () =
    let y, dy = Ad.jvp1 f (s 3.) (s 1.) in
    Check.value "jvp1 primal" ~expected:(f (s 3.)) y;
    Check.value "jvp1 tangent" ~expected:(df (s 3.)) dy
  ;;

  let jvp1 = Ad.jvp1

  let () =
    Check.value "deriv" ~expected:(df (s 3.)) (Ad.deriv f (s 3.));
    Check.value "deriv elementwise" ~expected:(dg x) (Ad.deriv g x);
    Check.value
      "deriv nests"
      ~expected:Ops.(scalar 2. * sin (scalar 3.))
      (Ad.deriv (Ad.deriv f) (s 3.))
  ;;

  let deriv = Ad.deriv

  let () =
    Check.value
      "jacfwd of an elementwise g is diagonal"
      ~expected:(diag (dg x))
      (Ad.jacfwd g x)
  ;;

  let jacfwd = Ad.jacfwd

  let () =
    let y, back = Ad.vjp pair Tree.value k (s 2., s 1.) in
    Check.value "vjp primal" ~expected:(k (s 2., s 1.)) y;
    let ga, gb = back (s 1.) in
    Check.value "vjp cotangent a" ~expected:(Ops.sin (s 1.)) ga;
    Check.value "vjp cotangent b" ~expected:Ops.(scalar 2. * cos (scalar 1.)) gb;
    let ga, _ = back (s 2.) in
    Check.value "vjp back is linear" ~expected:Ops.(scalar 2. * sin (scalar 1.)) ga;
    Check.raises
      ~matches:Check.is_type_error
      "vjp rejects a cotangent of another type"
      (fun () -> back (Ops.ones [| 2 |]))
  ;;

  let vjp = Ad.vjp

  let () =
    let y, back = Ad.vjp1 g x in
    Check.value "vjp1 primal" ~expected:(g x) y;
    Check.value "vjp1 back ones" ~expected:(dg x) (back (Ops.ones_like x));
    Check.value
      "vjp1 back twice"
      ~expected:Ops.(scalar 2. * dg x)
      (back (Ops.full [| 3 |] 2.))
  ;;

  let vjp1 = Ad.vjp1

  let () =
    let l, (ga, gb) = Ad.value_and_grad pair k (s 2., s 1.) in
    Check.value "value_and_grad value" ~expected:(k (s 2., s 1.)) l;
    Check.value "value_and_grad a" ~expected:(Ops.sin (s 1.)) ga;
    Check.value "value_and_grad b" ~expected:Ops.(scalar 2. * cos (scalar 1.)) gb
  ;;

  let value_and_grad = Ad.value_and_grad

  let () =
    let v, d = Ad.value_and_grad1 h x in
    Check.value "value_and_grad1 value" ~expected:(h x) v;
    Check.value "value_and_grad1 grad = 3x^2" ~expected:Ops.(scalar 3. * x * x) d;
    Check.raises
      ~matches:Check.is_type_error
      "value_and_grad1 rejects a non-scalar output"
      (fun () -> Ad.value_and_grad1 g x)
  ;;

  let value_and_grad1 = Ad.value_and_grad1

  let () =
    let ga, gb = Ad.grad pair k (s 2., s 1.) in
    Check.value "grad a" ~expected:(Ops.sin (s 1.)) ga;
    Check.value "grad b" ~expected:Ops.(scalar 2. * cos (scalar 1.)) gb
  ;;

  let grad = Ad.grad

  let () =
    Check.value "grad1" ~expected:(df (s 3.)) (Ad.grad1 f (s 3.));
    Check.value
      "grad1 (grad1 f) = deriv (deriv f)"
      ~expected:(Ad.deriv (Ad.deriv f) (s 3.))
      (Ad.grad1 (Ad.grad1 f) (s 3.))
  ;;

  let grad1 = Ad.grad1
  let () = Check.value "jacrev matches jacfwd" ~expected:(Ad.jacfwd g x) (Ad.jacrev g x)
  let jacrev = Ad.jacrev

  let () =
    Check.value
      "hessian = diag (6x)"
      ~expected:(diag Ops.(scalar 6. * x))
      (Ad.hessian h x);
    Check.value
      "hessian = jacfwd (grad1 h)"
      ~expected:(Ad.jacfwd (Ad.grad1 h) x)
      (Ad.hessian h x)
  ;;

  let hessian = Ad.hessian
end

let () =
  let module Z = Maybe_zero in
  let x = s 2. in
  let z = Z.zero_like x in
  let nz v = Z.Nonzero (s v) in
  Check.bool "zero_like is a zero" (Z.is_zero z);
  Check.bool "Nonzero is not" (not (Z.is_zero (nz 1.)));
  Check.bool "map keeps a zero" (Z.is_zero (Z.map x Ops.neg z));
  Check.value "map maps" ~expected:(s (-3.)) (Z.instantiate (Z.map x Ops.neg (nz 3.)));
  Check.value "add skips a zero" ~expected:(s 3.) (Z.instantiate (Z.add x z (nz 3.)));
  Check.value "add adds" ~expected:(s 5.) (Z.instantiate (Z.add x (nz 2.) (nz 3.)));
  Check.bool "add of zeros is a zero" (Z.is_zero (Z.add x z z));
  Check.value
    "sub of a zero negates"
    ~expected:(s (-3.))
    (Z.instantiate (Z.sub x z (nz 3.)));
  Check.value
    "sub subtracts"
    ~expected:(s (-1.))
    (Z.instantiate (Z.sub x (nz 2.) (nz 3.)));
  Check.value "instantiate a zero" ~expected:(s 0.) (Z.instantiate z)
;;

let () = Check.finish ()
