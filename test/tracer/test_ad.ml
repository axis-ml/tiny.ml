open Tracer
open Expert

let s = Ops.scalar
let f x = Ops.(-(sin x * scalar 2.) + x)
let df x = Ops.(scalar 1. - (scalar 2. * cos x))
let g x = Ops.(sin x * exp x)
let dg x = Ops.((cos x + sin x) * exp x)
let h x = Ops.(reduce_sum (x * x * x))
let x = Ops.array [| 0.5; 1.; 2. |]
let diag v = Ops.(eye (size v) * v)
let pair = Tree.(pair value value)
let k (a, b) = Ops.(a * sin b)

module _ : module type of Ad = struct
  let () =
    let y, dy = Ad.jvp_tree ~input:pair ~output:Tree.value k (s 2., s 1.) (s 1., s 0.) in
    Check.value "jvp primal" ~expected:(k (s 2., s 1.)) y;
    Check.value "jvp tangent along a" ~expected:(Ops.sin (s 1.)) dy;
    let _, dy = Ad.jvp_tree ~input:pair ~output:Tree.value k (s 2., s 1.) (s 0., s 1.) in
    Check.value "jvp tangent along b" ~expected:Ops.(scalar 2. * cos (scalar 1.)) dy;
    Check.raises
      ~matches:Check.is_type_error
      "jvp rejects a tangent of another type"
      (fun () ->
         Ad.jvp_tree ~input:pair ~output:Tree.value k (s 2., s 1.) (s 1., Ops.ones [| 2 |]))
  ;;

  let jvp_tree = Ad.jvp_tree

  let () =
    let y, dy = Ad.jvp f (s 3.) (s 1.) in
    Check.value "jvp primal" ~expected:(f (s 3.)) y;
    Check.value "jvp tangent" ~expected:(df (s 3.)) dy
  ;;

  let jvp = Ad.jvp

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
    let y, back = Ad.vjp_tree ~input:pair ~output:Tree.value k (s 2., s 1.) in
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

  let vjp_tree = Ad.vjp_tree

  let () =
    let y, back = Ad.vjp g x in
    Check.value "vjp primal" ~expected:(g x) y;
    Check.value "vjp back ones" ~expected:(dg x) (back (Ops.ones_like x));
    Check.value
      "vjp back twice"
      ~expected:Ops.(scalar 2. * dg x)
      (back (Ops.full [| 3 |] 2.))
  ;;

  let vjp = Ad.vjp

  let () =
    let l, (ga, gb) = Ad.value_and_grad_tree ~wrt:pair k (s 2., s 1.) in
    Check.value "value_and_grad value" ~expected:(k (s 2., s 1.)) l;
    Check.value "value_and_grad a" ~expected:(Ops.sin (s 1.)) ga;
    Check.value "value_and_grad b" ~expected:Ops.(scalar 2. * cos (scalar 1.)) gb
  ;;

  let value_and_grad_tree = Ad.value_and_grad_tree

  let () =
    let v, d = Ad.value_and_grad h x in
    Check.value "value_and_grad value" ~expected:(h x) v;
    Check.value "value_and_grad grad = 3x^2" ~expected:Ops.(scalar 3. * x * x) d;
    Check.raises
      ~matches:Check.is_type_error
      "value_and_grad rejects a non-scalar output"
      (fun () -> Ad.value_and_grad g x)
  ;;

  let value_and_grad = Ad.value_and_grad

  let () =
    let ga, gb = Ad.grad_tree ~wrt:pair k (s 2., s 1.) in
    Check.value "grad a" ~expected:(Ops.sin (s 1.)) ga;
    Check.value "grad b" ~expected:Ops.(scalar 2. * cos (scalar 1.)) gb
  ;;

  let grad_tree = Ad.grad_tree

  let () =
    Check.value "grad" ~expected:(df (s 3.)) (Ad.grad f (s 3.));
    Check.value
      "grad (grad f) = deriv (deriv f)"
      ~expected:(Ad.deriv (Ad.deriv f) (s 3.))
      (Ad.grad (Ad.grad f) (s 3.))
  ;;

  let grad = Ad.grad
  let () = Check.value "jacrev matches jacfwd" ~expected:(Ad.jacfwd g x) (Ad.jacrev g x)

  let () =
    let empty = Ops.zeros [| 0 |] in
    let to_3 x = Ops.(reduce_sum x + zeros [| 3 |]) in
    Check.bool "jacfwd of an empty input" (Ops.shape (Ad.jacfwd to_3 empty) = [| 3; 0 |]);
    Check.bool "jacrev of an empty input" (Ops.shape (Ad.jacrev to_3 empty) = [| 3; 0 |]);
    Check.bool
      "jacrev of an empty output"
      (Ops.shape (Ad.jacrev (fun _ -> empty) (Ops.zeros [| 2 |])) = [| 0; 2 |])
  ;;

  let jacrev = Ad.jacrev

  let () =
    Check.value
      "hessian = diag (6x)"
      ~expected:(diag Ops.(scalar 6. * x))
      (Ad.hessian h x);
    Check.value
      "hessian = jacfwd (grad h)"
      ~expected:(Ad.jacfwd (Ad.grad h) x)
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
  Check.bool "Nonzero is not" (nz 1. |> Z.is_zero |> not);
  Check.bool "map keeps a zero" (Z.map ~like:x Ops.neg z |> Z.is_zero);
  Check.value
    "map maps"
    ~expected:(s (-3.))
    (Z.map ~like:x Ops.neg (nz 3.) |> Z.instantiate);
  Check.value
    "add skips a zero"
    ~expected:(s 3.)
    (Z.add ~like:x z (nz 3.) |> Z.instantiate);
  Check.value "add adds" ~expected:(s 5.) (Z.add ~like:x (nz 2.) (nz 3.) |> Z.instantiate);
  Check.bool "add of zeros is a zero" (Z.add ~like:x z z |> Z.is_zero);
  Check.value
    "sub of a zero negates"
    ~expected:(s (-3.))
    (Z.sub ~like:x z (nz 3.) |> Z.instantiate);
  Check.value
    "sub subtracts"
    ~expected:(s (-1.))
    (Z.sub ~like:x (nz 2.) (nz 3.) |> Z.instantiate);
  Check.value "instantiate a zero" ~expected:(s 0.) (Z.instantiate z)
;;

let () = Check.finish ()
