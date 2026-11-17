open Tracer
open Expert

let s = Ops.scalar
let () = Random.init 11
let rand = Check.rand
let vs = Tree.(list value)
let inner a b = Ops.to_float (Ops.reduce_sum (Ops.mul a b))
let sum_inner xs ys = List.fold_left2 (fun acc a b -> acc +. inner a b) 0. xs ys

(* <ct, J dx> = <J^T ct, dx>: reverse mode against forward mode. *)
let dot_check name f xs =
  let dxs = List.map (fun x -> rand ~lo:(-1.) ~hi:1. (Ops.shape x)) xs in
  let ys, dys = Ad.jvp vs vs f xs dxs in
  let ys', back = Ad.vjp vs vs f xs in
  List.iter2 (fun a b -> Check.value ("vjp primal " ^ name) ~expected:a b) ys ys';
  let cts = List.map (fun y -> rand ~lo:(-1.) ~hi:1. (Ops.shape y)) ys in
  let cxs = back cts in
  Check.float
    ~rtol:1e-10
    ~atol:1e-10
    ("vjp " ^ name)
    ~expected:(sum_inner cts dys)
    (sum_inner cxs dxs)
;;

let un name op x =
  dot_check
    name
    (function
      | [ x ] -> [ op x ]
      | _ -> assert false)
    [ x ]
;;

let bin name op x y =
  dot_check
    name
    (function
      | [ x; y ] -> [ op x y ]
      | _ -> assert false)
    [ x; y ]
;;

type Prim.t += Softplus

let softplus x = Interp.bind1 Softplus [ x ]

let () =
  Prim.register
    Softplus
    ~name:"softplus_vjp"
    ~abstract_eval:(fun _ avals -> avals)
    ~impl:(fun _ -> List.map (Ndarray.map (fun x -> Float.log1p (Float.exp x))))
;;

type layer =
  { w : Ops.value
  ; b : Ops.value
  }

let layer =
  Tree.(
    conv
      ~to_tree:(fun { w; b } -> w, b)
      ~of_tree:(fun (w, b) -> { w; b })
      (pair value value))
;;

let () =
  let a () = rand [| 2; 3 |] in
  let signed () = rand ~lo:(-2.) [| 2; 3 |] in
  bin "add" Ops.add (a ()) (a ());
  bin "sub" Ops.sub (a ()) (a ());
  bin "mul" Ops.mul (a ()) (a ());
  bin "div" Ops.div (a ()) (a ());
  bin "maximum" Ops.maximum (a ()) (a ());
  bin "minimum" Ops.minimum (a ()) (a ());
  bin "greater (zero cotangent)" (fun x y -> Ops.mul x (Ops.greater x y)) (a ()) (a ());
  un "neg" Ops.neg (a ());
  un "sin" Ops.sin (signed ());
  un "cos" Ops.cos (signed ());
  un "exp" Ops.exp (signed ());
  un "log" Ops.log (a ());
  un "sqrt" Ops.sqrt (a ());
  un "integer_pow 3" (Ops.integer_pow ~exponent:3) (signed ());
  un "integer_pow 0" (fun x -> Ops.add x (Ops.integer_pow ~exponent:0 x)) (a ());
  un "pow 2.5" (fun x -> Ops.pow ~exponent:2.5 x) (a ());
  un "tanh" Ops.tanh (signed ());
  un "logistic" Ops.logistic (signed ());
  un "tan" Ops.tan (rand ~lo:(-1.) ~hi:1. [| 2; 3 |]);
  un "sinh" Ops.sinh (signed ());
  un "cosh" Ops.cosh (signed ());
  un "asinh" Ops.asinh (signed ());
  un "acosh" Ops.acosh (rand ~lo:1.5 ~hi:3. [| 2; 3 |]);
  un "atanh" Ops.atanh (rand ~lo:(-0.8) ~hi:0.8 [| 2; 3 |]);
  un "relu" Ops.relu (signed ());
  un "abs" Ops.abs (signed ());
  let mask = Ops.greater (rand ~lo:0. ~hi:1. [| 2; 3 |]) (s 0.5) in
  bin "select" (Ops.select mask) (a ()) (a ());
  un "reduce_sum" (fun x -> Ops.reduce_sum x) (a ());
  un "reduce_sum axes=[0]" (Ops.reduce_sum ~axes:[ 0 ]) (a ());
  un "reduce_max axes=[1]" (Ops.reduce_max ~axes:[ 1 ]) (a ());
  un "reduce_max with ties" (fun x -> Ops.reduce_max (Ops.concatenate [ x; x ])) (a ());
  un "mean keepdims" (Ops.mean ~axes:[ 0 ] ~keepdims:true) (a ());
  un "broadcast" (Ops.broadcast ~shape:[| 4; 2; 3 |] ~axes:[ 0 ]) (a ());
  un "transpose" (fun x -> Ops.transpose x) (a ());
  un
    "transpose rank 3"
    (Ops.transpose ~perm:[| 2; 0; 1 |])
    (Ops.reshape ~shape:[| 1; 2; 3 |] (a ()));
  un "reshape" (Ops.reshape ~shape:[| 3; 2 |]) (a ());
  bin "concatenate" (fun x y -> Ops.concatenate ~axis:1 [ x; y ]) (a ()) (rand [| 2; 2 |]);
  bin "stack" (fun x y -> Ops.stack ~axis:(-1) [ x; y ]) (a ()) (a ());
  un "slice" (Ops.slice ~axis:1 ~start:1 ~stop:3) (a ());
  un "slice from 0" (Ops.slice ~start:0 ~stop:1) (a ());
  un "slice to the end" (Ops.slice ~axis:(-1) ~start:(-2) ~stop:3) (a ());
  bin "dot matrix-matrix" Ops.dot (a ()) (rand [| 3; 4 |]);
  bin "dot matrix-vector" Ops.dot (a ()) (rand [| 3 |]);
  bin "dot vector-matrix" Ops.dot (rand [| 2 |]) (a ());
  bin "dot vector-vector" Ops.dot (rand [| 3 |]) (rand [| 3 |]);
  let m = rand [| 4; 2 |] in
  un "dot with a constant" (Ops.dot m) (a ());
  bin "column + row (stretch size-1)" Ops.add (rand [| 3; 1 |]) (rand [| 1; 4 |]);
  dot_check
    "two outputs, one input unused"
    (function
      | [ x; _ ] -> [ Ops.sin x; Ops.reduce_sum (Ops.mul x x) ]
      | _ -> assert false)
    [ a (); a () ];
  dot_check
    "an output that is an input"
    (function
      | [ x ] -> [ x; Ops.exp x ]
      | _ -> assert false)
    [ a () ];
  let x = Ops.array [| 0.; 1.; 2. |] in
  let f x = Ops.(reduce_sum (sin x * x)) in
  let y, g = Ad.value_and_grad Tree.value f x in
  Check.float "value_and_grad: value" ~expected:(Ops.to_float (f x)) (Ops.to_float y);
  Check.value
    "grad (sum (x sin x)) = sin x + x cos x"
    ~expected:Ops.(sin x + (x * cos x))
    g;
  Check.value
    "slice values"
    ~expected:(Ops.array ~shape:[| 2; 2 |] [| 2.; 3.; 5.; 6. |])
    (Ops.slice
       ~axis:1
       ~start:1
       ~stop:3
       (Ops.array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]));
  Check.value
    "grad of an unused input is zeros"
    ~expected:(Ops.zeros [| 3 |])
    (Ad.grad1 (fun _ -> s 1.) x);
  Check.value
    "grad at ties splits like jvp (relu at 0)"
    ~expected:(Ops.array [| 0.; 0.5; 1. |])
    (Ad.grad1 (fun x -> Ops.reduce_sum (Ops.relu (Ops.sub x (s 1.)))) x);
  let back = snd (Ad.vjp Tree.value Tree.value Ops.sin x) in
  Check.value
    "back can be called twice"
    ~expected:(back (Ops.ones [| 3 |]))
    (back (Ops.ones [| 3 |]));
  let kept f =
    let w = Weak.create 1 in
    let _, back =
      Ad.vjp1
        (fun x ->
           let y, out = f x in
           Weak.set w 0 (Some (Ops.to_ndarray y));
           out)
        x
    in
    Gc.full_major ();
    let alive = Weak.check w 0 in
    ignore (back (Ops.ones [| 3 |]));
    alive
  in
  Check.bool
    "back keeps an intermediate its rule reads"
    (kept (fun x ->
       let y = Ops.add x x in
       y, Ops.sin y));
  Check.bool
    "back drops an intermediate a linear rule does not read"
    (not
       (kept (fun x ->
          let y = Ops.add x x in
          y, Ops.neg y)));
  Check.bool
    "back drops an op that does not reach the output"
    (not
       (kept (fun x ->
          let y = Ops.exp x in
          y, Ops.neg x)));
  let g x = Ops.(sin x * reduce_sum x) in
  Check.value "jacrev = jacfwd" ~expected:(Ad.jacfwd g x) (Ad.jacrev g x);
  let m = rand [| 2; 2 |] in
  let h x = Ops.(exp (dot x x)) in
  Check.value "jacrev = jacfwd on a matrix" ~expected:(Ad.jacfwd h m) (Ad.jacrev h m);
  Check.value
    "jacrev of a scalar function = grad"
    ~expected:(Ad.grad1 f x)
    (Ad.jacrev f x);
  let hs = Ad.hessian (fun x -> Ops.(reduce_sum (x * x * x) * reduce_sum (sin x))) x in
  Check.value "hessian is symmetric" ~expected:(Ops.transpose hs) hs;
  Check.value
    "hessian of sum (x sin x) = diag (2 cos x - x sin x)"
    ~expected:Ops.(eye 3 * ((scalar 2. * cos x) - (x * sin x)))
    (Ad.hessian f x);
  let cube x = Ops.(reduce_sum (x * x * x)) in
  let p = Ops.array [| 1.; 2. |] in
  Check.value
    "reverse over reverse: grad (sum (grad sum x^3)) = 6x"
    ~expected:Ops.(scalar 6. * p)
    (Ad.grad1 (fun x -> Ops.reduce_sum (Ad.grad1 cube x)) p);
  Check.value
    "reverse over forward: grad (sum (deriv sin)) = -sin"
    ~expected:(Ops.neg (Ops.sin x))
    (Ad.grad1 (fun x -> Ops.reduce_sum (Ad.deriv Ops.sin x)) x);
  Check.float
    "forward over reverse: deriv (grad sin) 1 = -sin 1"
    ~expected:(-.Float.sin 1.)
    (Ops.to_float (Ad.deriv (Ad.grad1 Ops.sin) (s 1.)));
  Check.float
    "third order: grad (grad (grad x^4)) 2 = 48"
    ~expected:48.
    (Ops.to_float Ad.(grad1 (grad1 (grad1 (Ops.integer_pow ~exponent:4))) (s 2.)));
  (* Perturbation confusion: d/dx (x * d/dy (x + y)) = 1. *)
  Check.float
    "no perturbation confusion"
    ~expected:1.
    (Ops.to_float
       (Ad.grad1 (fun x -> Ops.mul x (Ad.grad1 (fun y -> Ops.add x y) (s 1.))) (s 1.)));
  let w = Ops.array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Ops.array [| 0.5; -1. |] in
  let loss (w, b) = Ops.(mean (square (matmul w x + b))) in
  let gw, gb = Ad.grad Tree.(pair value value) loss (w, b) in
  let r = Ops.array [| 8.5; 16. |] in
  Check.value "grad on a pair: b" ~expected:r gb;
  Check.value
    "grad on a pair: w = r outer x"
    ~expected:Ops.(expand_dims ~axes:[ 1 ] r * expand_dims ~axes:[ 0 ] x)
    gw;
  let net { w; b } = Ops.(reduce_sum (dot w x + b)) in
  let gl = Ad.grad layer net { w; b } in
  Check.value "grad on a record: b" ~expected:(Ops.ones [| 2 |]) gl.b;
  Check.value "grad on a record: w" ~expected:(Ops.stack [ x; x ]) gl.w;
  let ys, back =
    Ad.vjp Tree.value Tree.(pair value value) (fun x -> Ops.sin x, Ops.cos x) x
  in
  Check.value "vjp with a pair out: primal" ~expected:(Ops.cos x) (snd ys);
  Check.value
    "vjp with a pair out: cotangent"
    ~expected:Ops.(cos x - sin x)
    (back (Ops.ones [| 3 |], Ops.ones [| 3 |]));
  Check.raises ~matches:Check.is_type_error "grad rejects a non-scalar output" (fun () ->
    Ad.grad1 Ops.sin x);
  Check.raises
    ~matches:Check.is_type_error
    "back rejects a cotangent of another shape"
    (fun () -> back (Ops.ones [| 2 |], Ops.ones [| 3 |]));
  Check.raises
    ~matches:Check.is_type_error
    "slice rejects an out-of-range stop"
    (fun () -> Ops.slice ~start:0 ~stop:4 x);
  Check.raises
    ~matches:(function
      | Missing_rule m -> m = "vjp has no rule for softplus_vjp"
      | _ -> false)
    "a primitive without a vjp rule is rejected by name"
    (fun () -> Ad.grad1 softplus (s 0.));
  Check.bool
    "missing_rules lists softplus_vjp"
    (List.mem "softplus_vjp" (Vjp.missing_rules ()));
  Derivatives.register_elementwise Softplus ~partials:(fun _ xs _ ->
    List.map (fun x -> Some (fun t -> Ops.mul t (Ops.logistic x))) xs);
  Check.bool
    "softplus_vjp has a vjp rule once its partials are registered"
    (not (List.mem "softplus_vjp" (Vjp.missing_rules ())));
  Check.float
    "user primitive: grad softplus 0 = 1/2"
    ~expected:0.5
    (Ops.to_float (Ad.grad1 softplus (s 0.)));
  Check.float
    "user primitive nests: grad (grad softplus) 0 = 1/4"
    ~expected:0.25
    (Ops.to_float (Ad.grad1 (Ad.grad1 softplus) (s 0.)));
  Check.finish ()
;;
