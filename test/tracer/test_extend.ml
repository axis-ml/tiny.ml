open Tracer
open Expert

let s = Ops.scalar
let () = Random.init 7
let vs = Tree.(list value)
let rand = Check.rand
let un = Check.un
let bin = Check.bin

type Prim.t += Softplus

let softplus x = Interp.bind1 Softplus [ x ]

let () =
  Prim.register
    Softplus
    ~name:"softplus_ext"
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
  bin "maximum" Ops.maximum (a ()) (a ());
  bin "minimum" Ops.minimum (a ()) (a ());
  un "sqrt" Ops.sqrt (a ());
  un "integer_pow 3" (Ops.integer_pow ~exponent:3) (a ());
  un "pow 2.5" (fun x -> Ops.pow ~exponent:2.5 x) (a ());
  un "abs" Ops.abs (rand ~lo:(-2.) [| 2; 3 |]);
  un "tanh" Ops.tanh (rand ~lo:(-2.) [| 2; 3 |]);
  un "logistic" Ops.logistic (rand ~lo:(-2.) [| 2; 3 |]);
  un "tan" Ops.tan (rand ~lo:(-1.) [| 2; 3 |]);
  un "sinh" Ops.sinh (rand ~lo:(-2.) [| 2; 3 |]);
  un "cosh" Ops.cosh (rand ~lo:(-2.) [| 2; 3 |]);
  un "asinh" Ops.asinh (rand ~lo:(-2.) [| 2; 3 |]);
  un "acosh" Ops.acosh (rand ~lo:1.5 ~hi:3. [| 2; 3 |]);
  un "atanh" Ops.atanh (rand ~lo:(-0.8) ~hi:0.8 [| 2; 3 |]);
  un "relu" Ops.relu (rand ~lo:(-2.) [| 2; 3 |]);
  let mask = Ops.greater (rand ~lo:0. ~hi:1. [| 2; 3 |]) (s 0.5) in
  bin "select" (Ops.select mask) (a ()) (a ());
  un "reduce_max axes=[1]" (Ops.reduce_max ~axes:[ 1 ]) (a ());
  un "reduce_min" (fun x -> Ops.reduce_min x) (a ());
  un "mean keepdims" (Ops.mean ~axes:[ 0 ] ~keepdims:true) (a ());
  un "reshape" (Ops.reshape ~shape:[| 3; 2 |]) (a ());
  un
    "squeeze/expand_dims"
    (fun x -> x |> Ops.expand_dims ~axes:[ 1 ] |> Ops.squeeze)
    (a ());
  bin "concatenate" (fun x y -> Ops.concatenate ~axis:1 [ x; y ]) (a ()) (rand [| 2; 2 |]);
  bin "stack" (fun x y -> Ops.stack ~axis:(-1) [ x; y ]) (a ()) (a ());
  un "slice" (Ops.slice ~axis:1 ~start:1 ~stop:3) (a ());
  bin "dot matrix-matrix" Ops.dot (a ()) (rand [| 3; 4 |]);
  bin "dot matrix-vector" Ops.dot (a ()) (rand [| 3 |]);
  bin "dot vector-matrix" Ops.dot (rand [| 2 |]) (a ());
  bin "column + row (stretch size-1)" Ops.add (rand [| 3; 1 |]) (rand [| 1; 4 |]);
  Check.value
    "reduce_max"
    ~expected:(Ops.array [| 3.; 6. |])
    (Ops.array ~shape:[| 2; 3 |] [| 1.; 3.; 2.; 6.; 5.; 4. |]
     |> Ops.reduce_max ~axes:[ 1 ]);
  Check.value
    "dot = matmul by hand"
    ~expected:(Ops.array ~shape:[| 2; 2 |] [| 19.; 22.; 43.; 50. |])
    (Ops.dot
       (Ops.array ~shape:[| 2; 2 |] [| 1.; 2.; 3.; 4. |])
       (Ops.array ~shape:[| 2; 2 |] [| 5.; 6.; 7.; 8. |]));
  Check.value
    "[3,1] + [1,4] broadcasts to [3,4]"
    ~expected:
      (Array.init 12 (fun i -> float ((i / 4 * 10) + (i mod 4)))
       |> Ops.array ~shape:[| 3; 4 |])
    (Ops.add
       (Ops.array ~shape:[| 3; 1 |] [| 0.; 10.; 20. |])
       (Ops.array ~shape:[| 1; 4 |] [| 0.; 1.; 2.; 3. |]));
  Check.value
    "concatenate"
    ~expected:(Ops.array [| 1.; 2.; 3. |])
    (Ops.concatenate [ Ops.array [| 1. |]; Ops.array [| 2.; 3. |] ]);
  Check.value
    "jacfwd sin = diag cos"
    ~expected:Ops.(eye 3 * cos (array [| 0.; 1.; 2. |]))
    (Ad.jacfwd Ops.sin (Ops.array [| 0.; 1.; 2. |]));
  Check.raises ~matches:Check.is_type_error "to_float rejects a vector" (fun () ->
    Ops.array [| 1.; 2. |] |> Ops.to_float);
  Check.raises ~matches:Check.is_type_error "array rejects a wrong length" (fun () ->
    Ops.array ~shape:[| 2; 2 |] [| 1. |]);
  Check.raises ~matches:Check.is_type_error "dot rejects misaligned shapes" (fun () ->
    Ops.dot (Ops.zeros [| 2; 3 |]) (Ops.zeros [| 2 |]));
  Check.raises
    ~matches:Check.is_type_error
    "the Dot primitive rejects misaligned shapes"
    (fun () -> Interp.bind1 Builtin.Dot [ Ops.zeros [| 2; 3 |]; Ops.zeros [| 2; 3 |] ]);
  Check.raises ~matches:Check.is_type_error "the Dot primitive rejects vectors" (fun () ->
    Interp.bind1 Builtin.Dot [ Ops.zeros [| 3 |]; Ops.zeros [| 3 |] ]);
  Check.value
    "dot of rectangular matrices"
    ~expected:(Ops.array ~shape:[| 1; 3 |] [| 9.; 12.; 15. |])
    (Ops.dot
       (Ops.array ~shape:[| 1; 2 |] [| 1.; 2. |])
       (Ops.array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]));
  Check.value
    "dot vector-matrix"
    ~expected:(Ops.array [| 9.; 12.; 15. |])
    (Ops.dot
       (Ops.array [| 1.; 2. |])
       (Ops.array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]));
  let f (x, ys) = List.fold_left Ops.add (s 0.) ys |> Ops.mul x in
  let ins = Tree.(pair value (list value)) in
  let y, dy =
    Ad.jvp_tree
      ~input:ins
      ~output:Tree.value
      f
      (s 2., [ s 3.; s 4. ])
      (s 1., [ s 0.; s 1. ])
  in
  Check.float "jvp on a pair of value and list: primal" ~expected:14. (Ops.to_float y);
  Check.float "jvp on a pair of value and list: tangent" ~expected:9. (Ops.to_float dy);
  let net { w; b } x = Ops.(dot w x + b) in
  let x = Ops.array [| 1.; 2. |] in
  let p = { w = Ops.eye 2; b = Ops.zeros [| 2 |] } in
  let dp = { w = Ops.zeros [| 2; 2 |]; b = Ops.ones [| 2 |] } in
  let _, dy = Ad.jvp_tree ~input:layer ~output:Tree.value (fun p -> net p x) p dp in
  Check.value "jvp on a record (Tree.conv)" ~expected:(Ops.ones [| 2 |]) dy;
  let _, dlayer =
    Ad.jvp_tree
      ~input:Tree.value
      ~output:layer
      (fun x -> { w = Ops.mul x x; b = x })
      (s 3.)
      (s 1.)
  in
  Check.float "jvp returning a record" ~expected:6. (Ops.to_float dlayer.w);
  Check.raises
    ~matches:Check.is_type_error
    "jvp rejects tangents of another structure"
    (fun () -> Ad.jvp_tree ~input:vs ~output:vs Fun.id [ s 1.; s 2. ] [ s 1. ]);
  Check.bool
    "Tree.map doubles every leaf"
    (Tree.map ins (Ops.mul (s 2.)) (s 1., [ s 2. ])
     |> Tree.leaves ins
     |> List.map Ops.to_float
     = [ 2.; 4. ]);
  Check.bool
    "Tree.map2 pairs the leaves"
    (Tree.map2 ins Ops.sub (s 5., [ s 7. ]) (s 1., [ s 2. ])
     |> Tree.leaves ins
     |> List.map Ops.to_float
     = [ 4.; 5. ]);
  Check.raises
    ~matches:Check.is_type_error
    "Tree.map2 rejects values of different structures"
    (fun () -> Tree.map2 ins Ops.add (s 1., [ s 2. ]) (s 1., []));
  let wrong_count name f =
    Check.raises ~matches:Check.is_type_error ("wrong operand count: " ^ name) f
  in
  let bad_sin x = Interp.bind1 Builtin.Sin [ x; x ] in
  let bad_add x = Interp.bind1 Builtin.Add [ x ] in
  wrong_count "sin, eval" (fun () -> bad_sin (s 1.));
  wrong_count "add, eval" (fun () -> bad_add (s 1.));
  wrong_count "sin, jvp" (fun () -> Ad.deriv bad_sin (s 1.));
  wrong_count "add, jvp" (fun () -> Ad.deriv bad_add (s 1.));
  wrong_count "sin, vjp" (fun () -> Ad.grad bad_sin (s 1.));
  wrong_count "add, vjp" (fun () -> Ad.grad bad_add (s 1.));
  wrong_count "concatenate of nothing" (fun () -> Interp.bind1 (Builtin.Concatenate 0) []);
  wrong_count "impl called directly" (fun () -> Prim.impl Builtin.Sin []);
  Check.raises
    ~matches:(function
      | Invalid_argument _ -> true
      | _ -> false)
    "an unregistered primitive is Invalid_argument"
    (fun () ->
       let module M = struct
         type Prim.t += Unregistered
       end
       in
       Interp.bind1 M.Unregistered [ s 1. ]);
  Check.raises
    ~matches:(function
      | Error (Missing_rule m) -> m = "jvp has no rule for softplus_ext"
      | _ -> false)
    "a primitive without a jvp rule is rejected by name"
    (fun () -> Ad.deriv softplus (s 0.));
  Check.float
    "a user primitive evaluates"
    ~expected:(Float.log 2.)
    (s 0. |> softplus |> Ops.to_float);
  Check.bool
    "missing_rules lists softplus_ext"
    (List.mem "softplus_ext" (Jvp.missing_rules ()));
  Derivatives.register_elementwise Softplus ~partials:(fun _ xs _ ->
    List.map (fun x -> Some (fun t -> Ops.(t * logistic x))) xs);
  Check.bool
    "softplus_ext has a jvp rule once its partials are registered"
    (Jvp.missing_rules () |> List.mem "softplus_ext" |> not);
  un "user primitive softplus" softplus (rand ~lo:(-2.) [| 2; 3 |]);
  Check.float
    "user primitive nests: softplus'' 0 = 1/4"
    ~expected:0.25
    (Ad.deriv (Ad.deriv softplus) (s 0.) |> Ops.to_float);
  Check.float
    "user primitive, reverse over forward: grad (deriv softplus) 0 = 1/4"
    ~expected:0.25
    (Ad.grad (Ad.deriv softplus) (s 0.) |> Ops.to_float);
  let i1 : int Interp.t = Interp.create ~name:"a" (fun _ -> Interp.bind)
  and i2 : int Interp.t = Interp.create ~name:"b" (fun _ -> Interp.bind) in
  Check.bool
    "interpreters get distinct non-zero tags"
    (Interp.id i1 <> Interp.id i2 && Interp.id i1 > 0);
  Check.bool "payload ignores arrays" (Interp.payload i1 (s 1.) = None);
  Check.bool
    "payload gets an interpreter's own payload back"
    (Interp.tracer i1 (Aval.create [||]) 42 |> Interp.payload i1 = Some 42);
  Check.bool
    "payload ignores another interpreter's tracers"
    (Interp.tracer i1 (Aval.create [||]) 42 |> Interp.payload i2 = None);
  let module M = struct
    type Prim.t += Twice
  end
  in
  Prim.register
    M.Twice
    ~name:"twice_ext"
    ~abstract_eval:(fun _ avals -> avals)
    ~impl:(fun _ -> List.map (Ndarray.map (fun x -> 2. *. x)));
  Jvp.register_rule M.Twice (fun _ primals _ -> primals, []);
  Check.raises
    ~matches:Check.is_type_error
    "a jvp rule with too few tangents is a Type_error"
    (fun () -> Ad.deriv (fun x -> Interp.bind1 M.Twice [ x ]) (s 1.));
  Check.raises
    ~matches:(function
      | Invalid_argument _ -> true
      | _ -> false)
    "another primitive cannot take a registered name"
    (fun () ->
       let module N = struct
         type Prim.t += Other
       end
       in
       Prim.register
         N.Other
         ~name:"twice_ext"
         ~abstract_eval:(fun _ a -> a)
         ~impl:(fun _ a -> a));
  Prim.register
    M.Twice
    ~name:"twice_ext"
    ~abstract_eval:(fun _ avals -> avals)
    ~impl:(fun _ -> List.map (Ndarray.map (fun x -> 2. *. x)));
  Check.bool "re-registering the same constructor replaces it" true;
  let module W = struct
    type Prim.t += Wrong
  end
  in
  Prim.register
    W.Wrong
    ~name:"wrong_ext"
    ~abstract_eval:(fun _ avals -> avals)
    ~impl:(fun _ arrays -> arrays);
  Jvp.register_rule W.Wrong (fun _ primals _ ->
    primals, [ Maybe_zero.Nonzero (Ops.ones [| 2 |]) ]);
  let wrong x = Interp.bind1 W.Wrong [ x ] in
  let module L = struct
    type Prim.t += Wrong_t
  end
  in
  Prim.register
    L.Wrong_t
    ~name:"wrong_transpose_ext"
    ~abstract_eval:(fun _ avals -> avals)
    ~impl:(fun _ arrays -> arrays);
  Derivatives.register_linear L.Wrong_t ~transpose:(fun _ _ _ ->
    [ Maybe_zero.Nonzero (Ops.ones [| 2 |]) ]);
  let wrong_t x = Interp.bind1 L.Wrong_t [ x ] in
  let with_debug f =
    let saved = !Private.debug in
    Private.debug := true;
    Fun.protect ~finally:(fun () -> Private.debug := saved) f
  in
  Check.raises
    ~matches:Check.is_type_error
    "debug: a jvp rule's tangent of the wrong type is a Type_error"
    (fun () -> with_debug (fun () -> Ad.deriv wrong (s 1.)));
  Check.raises
    ~matches:Check.is_type_error
    "debug: a transpose rule's cotangent of the wrong type is a Type_error"
    (fun () -> with_debug (fun () -> Ad.grad wrong_t (s 1.)));
  let cube =
    Define.unary
      ~name:"cube_def"
      ~impl:(fun x -> x *. x *. x)
      ~deriv:(fun x t -> Ops.(t * (s 3. * x * x)))
  in
  let hypot =
    Define.binary ~name:"hypot_def" ~impl:Float.hypot ~deriv:(fun x y ->
      let r = Ops.sqrt Ops.((x * x) + (y * y)) in
      (fun tx -> Ops.(tx * (x / r))), fun ty -> Ops.(ty * (y / r)))
  in
  let xs = Ops.array [| 1.; 2.; -3. |] in
  Check.value "Define.unary evaluates" ~expected:(Ops.array [| 1.; 8.; -27. |]) (cube xs);
  Check.value
    "Define.unary deriv"
    ~expected:(Ops.array [| 3.; 12.; 27. |])
    (Ad.deriv cube xs);
  Check.value
    "Define.unary grad transposes its deriv"
    ~expected:(Ops.array [| 3.; 12.; 27. |])
    (Ad.grad (fun x -> cube x |> Ops.reduce_sum) xs);
  Check.float
    "Define.unary nests: cube'' 2 = 12"
    ~expected:12.
    (Ad.deriv (Ad.deriv cube) (s 2.) |> Ops.to_float);
  un "Define.unary cube" cube (rand ~lo:(-2.) [| 2; 3 |]);
  let m = Ops.array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  Check.value "Define.unary vmap" ~expected:(cube m) (Vmap.vmap cube m);
  Check.value
    "Define.unary vmap over axis 1"
    ~expected:(Ops.transpose (cube m))
    (Vmap.vmap ~in_axis:1 cube m);
  Check.value
    "Define.unary jit"
    ~expected:(cube m)
    (Compile.compile ~backend:Compile.jit cube m);
  let sum_cube x = cube x |> Ops.reduce_sum in
  Check.value
    "Define.unary grad under jit"
    ~expected:(Ad.grad sum_cube m)
    (Compile.compile ~backend:Compile.jit (Ad.grad sum_cube) m);
  Check.bool
    "Define.unary registers every rule"
    ([ Jvp.missing_rules (); Batching.missing_rules () ]
     |> List.exists (List.mem "cube_def")
     |> not);
  Check.raises
    ~matches:(function
      | Invalid_argument _ -> true
      | _ -> false)
    "Define.unary rejects a taken name"
    (fun () -> Define.unary ~name:"cube_def" ~impl:Fun.id ~deriv:(fun _ t -> t));
  let x = Ops.array [| 3.; 5. |]
  and y = Ops.array [| 4.; 12. |] in
  Check.value "Define.binary evaluates" ~expected:(Ops.array [| 5.; 13. |]) (hypot x y);
  Check.value
    "Define.binary broadcasts like add"
    ~expected:
      (Ops.array ~shape:[| 2; 2 |] [| 5.; Float.hypot 3. 8.; Float.hypot 6. 4.; 10. |])
    (hypot (Ops.array ~shape:[| 2; 1 |] [| 3.; 6. |]) (Ops.array [| 4.; 8. |]));
  Check.value
    "Define.binary scalar operand"
    ~expected:(Ops.array [| 5.; Float.hypot 5. 4. |])
    (hypot x (s 4.));
  Check.value
    "Define.binary deriv in x"
    ~expected:(Ops.array [| 0.6; 5. /. 13. |])
    (Ad.deriv (fun x -> hypot x y) x);
  Check.value
    "Define.binary grad in y"
    ~expected:(Ops.array [| 0.8; 12. /. 13. |])
    (Ad.grad (fun y -> hypot x y |> Ops.reduce_sum) y);
  Check.float
    "Define.binary grad through a broadcast scalar"
    ~expected:(0.8 +. (4. /. Float.hypot 5. 4.))
    (Ad.grad (fun t -> hypot x t |> Ops.reduce_sum) (s 4.) |> Ops.to_float);
  bin "Define.binary hypot" hypot (rand [| 2; 3 |]) (rand [| 2; 3 |]);
  bin "Define.binary hypot, broadcast" hypot (rand [| 2; 3 |]) (rand [| 3 |]);
  let p = Ops.array ~shape:[| 2; 2 |] [| 3.; 5.; 6.; 8. |] in
  Check.value
    "Define.binary vmap, one side batched"
    ~expected:(hypot p (Ops.array [| 4.; 12. |]))
    (Vmap.vmap (fun r -> hypot r (Ops.array [| 4.; 12. |])) p);
  Check.value
    "Define.binary vmap, both sides batched on different axes"
    ~expected:(hypot p (Ops.transpose p))
    (Vmap.vmap_tree
       ~in_axes:[ Some 0; Some 1 ]
       ~input:vs
       ~output:Tree.value
       (function
         | [ a; b ] -> hypot a b
         | _ -> assert false)
       [ p; p ]);
  Check.value
    "Define.binary jit"
    ~expected:(hypot p (s 4.))
    (Compile.compile ~backend:Compile.jit (fun a -> hypot a (s 4.)) p);
  Check.value
    "Define.binary vmap of grad under jit"
    ~expected:Ops.(p / hypot p (s 4.))
    (Compile.compile
       ~backend:Compile.jit
       (Vmap.vmap (Ad.grad (fun r -> hypot r (s 4.) |> Ops.reduce_sum)))
       p);
  Check.raises
    ~matches:Check.is_type_error
    "Define.binary rejects shapes that do not broadcast"
    (fun () -> hypot (Ops.zeros [| 2 |]) (Ops.zeros [| 3 |]));
  Check.finish ()
;;
