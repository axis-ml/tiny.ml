open Tracer
open Expert

let s = Ops.scalar
let f x = Ops.(-(sin x * scalar 2.) + x)
let prim_names (g : Ir.t) = List.map (fun (e : Ir.eqn) -> Prim.name e.prim) g.eqns

let () =
  let g = Ir.make1 f (s 3.) in
  Check.bool "f has 4 eqns" (prim_names g = [ "sin"; "mul"; "neg"; "add" ]);
  Check.bool "f has 1 const" (List.length g.consts = 1);
  Check.value "run1 = f" ~expected:(f (s 5.)) (Ir.run1 g (s 5.));
  Check.value
    "deriv run1 = deriv f"
    ~expected:(Ad.deriv f (s 5.))
    (Ad.deriv (Ir.run1 g) (s 5.));
  Check.value
    "grad run1 = grad f"
    ~expected:(Ad.grad1 f (s 5.))
    (Ad.grad1 (Ir.run1 g) (s 5.));
  Check.bool "retrace run1" (prim_names (Ir.make1 (Ir.run1 g) (s 0.)) = prim_names g)
;;

let () =
  let x = Check.rand [| 2; 3 |]
  and w = Check.rand [| 3; 4 |] in
  let g = Ir.make1 (fun x -> Ops.dot x w) x in
  Check.bool "dot is one eqn" (prim_names g = [ "dot" ]);
  let gv = Ir.make1 (fun x -> Ops.dot x (Ops.array [| 1.; 2.; 3. |])) x in
  Check.bool
    "matrix-vector dot is a Dot between reshapes"
    (prim_names gv = [ "reshape"; "dot"; "reshape" ]);
  let loss x = Ops.(reduce_sum (sin (dot x w))) in
  let fast = Compile.compile1 ~backend:Compile.jit loss in
  Check.value "jit dot" ~expected:(loss x) (fast x);
  Check.value "grad of jit dot" ~expected:(Ad.grad1 loss x) (Ad.grad1 fast x);
  let dx = Check.rand ~lo:(-1.) ~hi:1. [| 2; 3 |] in
  Check.value
    "jvp of jit dot"
    ~expected:(snd (Ad.jvp1 loss x dx))
    (snd (Ad.jvp1 fast x dx));
  let jit_grad = Compile.compile1 ~backend:Compile.jit (Ad.grad1 loss) in
  Check.value "jit of grad dot" ~expected:(Ad.grad1 loss x) (jit_grad x);
  let gg = Ir.make1 (Ad.grad1 loss) x in
  Check.bool
    "grad of dot: one Dot forward, one back, none for the constant"
    (List.length (List.filter (String.equal "dot") (prim_names gg)) = 2)
;;

let () =
  let g =
    Ir.make1
      (fun x ->
         let a = Ops.exp x in
         Ops.((a * a) + a))
      (s 0.)
  in
  Check.bool "shared node: 3 eqns" (prim_names g = [ "exp"; "mul"; "add" ]);
  match g.eqns with
  | [ exp; mul; add ] ->
    let a = List.hd exp.outputs in
    Check.bool "shared node: mul uses it twice" (mul.inputs = [ a; a ]);
    Check.bool "shared node: add uses it" (List.mem a add.inputs)
  | _ -> Check.bool "shared node: shape" false
;;

let () =
  let layer (x, w) = Ops.(reduce_sum (relu (matmul x w))) in
  let x = Ops.ones [| 2; 3 |]
  and w = Ops.full [| 3; 4 |] 0.5 in
  let g = Ir.make Tree.(pair value value) Tree.value layer (x, w) in
  Check.bool
    "layer graph is well formed"
    (match Ir.invariant g with
     | () -> true
     | exception Failure _ -> false);
  Check.value "layer graph" ~expected:(layer (x, w)) (List.hd (Ir.run g [ x; w ]))
;;

let () =
  let g = Ir.trace (fun xs -> xs @ [ Ops.scalar 7. ]) [ Aval.create [| 2 |] ] in
  Check.bool "identity graph has no eqns" (g.eqns = []);
  match Ir.run g [ Ops.array [| 1.; 2. |] ] with
  | [ x; c ] ->
    Check.value "param output" ~expected:(Ops.array [| 1.; 2. |]) x;
    Check.value "const output" ~expected:(s 7.) c
  | _ -> Check.bool "two outputs" false
;;

let () =
  let g = Ir.make1 f (s 3.) in
  Check.raises ~matches:Check.is_type_error "run rejects another type" (fun () ->
    Ir.run1 g (Ops.array [| 1.; 2. |]));
  Check.raises ~matches:Check.is_type_error "run rejects an input count" (fun () ->
    Ir.run g []);
  Check.raises ~matches:Check.is_concretization "tracing cannot read data" (fun () ->
    Ir.make1 (fun x -> s (Ops.to_float x)) (s 3.))
;;

let () =
  let d =
    Ad.deriv
      (fun y ->
         let g = Ir.make1 (fun x -> Ops.(x * y)) (s 0.) in
         Ir.run1 g (s 2.))
      (s 5.)
  in
  Check.value "closed-over jvp tracer" ~expected:(s 2.) d
;;

let () =
  let traces = ref 0 in
  let jit = Compile.jit_with ~on_trace:(fun _ -> incr traces) in
  let g = Compile.compile1 ~backend:jit f in
  Check.value "jit 3" ~expected:(f (s 3.)) (g (s 3.));
  Check.value "jit 4" ~expected:(f (s 4.)) (g (s 4.));
  Check.bool "jit traces once per type" (!traces = 1);
  let v = Ops.array [| 0.; 1.; 2. |] in
  Check.value "jit on arrays" ~expected:(f v) (g v);
  Check.bool "jit retraces for a new type" (!traces = 2);
  Check.value "deriv of jit" ~expected:(Ad.deriv f (s 3.)) (Ad.deriv g (s 3.));
  Check.value "grad of jit" ~expected:(Ad.grad1 f (s 3.)) (Ad.grad1 g (s 3.));
  Check.bool "transformations reuse the trace" (!traces = 2);
  let h (x, ys) = Ops.(List.fold_left ( + ) x ys), List.map Ops.sin ys in
  let g =
    Compile.compile
      ~backend:Compile.jit
      Tree.(pair value (list value))
      Tree.(pair value (list value))
      h
  in
  List.iter
    (fun n ->
       let ys = List.init n (fun i -> s (float_of_int i)) in
       let sum, sins = g (s 10., ys) in
       let esum, esins = h (s 10., ys) in
       Check.value (Printf.sprintf "jit pytree sum, %d leaves" n) ~expected:esum sum;
       Check.bool
         (Printf.sprintf "jit pytree list, %d leaves" n)
         (List.length sins = n && List.for_all2 Ops.allclose sins esins))
    [ 2; 5; 2; 0; 5 ];
  let k (xs, ys) = [ Ops.(List.fold_left ( + ) (scalar 0.) xs * scalar 10.) ] @ ys in
  let g =
    Compile.compile
      ~backend:Compile.jit
      Tree.(pair (list value) (list value))
      Tree.(list value)
      k
  in
  let a = [ s 1.; s 2. ], [ s 3. ]
  and b = [ s 1. ], [ s 2.; s 3. ] in
  List.iter
    (fun (name, x) ->
       Check.bool
         ("jit keys on structure: " ^ name)
         (List.for_all2 Ops.allclose (g x) (k x)))
    [ "a", a; "b", b; "a again", a ]
;;

let () =
  let g = Ir.make1 f (s 3.) in
  Check.raises
    ~matches:(function
      | Invalid_argument _ -> true
      | _ -> false)
    "run rejects a var used before it is defined"
    (fun () -> Ir.run1 { g with eqns = List.tl g.eqns } (s 1.));
  let is_failure = function
    | Failure _ -> true
    | _ -> false
  in
  Check.raises
    ~matches:is_failure
    "invariant rejects a var used before it is defined"
    (fun () -> Ir.invariant { g with eqns = List.tl g.eqns });
  Check.raises
    ~matches:is_failure
    "invariant rejects an output of the wrong type"
    (fun () ->
       Ir.invariant
         { g with
           eqns =
             List.map
               (fun (e : Ir.eqn) ->
                  { e with
                    outputs =
                      List.map
                        (fun (v : Ir.var) -> { v with aval = Aval.create [| 7 |] })
                        e.outputs
                  })
               g.eqns
         });
  Check.raises ~matches:is_failure "invariant rejects a var defined twice" (fun () ->
    Ir.invariant { g with params = g.params @ g.params })
;;

let () =
  let traces = ref 0 in
  let w = ref (s 1.) in
  let fast =
    Compile.compile1
      ~backend:(Compile.jit_with ~on_trace:(fun _ -> incr traces))
      (fun x -> Ops.(x * !w))
  in
  let loss wv =
    w := wv;
    fast (s 3.)
  in
  Check.value "jit capture: grad" ~expected:(s 3.) (Ad.grad1 loss (s 2.));
  Check.value "jit capture: grad again" ~expected:(s 3.) (Ad.grad1 loss (s 5.));
  Check.bool "jit capture: retraced, not cached" (!traces = 2);
  w := s 10.;
  Check.value "jit after captures: arrays work" ~expected:(s 30.) (fast (s 3.));
  Check.value "jit after captures: cached" ~expected:(s 40.) (fast (s 4.));
  Check.bool "jit after captures: traced once more" (!traces = 3);
  let leaked = ref None in
  let _ =
    Ad.deriv
      (fun y ->
         let g = Ir.make1 (fun x -> Ops.(x * y)) (s 0.) in
         Check.bool "capture recorded" (List.length g.captures = 1 && g.consts = []);
         let g', vs = Ir.closure_convert g in
         Check.bool "closure_convert: captures become params" (List.length g'.params = 2);
         Check.value
           "closure_convert: run"
           ~expected:(Ops.mul (s 2.) y)
           (List.hd (Ir.run g' (vs @ [ s 2. ])));
         leaked := Some g;
         y)
      (s 5.)
  in
  match !leaked with
  | Some g ->
    Check.raises ~matches:Check.is_escaped "run with a dead capture" (fun () ->
      Ir.run1 g (s 1.))
  | None -> ()
;;

let () =
  let c = s 2. in
  let g = Ir.make1 (fun x -> Ops.((x * c) + c)) (s 0.) in
  Check.bool "one const per array" (List.length g.consts = 1);
  let g =
    Ir.make1
      (fun x ->
         let _unused = Ops.sin x in
         Ops.(exp x * exp x))
      (s 0.)
  in
  Check.bool
    "dead: the unused sin"
    (List.map (fun (e : Ir.eqn) -> Prim.name e.prim) (Ir.dead g) = [ "sin" ]);
  let x = List.hd g.params in
  Check.bool "users of the param" (List.length (Ir.users g).(x.id) = 3);
  Check.bool "params have no producer" ((Ir.producers g).(x.id) = None);
  (match List.rev g.eqns, g.outputs with
   | last :: _, [ out ] ->
     Check.bool
       "output producer"
       (match (Ir.producers g).(out.id) with
        | Some e -> e == last
        | None -> false)
   | _ -> Check.bool "eqns" false);
  let dot = Ir.to_dot g in
  let contains = Check.contains dot in
  Check.bool "to_dot: digraph" (contains "digraph ir {");
  Check.bool "to_dot: dead eqn dashed" (contains "label=\"sin\" style=dashed");
  Check.bool
    "num_vars covers ids"
    (List.for_all
       (fun (e : Ir.eqn) ->
          List.for_all (fun (v : Ir.var) -> v.id < g.num_vars) e.outputs)
       g.eqns)
;;

let () = Check.finish ()
