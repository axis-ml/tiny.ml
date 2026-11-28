open Tracer

let s = Ops.scalar
let f x = Ops.(-(sin x * scalar 2.) + x)

let () =
  let g = Compile.compile f in
  Check.value "eager compile = f" ~expected:(f (s 3.)) (g (s 3.));
  let x = Ops.array [| 0.; 1.; 2. |] in
  Check.value "eager compile on arrays" ~expected:(f x) (g x);
  Check.value
    "deriv of compiled = deriv"
    ~expected:(Ad.deriv f (s 3.))
    (Ad.deriv g (s 3.));
  Check.value
    "nested deriv of compiled"
    ~expected:(Ad.deriv (Ad.deriv f) (s 3.))
    (Ad.deriv (Ad.deriv g) (s 3.))
;;

let () =
  let h (x, ys) = Ops.(List.fold_left ( + ) x ys), List.map Ops.sin ys in
  let g =
    Compile.compile_tree
      ~input:Tree.(pair value (list value))
      ~output:Tree.(pair value (list value))
      h
  in
  List.iter
    (fun n ->
       let ys = List.init n (fun i -> s (float_of_int i)) in
       let sum, sins = g (s 10., ys) in
       let esum, esins = h (s 10., ys) in
       Check.value (Printf.sprintf "pytree sum, %d leaves" n) ~expected:esum sum;
       Check.bool
         (Printf.sprintf "pytree list, %d leaves" n)
         (List.length sins = n && List.for_all2 Ops.allclose sins esins))
    [ 0; 2; 5 ]
;;

let () =
  let prepared = ref 0
  and calls = ref 0 in
  let module Counting = struct
    let name = "counting"

    let compile f =
      incr prepared;
      fun xs ->
        incr calls;
        f xs
    ;;
  end
  in
  let g = Compile.compile ~backend:(module Counting) f in
  ignore (g (s 1.));
  ignore (g (s 2.));
  Check.bool "backend prepares once" (!prepared = 1);
  Check.bool "backend runs per call" (!calls = 2);
  Check.value "backend result" ~expected:(f (s 2.)) (g (s 2.))
;;

let () =
  let module Broken = struct
    let name = "broken"

    let compile f xs =
      let ys, out = f xs in
      List.tl ys, out
    ;;
  end
  in
  let g = Compile.compile ~backend:(module Broken) f in
  Check.bool
    "backend that drops a leaf raises Invalid_argument"
    (match g (s 1.) with
     | _ -> false
     | exception Invalid_argument _ -> true)
;;

let calls g =
  List.filter
    (fun (e : Ir.eqn) ->
       match e.prim with
       | Compile.Call _ -> true
       | _ -> false)
    g.Ir.eqns
;;

let only_calls g = List.length (calls g) = List.length g.Ir.eqns

let () =
  let h x = Ops.(reduce_sum (sin x * x) + mean (exp x)) in
  let traces = ref 0 in
  let jh =
    Compile.compile ~backend:(Compile.jit_with ~on_trace:(fun _ -> incr traces)) h
  in
  let x = Check.rand [| 4 |]
  and dx = Check.rand ~lo:(-1.) ~hi:1. [| 4 |] in
  let g = Ir.make jh x in
  Check.bool "jit is one Call" (List.length g.eqns = 1 && only_calls g);
  Check.bool "jit: the Call shows its body" (Check.contains (Ir.to_string g) "jit:");
  let gg = Ir.make (Ad.grad jh) x in
  Check.bool "grad of jit: forward and backward calls" (List.length (calls gg) = 2);
  Check.bool "grad of jit: no inlined ops" (only_calls gg);
  Check.bool
    "grad of jit: backward is a transpose call"
    (Check.contains (Ir.to_string gg) "transpose(");
  Check.bool "grad of jit graph is well formed" (Ir.invariant gg = ());
  Check.bool
    "jvp of jit is one call"
    (only_calls (Ir.make (fun x -> snd (Ad.jvp jh x dx)) x));
  Check.bool
    "vmap of jit is one call"
    (only_calls (Ir.make (Vmap.vmap jh) (Check.rand [| 3; 4 |])));
  Check.value "jit" ~expected:(h x) (jh x);
  Check.value "grad of jit" ~expected:(Ad.grad h x) (Ad.grad jh x);
  let y, dy = Ad.jvp jh x dx
  and y', dy' = Ad.jvp h x dx in
  Check.value "jvp of jit: primal" ~expected:y' y;
  Check.value "jvp of jit: tangent" ~expected:dy' dy;
  let xs = Check.rand [| 3; 4 |] in
  Check.value "vmap of jit" ~expected:(Vmap.vmap h xs) (Vmap.vmap jh xs);
  Check.value
    "vmap of grad of jit"
    ~expected:(Vmap.vmap (Ad.grad h) xs)
    (Vmap.vmap (Ad.grad jh) xs);
  Check.value
    "grad of vmap of jit"
    ~expected:(Ad.grad (fun x -> Ops.reduce_sum (Vmap.vmap h x)) xs)
    (Ad.grad (fun x -> Ops.reduce_sum (Vmap.vmap jh x)) xs);
  Check.value "hessian of jit" ~expected:(Ad.hessian h x) (Ad.hessian jh x);
  Check.value
    "jacrev of grad of jit"
    ~expected:(Ad.jacrev (Ad.grad h) x)
    (Ad.jacrev (Ad.grad jh) x);
  Check.value
    "grad of grad of jit"
    ~expected:(Ad.grad (fun x -> Ops.reduce_sum (Ad.grad h x)) x)
    (Ad.grad (fun x -> Ops.reduce_sum (Ad.grad jh x)) x);
  Check.bool "transformations of jit do not retrace" (!traces = 1);
  ignore (jh (Check.rand [| 2; 2 |]));
  Check.bool "jit retraces for a new shape" (!traces = 2);
  Check.value
    "jit on the new shape"
    ~expected:(h (Ops.ones [| 2; 2 |]))
    (jh (Ops.ones [| 2; 2 |]));
  Check.bool "jit: no retrace for a cached shape" (!traces = 2)
;;

let () =
  let h x = Ops.(sin x * x) in
  let inner = Compile.compile ~backend:Compile.jit h in
  let outer =
    Compile.compile ~backend:Compile.jit (fun x -> Ops.(inner x + inner (x * x)))
  in
  let x = Check.rand [| 3 |] in
  let g = Ir.make outer x in
  (match calls g with
   | [ e ] ->
     (match e.prim with
      | Compile.Call { graph; _ } ->
        Check.bool "jit of jit: the inner calls stay calls" (List.length (calls graph) = 2)
      | _ -> ())
   | _ -> Check.bool "jit of jit is one call" false);
  let expected x = Ops.(h x + h (x * x)) in
  Check.value "jit of jit" ~expected:(expected x) (outer x);
  Check.value
    "grad of jit of jit"
    ~expected:(Ad.grad (fun x -> Ops.reduce_sum (expected x)) x)
    (Ad.grad (fun x -> Ops.reduce_sum (outer x)) x);
  Check.value
    "vmap of jit of jit"
    ~expected:(Vmap.vmap expected (Check.rand ~lo:1. ~hi:1. [| 2; 3 |]))
    (Vmap.vmap outer (Check.rand ~lo:1. ~hi:1. [| 2; 3 |]));
  Check.value
    "hessian of jit of jit"
    ~expected:(Ad.hessian (fun x -> Ops.reduce_sum (expected x)) x)
    (Ad.hessian (fun x -> Ops.reduce_sum (outer x)) x)
;;

let () =
  let h y x = Ops.(sin (x * y) + (y * y)) in
  let jh y = Compile.compile ~backend:Compile.jit (h y) (s 3.) in
  Check.value
    "grad of jit capturing a tracer"
    ~expected:(Ad.grad (fun y -> h y (s 3.)) (s 2.))
    (Ad.grad jh (s 2.));
  Check.value
    "deriv of jit capturing a tracer"
    ~expected:(Ad.deriv (fun y -> h y (s 3.)) (s 2.))
    (Ad.deriv jh (s 2.));
  Check.value
    "hessian of jit capturing a tracer"
    ~expected:(Ad.hessian (fun y -> h y (s 3.)) (s 2.))
    (Ad.hessian jh (s 2.));
  Check.value
    "vmap of jit capturing a tracer"
    ~expected:(Vmap.vmap (fun y -> h y (s 3.)) (Ops.arange 3))
    (Vmap.vmap jh (Ops.arange 3));
  let g = Ir.make jh (s 2.) in
  Check.bool
    "a captured tracer is an operand"
    (match calls g with
     | [ e ] -> List.length e.inputs = 2
     | _ -> false)
;;

let () =
  let leaky x =
    Control.cond
      Ops.(x > scalar 0.)
      ~then_:Ops.(fun x -> x * x)
      ~else_:Ops.(fun x -> scalar 0.1 * x)
      x
  in
  let jl = Compile.compile ~backend:Compile.jit leaky in
  List.iter
    (fun v ->
       let x = s v in
       let name what = Printf.sprintf "%s at %g" what v in
       Check.value (name "cond inside jit") ~expected:(leaky x) (jl x);
       Check.value
         (name "grad of cond inside jit")
         ~expected:(Ad.grad leaky x)
         (Ad.grad jl x);
       Check.value
         (name "hessian of cond inside jit")
         ~expected:(Ad.hessian leaky x)
         (Ad.hessian jl x))
    [ 2.; -3. ];
  let xs = Ops.array [| -2.; 0.5; 3. |] in
  Check.value "vmap of cond inside jit" ~expected:(Vmap.vmap leaky xs) (Vmap.vmap jl xs);
  let sq = Compile.compile ~backend:Compile.jit Ops.(fun x -> sin x * x) in
  let br x =
    Control.cond Ops.(x > scalar 0.) ~then_:sq ~else_:Ops.(fun x -> sq (x * x) + x) x
  in
  let br' x =
    Control.cond
      Ops.(x > scalar 0.)
      ~then_:Ops.(fun x -> sin x * x)
      ~else_:Ops.(fun x -> (sin (x * x) * x * x) + x)
      x
  in
  List.iter
    (fun v ->
       let x = s v in
       let name what = Printf.sprintf "%s at %g" what v in
       Check.value (name "jit inside cond") ~expected:(br' x) (br x);
       Check.value
         (name "grad of jit inside cond")
         ~expected:(Ad.grad br' x)
         (Ad.grad br x);
       Check.value
         (name "hessian of jit inside cond")
         ~expected:(Ad.hessian br' x)
         (Ad.hessian br x))
    [ 2.; -3. ];
  Check.value "vmap of jit inside cond" ~expected:(Vmap.vmap br' xs) (Vmap.vmap br xs)
;;

let () =
  let h x = Ops.(sin x, scalar 2. * ones [| 3 |]) in
  let jh =
    Compile.compile_tree
      ~input:Tree.value
      ~output:Tree.(pair value value)
      ~backend:Compile.jit
      h
  in
  let x = Check.rand [| 3 |] in
  let (_, c), (_, dc) =
    Ad.jvp_tree ~input:Tree.value ~output:Tree.(pair value value) jh x x
  in
  Check.value "jit: constant output" ~expected:(Ops.full [| 3 |] 2.) c;
  Check.value "jvp of jit: zero tangent" ~expected:(Ops.zeros [| 3 |]) dc;
  let a, b =
    Vmap.vmap_tree
      ~input:Tree.value
      ~output:Tree.(pair value value)
      jh
      (Check.rand [| 2; 3 |])
  in
  Check.bool "vmap of jit: shapes" (Ops.shape a = [| 2; 3 |] && Ops.shape b = [| 2; 3 |]);
  Check.value
    "grad of jit with a constant output"
    ~expected:(Ops.cos x)
    (Ad.grad (fun x -> Ops.(reduce_sum (fst (jh x) + snd (jh x)))) x)
;;

let () = Check.finish ()
