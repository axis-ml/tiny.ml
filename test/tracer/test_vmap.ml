open Tracer
open Expert

let s = Ops.scalar
let () = Random.init 5
let rand = Check.rand

let loop ?(axis = 0) ?(out = 0) f x =
  let n = (Ops.shape x).(axis) in
  let at i = Ops.slice ~axis ~start:i ~stop:(i + 1) x |> Ops.squeeze ~axes:[ axis ] in
  List.init n (fun i -> at i |> f) |> Ops.stack ~axis:out
;;

let same ?(axis = 0) ?(out = 0) name f x =
  Check.value
    ~rtol:1e-10
    ~atol:1e-10
    name
    ~expected:(loop ~axis ~out f x)
    (Vmap.vmap ~in_axis:axis ~out_axis:out f x)
;;

let () =
  let m = rand [| 4; 3 |]
  and v = rand [| 3 |] in
  let cases =
    Ops.
      [ ("add", [| 3 |], fun x -> x + v)
      ; ("add, broadcast", [| 3 |], fun x -> x + m)
      ; ("sub, scalar", [| 3 |], fun x -> scalar 2. - x)
      ; ("mul self", [| 4; 3 |], fun x -> x * x)
      ; ("div", [| 4; 3 |], fun x -> x / m)
      ; ("maximum", [| 3 |], fun x -> maximum x (scalar 1.))
      ; ("greater", [| 3 |], fun x -> x > v)
      ; ("less", [| 3 |], fun x -> x < v)
      ; ("select", [| 3 |], fun x -> select (x > scalar 1.) x v)
      ; ("neg sin cos", [| 3 |], fun x -> neg (sin x) + cos x)
      ; ("exp log sqrt", [| 3 |], fun x -> exp x + log x + sqrt x)
      ; "integer_pow", [| 3 |], integer_pow ~exponent:3
      ; ("reduce_sum all", [| 4; 3 |], fun x -> reduce_sum x)
      ; ("reduce_sum axis", [| 4; 3 |], fun x -> reduce_sum ~axes:[ 1 ] x)
      ; ("reduce_max axis", [| 4; 3 |], fun x -> reduce_max ~axes:[ 0 ] x)
      ; ("broadcast", [| 3 |], fun x -> broadcast ~shape:[| 2; 3; 5 |] ~axes:[ 0; 2 ] x)
      ; ("transpose", [| 4; 3 |], fun x -> transpose x)
      ; ("reshape", [| 4; 3 |], fun x -> reshape ~shape:[| 2; 6 |] x)
      ; ("concatenate", [| 4; 3 |], fun x -> concatenate ~axis:1 [ m; x; m ])
      ; ("stack", [| 3 |], fun x -> stack [ x; v ])
      ; ("slice", [| 4; 3 |], fun x -> slice ~axis:1 ~start:1 ~stop:3 x)
      ; ("dot vv", [| 3 |], fun x -> dot x v)
      ; ("dot mv", [| 4; 3 |], fun x -> dot x v)
      ; ("dot vm", [| 4 |], fun x -> dot x m)
      ; ("dot m x", [| 3; 2 |], fun x -> dot m x)
      ; ("dot x m", [| 2; 4 |], fun x -> dot x m)
      ; ("dot x x", [| 3; 3 |], fun x -> dot x (transpose x))
      ; ("unbatched output", [| 3 |], fun _ -> v)
      ]
  in
  cases
  |> List.iter (fun (name, shape, f) ->
    let n = 5 in
    let r = Array.length shape in
    same ("vmap " ^ name) f (Array.append [| n |] shape |> rand);
    same
      ~axis:r
      ~out:(-1)
      ("vmap last axis " ^ name)
      f
      (Array.append shape [| n |] |> rand))
;;

let () =
  let x = rand [| 5; 3 |]
  and w = rand [| 3 |] in
  let f (x, w) = Ops.(x * w |> reduce_sum) in
  let p = Tree.(pair value value) in
  Check.value
    "in_axes: shared leaf"
    ~expected:(loop (fun x -> f (x, w)) x)
    (Vmap.vmap_tree ~in_axes:[ Some 0; None ] ~input:p ~output:Tree.value f (x, w))
;;

let () =
  let w = rand [| 3; 3 |] in
  let f x = Ops.(tanh (dot w x) * x |> reduce_sum) in
  let xs = rand [| 6; 3 |] in
  same "vmap of grad: per-example gradients" (Ad.grad f) xs;
  Check.value
    "grad of sum of vmap"
    ~expected:(Ad.grad (fun xs -> loop f xs |> Ops.reduce_sum) xs)
    (Ad.grad (fun xs -> Vmap.vmap f xs |> Ops.reduce_sum) xs);
  let dxs = rand [| 6; 3 |] in
  let y, dy = Ad.jvp (Vmap.vmap f) xs dxs in
  let y', dy' = Ad.jvp (loop f) xs dxs in
  Check.value "jvp of vmap: primal" ~expected:y' y;
  Check.value "jvp of vmap: tangent" ~expected:dy' dy;
  same "vmap of hessian" (Ad.hessian f) xs;
  let xss = rand [| 2; 6; 3 |] in
  same "vmap of vmap" (Vmap.vmap f) xss;
  same ~axis:1 "vmap of vmap, inner axis 1" (Vmap.vmap ~in_axis:0 f) xss;
  let traces = ref 0 in
  let jit f =
    Compile.compile ~backend:(Compile.jit_with ~on_trace:(fun _ -> incr traces)) f
  in
  let jv = jit (Vmap.vmap f) in
  Check.value "jit of vmap" ~expected:(loop f xs) (jv xs);
  Check.value "jit of vmap again" ~expected:(loop f dxs) (jv dxs);
  Check.bool "jit of vmap: one trace" (!traces = 1);
  same "vmap of jit" (jit f) xs;
  same "vmap of jit of grad" (jit (Ad.grad f)) xs
;;

let () =
  let a = rand [| 4 |]
  and b = rand [| 3 |] in
  let outer_product a = Vmap.vmap (fun x -> Vmap.vmap (fun y -> Ops.(x * y)) b) a in
  let outer = outer_product a in
  Check.value "nested vmap: outer product" ~expected:(loop (fun x -> Ops.(x * b)) a) outer;
  Check.value
    "grad through nested vmap"
    ~expected:Ops.(reduce_sum b |> broadcast_to ~shape:[| 4 |])
    (Ad.grad (fun a -> outer_product a |> Ops.reduce_sum) a)
;;

let () =
  let pos x = Ops.(x > scalar 0.) in
  let f x = Control.cond (pos x) ~then_:Ops.(fun x -> x * x * x) ~else_:Ops.sin x in
  let xs = Ops.array [| 1.5; -0.7; 2.; -3. |] in
  same "vmap cond, batched predicate" f xs;
  same "vmap grad cond, batched predicate" (Ad.grad f) xs;
  Check.value
    "grad of vmap cond"
    ~expected:(loop (Ad.grad f) xs)
    (Ad.grad (fun xs -> Vmap.vmap f xs |> Ops.reduce_sum) xs);
  let g p x = Control.cond p ~then_:Ops.(fun x -> x * scalar 2.) ~else_:Ops.neg x in
  same "vmap cond, shared predicate, then" (g (s 1.)) xs;
  same "vmap cond, shared predicate, else" (g (s (-1.))) xs;
  same "jit of vmap cond" (Compile.compile ~backend:Compile.jit f) xs;
  let grow x =
    Control.while_loop
      ~cond:Ops.(fun c -> c < scalar 10.)
      ~body:Ops.(fun c -> c * scalar 2.)
      x
  in
  let xs = Ops.array [| 1.5; 11.; 0.1; 6. |] in
  same "vmap while, per-element trip counts" grow xs;
  same "vmap deriv while" (Ad.deriv grow) xs;
  let pow y =
    Control.while_loop_tree
      ~carry:Tree.(pair value value)
      ~cond:(fun (i, _) -> Ops.(i < scalar 3.))
      ~body:(fun (i, acc) -> Ops.(i + scalar 1., acc * y))
      (s 0., s 1.)
    |> snd
  in
  same "vmap while, batched capture" pow xs;
  let count n =
    Control.while_loop_tree
      ~carry:Tree.(pair value value)
      ~cond:(fun (i, _) -> Ops.(i < n))
      ~body:(fun (i, acc) -> Ops.(i + scalar 1., acc * scalar 2.))
      (s 0., s 1.)
    |> fst
  in
  same "vmap while, batched bound" count (Ops.array [| 0.; 2.; 5. |])
;;

let () =
  let missing = Batching.missing_rules () in
  Check.bool
    "every builtin, cond, while and call have a vmap rule"
    ("cond" :: "while" :: "call" :: List.map Prim.name Builtin.all
     |> List.for_all (fun n -> not (List.mem n missing)));
  Check.raises ~matches:Check.is_type_error "vmap: different batch sizes" (fun () ->
    Vmap.vmap_tree
      ~input:Tree.(pair value value)
      ~output:Tree.value
      (fun (a, b) -> Ops.(a + b))
      (rand [| 3 |], rand [| 4 |]));
  Check.raises ~matches:Check.is_type_error "vmap: no batched input" (fun () ->
    Vmap.vmap_tree
      ~in_axes:[ None ]
      ~input:Tree.value
      ~output:Tree.value
      Fun.id
      (rand [| 3 |]));
  Check.raises ~matches:Check.is_concretization "vmap: no concrete value" (fun () ->
    Vmap.vmap (fun x -> Ops.to_float x |> s) (rand [| 3 |]))
;;

let () = Check.finish ()
