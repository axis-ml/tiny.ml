open Tracer
open Expert

let s = Ops.scalar
let () = Random.init 11
let pos x = Ops.(x > scalar 0.)
let f x = Control.cond (pos x) ~then_:Ops.(fun x -> x * x * x) ~else_:Ops.sin x
let f_ref x = if Ops.to_float x > 0. then Ops.(x * x * x) else Ops.sin x

let () =
  List.iter
    (fun v ->
       let x = s v in
       let name fmt = Printf.sprintf fmt v in
       Check.value (name "cond %g") ~expected:(f_ref x) (f x);
       Check.value (name "deriv cond %g") ~expected:(Ad.deriv f_ref x) (Ad.deriv f x);
       Check.value (name "grad cond %g") ~expected:(Ad.grad f_ref x) (Ad.grad f x);
       Check.value
         (name "grad grad cond %g")
         ~expected:(Ad.grad (Ad.grad f_ref) x)
         (Ad.grad (Ad.grad f) x);
       Check.value
         (name "hessian cond %g")
         ~expected:(Ad.hessian f_ref x)
         (Ad.hessian f x))
    [ 1.5; -0.7 ]
;;

let () =
  let traces = ref 0 in
  let fast =
    Compile.compile ~backend:(Compile.jit_with ~on_trace:(fun _ -> incr traces)) f
  in
  Check.value "jit cond, then" ~expected:(f_ref (s 2.)) (fast (s 2.));
  Check.value "jit cond, else" ~expected:(f_ref (s (-2.))) (fast (s (-2.)));
  Check.bool "jit cond: one trace for both branches" (!traces = 1);
  Check.value
    "grad of jit cond"
    ~expected:(Ad.grad f_ref (s (-2.)))
    (Ad.grad fast (s (-2.)));
  let jg = Compile.compile ~backend:Compile.jit (Ad.grad f) in
  Check.value "jit of grad cond" ~expected:(Ad.grad f_ref (s 2.)) (jg (s 2.));
  Check.value "jit of grad cond, else" ~expected:(Ad.grad f_ref (s (-2.))) (jg (s (-2.)));
  let g = Ir.make f (s 1.) in
  Check.bool
    "cond is one eqn"
    (List.map (fun (e : Ir.eqn) -> Prim.name e.prim) g.eqns |> List.mem "cond");
  let str = Ir.to_string g in
  Check.bool
    "to_string shows branches"
    (String.length str > 0
     && List.exists (fun l -> String.trim l = "then:") (String.split_on_char '\n' str));
  Check.bool "to_dot has clusters" (Check.contains (Ir.to_dot g) "cluster_")
;;

let () =
  let h y x =
    Control.cond (pos x) ~then_:(fun x -> Ops.(x * y)) ~else_:(fun x -> Ops.(x + y)) x
  in
  Check.value
    "deriv wrt captured, then"
    ~expected:(s 3.)
    (Ad.deriv (fun y -> h y (s 3.)) (s 2.));
  Check.value
    "deriv wrt captured, else"
    ~expected:(s 1.)
    (Ad.deriv (fun y -> h y (s (-3.))) (s 2.));
  Check.value
    "grad wrt captured, then"
    ~expected:(s 3.)
    (Ad.grad (fun y -> h y (s 3.)) (s 2.));
  Check.value
    "grad wrt captured, else"
    ~expected:(s 1.)
    (Ad.grad (fun y -> h y (s (-3.))) (s 2.));
  Check.value
    "grad through jit, captured"
    ~expected:(s 3.)
    (Ad.grad (fun y -> Compile.compile ~backend:Compile.jit (h y) (s 3.)) (s 2.))
;;

let () =
  let x = Check.rand [| 3 |]
  and w = Check.rand [| 3 |] in
  let swap (a, b) = b, a in
  let k p (a, b) =
    Control.cond_tree
      ~input:Tree.(pair value value)
      ~output:Tree.(pair value value)
      p
      ~then_:swap
      ~else_:(fun (a, b) -> Ops.(a * b, a + b))
      (a, b)
  in
  let a, b = k (s 1.) (x, w) in
  Check.value "tree cond fst" ~expected:w a;
  Check.value "tree cond snd" ~expected:x b;
  let a, _ = k (s 0.) (x, w) in
  Check.value "tree cond else" ~expected:(Ops.mul x w) a;
  let loss (x, w) = Ops.reduce_sum (fst (k (Ops.reduce_sum x) (x, w))) in
  let gx, gw = Ad.grad_tree ~wrt:Tree.(pair value value) loss (x, w) in
  Check.value "tree cond grad x" ~expected:(Ops.zeros [| 3 |]) gx;
  Check.value "tree cond grad w" ~expected:(Ops.ones [| 3 |]) gw;
  Check.fd_check
    "jvp tree cond"
    (function
      | [ x; w ] -> [ fst (k Ops.(reduce_sum x - scalar 4.) (x, w)) ]
      | _ -> assert false)
    [ x; w ];
  Check.raises ~matches:Check.is_type_error "branches of different types" (fun () ->
    Control.cond (s 1.) ~then_:Fun.id ~else_:(fun x -> Ops.reduce_sum x) x);
  Check.raises ~matches:Check.is_type_error "array predicate" (fun () ->
    Control.cond x ~then_:Fun.id ~else_:Fun.id x)
;;

let () =
  let grow x =
    Control.while_loop
      ~cond:Ops.(fun c -> c < scalar 10.)
      ~body:Ops.(fun c -> c * scalar 2.)
      x
  in
  Check.value "while" ~expected:(s 12.) (grow (s 1.5));
  Check.value "while, no iteration" ~expected:(s 11.) (grow (s 11.));
  Check.value "deriv while" ~expected:(s 8.) (Ad.deriv grow (s 1.5));
  Check.value
    "jacfwd while"
    ~expected:(Ops.eye 1)
    (Ad.jacfwd
       (fun x -> Ops.reshape ~shape:[| 1 |] (grow (Ops.reshape ~shape:[||] x)))
       (Ops.array [| 20. |]));
  let traces = ref 0 in
  let fast =
    Compile.compile ~backend:(Compile.jit_with ~on_trace:(fun _ -> incr traces)) grow
  in
  Check.value "jit while" ~expected:(s 12.) (fast (s 1.5));
  Check.value "jit while again" ~expected:(s 12.8) (fast (s 0.1));
  Check.bool "jit while: one trace" (!traces = 1);
  Check.value "deriv of jit while" ~expected:(s 128.) (Ad.deriv fast (s 0.1));
  Check.raises ~matches:Check.is_missing_rule "grad while: no reverse mode" (fun () ->
    Ad.grad grow (s 1.5));
  let pow3 y =
    let _, acc =
      Control.while_loop_tree
        ~carry:Tree.(pair value value)
        ~cond:(fun (i, _) -> Ops.(i < scalar 3.))
        ~body:(fun (i, acc) -> Ops.(i + scalar 1., acc * y))
        (s 0., s 1.)
    in
    acc
  in
  Check.value "while, captured" ~expected:(s 8.) (pow3 (s 2.));
  Check.value "deriv while wrt captured" ~expected:(s 12.) (Ad.deriv pow3 (s 2.));
  Check.value "deriv deriv while" ~expected:(s 12.) (Ad.deriv (Ad.deriv pow3) (s 2.));
  Check.bool
    "while graph is well formed"
    (match Ir.invariant (Ir.make pow3 (s 2.)) with
     | () -> true
     | exception Failure _ -> false);
  Check.raises ~matches:Check.is_type_error "while body changes the type" (fun () ->
    Control.while_loop ~cond:(fun _ -> s 0.) ~body:(fun c -> Ops.stack [ c; c ]) (s 1.))
;;

let () = Check.finish ()
