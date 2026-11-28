open Tracer
open Expert

let s = Ops.scalar

let f x =
  let open Ops in
  let y = sin x * scalar 2. in
  let z = -y + x in
  z
;;

let () = Random.init 42
let rand = Check.rand
let un = Check.un
let bin = Check.bin

let () =
  Check.float
    ~rtol:1e-15
    "f 3.0 = 2.7177599838802657"
    ~expected:2.7177599838802657
    (Ops.to_float (f (s 3.)));
  let y, dy = Ad.jvp f (s 3.) (s 1.) in
  Check.float "jvp f 3.0: primal" ~expected:2.7177599838802657 (Ops.to_float y);
  Check.float
    "jvp f 3.0: tangent = 1 - 2 cos 3"
    ~expected:2.979984993200891
    (Ops.to_float dy);
  let d = Ad.deriv
  and sin = Ops.sin
  and x = s 3. in
  Check.float "deriv sin = cos" ~expected:(Float.cos 3.) (Ops.to_float (d sin x));
  Check.float "deriv^2 sin = -sin" ~expected:(-.Float.sin 3.) (Ops.to_float (d (d sin) x));
  Check.float
    "deriv^3 sin = -cos"
    ~expected:(-.Float.cos 3.)
    (Ops.to_float (d (d (d sin)) x));
  Check.float
    "deriv^4 sin = sin"
    ~expected:(Float.sin 3.)
    (Ops.to_float (d (d (d (d sin))) x));
  let pc x =
    let g _ = x in
    let should_be_zero = Ad.deriv g (s 0.) in
    Ops.mul x should_be_zero
  in
  Check.float
    "perturbation confusion (Autodidax2) = 0"
    ~expected:0.
    (Ops.to_float (Ad.deriv pc (s 0.)));
  let sp x = Ops.mul x (Ad.deriv (fun y -> Ops.add x y) (s 1.)) in
  Check.float
    "perturbation confusion (Siskind-Pearlmutter) = 1"
    ~expected:1.
    (Ops.to_float (Ad.deriv sp (s 1.)));
  let a () = rand [| 2; 3 |] in
  un "neg" Ops.neg (a ());
  un "sin" Ops.sin (a ());
  un "cos" Ops.cos (a ());
  un "exp" Ops.exp (a ());
  un "log" Ops.log (a ());
  bin "add" Ops.add (a ()) (a ());
  bin "sub" Ops.sub (a ()) (a ());
  bin "mul" Ops.mul (a ()) (a ());
  bin "div" Ops.div (a ()) (a ());
  bin "greater" Ops.greater (a ()) (a ());
  bin "less" Ops.less (a ()) (a ());
  un "reduce_sum axes=[0]" (Ops.reduce_sum ~axes:[ 0 ]) (a ());
  un "reduce_sum axes=[-1]" (Ops.reduce_sum ~axes:[ -1 ]) (a ());
  un "reduce_sum all" (fun x -> Ops.reduce_sum x) (a ());
  un "broadcast" (Ops.broadcast ~shape:[| 2; 4; 3 |] ~axes:[ 1 ]) (a ());
  un "transpose" (fun x -> Ops.transpose x) (a ());
  un "transpose 3D" (Ops.transpose ~perm:[| 2; 0; 1 |]) (rand [| 2; 3; 4 |]);
  bin "scalar * array (auto broadcast)" Ops.mul (rand [||]) (a ());
  bin "row + matrix (auto broadcast)" Ops.add (rand [| 3 |]) (a ());
  un "composite" Ops.(fun x -> reduce_sum (sin x * x / exp x)) (a ());
  Check.raises
    ~matches:Check.is_type_error
    "jvp rejects tangent of wrong shape"
    (fun () -> Ad.jvp f (s 3.) (Ops.ones [| 2 |]));
  Check.raises
    ~matches:(function
      | Error (Type_error m) ->
        m
        = "add: cannot broadcast f64[2] and f64[3] together (dims must match from the \
           right, or be 1)"
      | _ -> false)
    "binop rejects incompatible shapes, naming the op"
    (fun () -> Ops.add (Ops.zeros [| 2 |]) (Ops.zeros [| 3 |]));
  Check.bool
    "uncaught errors print without the module path"
    (Printexc.to_string (Error (Type_error "boom")) = "Type_error: boom");
  Check.raises "exception inside jvp propagates" (fun () ->
    Ad.deriv (fun _ -> failwith "boom") (s 1.));
  Check.bool
    "current interpreter restored after exception"
    (Interp.stack_to_string () = "eval");
  let unwound = ref false in
  Check.raises
    ~matches:Check.is_type_error
    "error inside a jvp rule propagates"
    (fun () ->
       Ad.deriv
         (fun x ->
            Fun.protect
              ~finally:(fun () -> unwound := true)
              (fun () -> Ops.reduce_sum ~axes:[ 5 ] x))
         (s 1.));
  Check.bool "error inside a jvp rule unwinds the user function" !unwound;
  Check.bool
    "current interpreter restored after an error inside a rule"
    (Interp.stack_to_string () = "eval");
  let leaked = ref (s 0.) in
  ignore
    (Ad.deriv
       (fun x ->
          leaked := x;
          x)
       (s 1.));
  Check.raises
    ~matches:Check.is_escaped
    "escaped jvp tracer is rejected by eval"
    (fun () -> Ops.sin !leaked);
  let m = Ops.full [| 1000 |] 2. in
  let ops = ref [] in
  let counter =
    Interp.create ~name:"count" (fun _ prim args ->
      ops := Prim.name prim :: !ops;
      Interp.bind prim args)
  in
  ignore
    (Interp.handle counter (fun () ->
       Ad.jvp
         (fun x -> Ops.add (Ops.mul x m) m)
         (Ops.ones [| 1000 |])
         (Ops.ones [| 1000 |])));
  Check.bool
    "jvp of x * m + m binds only mul, mul, add: no zero tangent work"
    (List.sort compare !ops = [ "add"; "mul"; "mul" ]);
  Check.finish ()
;;
