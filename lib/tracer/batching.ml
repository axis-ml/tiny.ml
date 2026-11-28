open Common
open Lang

module Batching = struct
  open Interp
  module B = Builtin
  module Ir = Staging.Ir
  module C = Staging.Control
  module U = Staging.Util

  type rule =
    Prim.t -> size:int -> value list -> int option list -> value list * int option list

  let rules : rule Prim.Table.t = Prim.Table.create "vmap"
  let register_rule p rule = Prim.Table.add rules p rule
  let missing_rules () = Prim.Table.missing rules

  type batched =
    { value : value
    ; bdim : int
    }

  let move_axis ~from ~to_ x =
    if from = to_
    then x
    else (
      let rest = List.filter (( <> ) from) (Lists.range (Ops.rank x)) in
      let before, after = Lists.split_at to_ rest in
      Ops.transpose ~perm:(Array.of_list (before @ [ from ] @ after)) x)
  ;;

  let to_front ~size x = function
    | Some d -> move_axis ~from:d ~to_:0 x
    | None -> Ops.broadcast ~shape:(Array.append [| size |] (Ops.shape x)) ~axes:[ 0 ] x
  ;;

  let front_all ~size xs ds = List.map2 (to_front ~size) xs ds
  let at_front xs = List.map (fun _ -> Some 0) xs

  let unbatched_aval v bdim =
    let a = aval v in
    Aval.create ~dtype:a.dtype (Shape.remove_dims ~axes:[ bdim ] a.shape)
  ;;

  let unpack self v =
    match payload self v with
    | Some { value; bdim } -> value, Some bdim
    | None -> v, None
  ;;

  let pack self value = function
    | None -> value
    | Some bdim -> tracer self (unbatched_aval value bdim) { value; bdim }
  ;;

  let check_types prim ~size xs ds outs douts =
    let element_aval x = function
      | None -> aval x
      | Some d when (aval x).shape.(d) <> size ->
        Error.type_error
          "vmap rule for %s: an output has size %d at its batch dim, not %d"
          (Prim.name prim)
          (aval x).shape.(d)
          size
      | Some d -> unbatched_aval x d
    in
    let in_avals =
      List.map2 (fun x d -> Option.fold ~none:(aval x) ~some:(unbatched_aval x) d) xs ds
    in
    if
      not
        (List.equal
           Aval.equal
           (List.map2 element_aval outs douts)
           (Prim.abstract_eval prim in_avals))
    then
      Error.type_error
        "vmap rule for %s: outputs do not have the types abstract_eval gives"
        (Prim.name prim)
  ;;

  let process ~size self prim args =
    let xs, ds = List.split (List.map (unpack self) args) in
    if List.for_all Option.is_none ds
    then bind prim xs
    else (
      let outs, douts = (Prim.Table.find rules prim) prim ~size xs ds in
      if List.length outs <> List.length douts
      then
        Error.type_error
          "vmap rule for %s returned %d outputs and %d batch dims"
          (Prim.name prim)
          (List.length outs)
          (List.length douts);
      if !Debug.enabled then check_types prim ~size xs ds outs douts;
      List.map2 (pack self) outs douts)
  ;;

  let check_batch_dim ~size x = function
    | Some d when d < 0 || d >= Ops.rank x || (Ops.shape x).(d) <> size ->
      Error.type_error "vmap: %s has no axis %d of size %d" (describe x) d size
    | Some _ | None -> ()
  ;;

  let vmap_flat ~size f xs ds =
    List.iter2 (check_batch_dim ~size) xs ds;
    let self = create ~name:"vmap" (process ~size) in
    let outs, aux = handle self (fun () -> f (List.map2 (pack self) xs ds)) in
    let ys, dys = List.split (List.map (unpack self) outs) in
    ys, dys, aux
  ;;

  let run_front ~size g xs ds =
    let ys, dys, () = vmap_flat ~size (U.as_fn g) xs ds in
    front_all ~size ys dys
  ;;

  let batch_graph ~size (g : Ir.t) xs ds =
    U.traced (fun xs -> run_front ~size g xs ds) (List.map aval xs)
  ;;

  let select_front mask a b =
    let s = Ops.shape a in
    let mask = Ops.broadcast ~shape:s ~axes:(List.init (Array.length s - 1) succ) mask in
    bind1 B.Select [ mask; a; b ]
  ;;

  let shift = List.map succ
  let front_result y = [ y ], [ Some 0 ]

  let unary f : rule =
    fun p ~size xs ds ->
    match xs, ds with
    | [ x ], [ d ] -> f p ~size x d
    | _ -> Prim.arity_error p ~expected:1 xs
  ;;

  let binary f : rule =
    fun p ~size xs ds ->
    match xs, ds with
    | [ x; y ], [ dx; dy ] -> f p ~size x dx y dy
    | _ -> Prim.arity_error p ~expected:2 xs
  ;;

  let elementwise_rule p ~size xs ds =
    match ds with
    | [ d ] -> [ bind1 p xs ], [ d ]
    | _ -> [ bind1 p (front_all ~size xs ds) ], [ Some 0 ]
  ;;

  let elementwise p = register_rule p elementwise_rule

  let front1 p f =
    register_rule
      p
      (unary (fun p ~size x d -> front_result (bind1 (f p) [ to_front ~size x d ])))
  ;;

  let shape_rules () =
    let open Ops in
    front1 (B.Reduce_sum []) (function
      | B.Reduce_sum axes -> B.Reduce_sum (shift axes)
      | p -> p);
    front1 (B.Reduce_max []) (function
      | B.Reduce_max axes -> B.Reduce_max (shift axes)
      | p -> p);
    front1 (B.Transpose [||]) (function
      | B.Transpose perm -> B.Transpose (Array.append [| 0 |] (Array.map succ perm))
      | p -> p);
    front1
      (B.Slice { axis = 0; start = 0; stop = 0 })
      (function
        | B.Slice s -> B.Slice { s with axis = succ s.axis }
        | p -> p);
    register_rule
      (B.Broadcast { shape = [||]; axes = [] })
      (unary (fun p ~size x d ->
         match p with
         | B.Broadcast { shape; axes } ->
           front_result
             (bind1
                (B.Broadcast { shape = Array.append [| size |] shape; axes = shift axes })
                [ to_front ~size x d ])
         | _ -> Prim.arity_error p ~expected:1 [ x ]));
    register_rule
      (B.Reshape [||])
      (unary (fun p ~size x d ->
         match p with
         | B.Reshape shape ->
           front_result
             (reshape ~shape:(Array.append [| size |] shape) (to_front ~size x d))
         | _ -> Prim.arity_error p ~expected:1 [ x ]));
    register_rule (B.Concatenate 0) (fun p ~size xs ds ->
      match p with
      | B.Concatenate axis ->
        front_result (bind1 (B.Concatenate (succ axis)) (front_all ~size xs ds))
      | _ -> Prim.arity_error p ~expected:1 xs)
  ;;

  let dot_rule () =
    let open Ops in
    let mm x y = bind1 B.Dot [ x; y ] in
    let batched_left ~size x d y =
      let x = move_axis ~from:d ~to_:0 x in
      let s = shape x
      and n = (shape y).(1) in
      front_result
        (reshape
           ~shape:[| size; s.(1); n |]
           (mm (reshape ~shape:[| Int.mul size s.(1); s.(2) |] x) y))
    in
    let batched_right ~size x y d =
      let y = move_axis ~from:d ~to_:1 y in
      let s = shape y
      and m = (shape x).(0) in
      ( [ reshape
            ~shape:[| m; size; s.(2) |]
            (mm x (reshape ~shape:[| s.(0); Int.mul size s.(2) |] y))
        ]
      , [ Some 1 ] )
    in
    let batched_both ~size x dx y dy =
      let x = to_front ~size x dx
      and y = to_front ~size y dy in
      let sx = shape x
      and sy = shape y in
      let full = [| size; sx.(1); sx.(2); sy.(2) |] in
      let x = broadcast ~shape:full ~axes:[ 3 ] x
      and y = broadcast ~shape:full ~axes:[ 1 ] y in
      front_result (reduce_sum ~axes:[ 2 ] (mul x y))
    in
    register_rule
      B.Dot
      (binary (fun _ ~size x dx y dy ->
         match dx, dy with
         | Some d, None -> batched_left ~size x d y
         | None, Some d -> batched_right ~size x y d
         | _ -> batched_both ~size x dx y dy))
  ;;

  let call_rule p ~size xs ds =
    match p with
    | Staging.Compile.Call { name; graph } ->
      let g, dims =
        U.traced_aux
          (fun xs ->
             let ys, dys, () = vmap_flat ~size (U.as_fn graph) xs ds in
             ys, dys)
          (List.map aval xs)
      in
      bind (Staging.Compile.Call { name = "vmap(" ^ name ^ ")"; graph = g }) xs, dims
    | _ -> Prim.arity_error p ~expected:0 xs
  ;;

  let cond_rule p ~size xs ds =
    match p, xs, ds with
    | C.Cond { then_; else_ }, pred :: rest, dpred :: drest ->
      let nt = List.length then_.params in
      let at, ae = Lists.split_at nt rest
      and dt, de = Lists.split_at nt drest in
      let outs =
        match dpred with
        | None ->
          bind
            (C.Cond
               { then_ = batch_graph ~size then_ at dt
               ; else_ = batch_graph ~size else_ ae de
               })
            ((pred :: at) @ ae)
        | Some d ->
          let mask = Ops.greater (move_axis ~from:d ~to_:0 pred) (Ops.scalar 0.) in
          let yt = run_front ~size then_ at dt
          and ye = run_front ~size else_ ae de in
          List.map2 (select_front mask) yt ye
      in
      outs, at_front outs
    | _ -> Prim.arity_error p ~expected:1 xs
  ;;

  let while_rule p ~size xs ds =
    match p with
    | C.While { cond; body } ->
      let n = List.length body.outputs in
      let nc = List.length cond.params - n
      and nb = List.length body.params - n in
      let ac, rest = Lists.split_at nc xs
      and dac, drest = Lists.split_at nc ds in
      let ab, carry = Lists.split_at nb rest
      and dab, dcarry = Lists.split_at nb drest in
      let carry = front_all ~size carry dcarry in
      let front = at_front carry in
      let still_running ac carry =
        match run_front ~size cond (ac @ carry) (dac @ front) with
        | [ pred ] -> Ops.greater pred (Ops.scalar 0.)
        | _ -> failwith "vmap of while_loop: cond must return one value"
      in
      let carry_avals = List.map aval carry in
      let cond' =
        U.traced
          (fun l ->
             let ac, carry = Lists.split_at nc l in
             [ Ops.reduce_max (still_running ac carry) ])
          (List.map aval ac @ carry_avals)
      in
      let body' =
        U.traced
          (fun l ->
             let ac, l = Lists.split_at nc l in
             let ab, carry = Lists.split_at nb l in
             let running = still_running ac carry in
             let next = run_front ~size body (ab @ carry) (dab @ front) in
             List.map2 (select_front running) next carry)
          (List.map aval ac @ List.map aval ab @ carry_avals)
      in
      bind (C.While { cond = cond'; body = body' }) (ac @ ac @ ab @ carry), front
    | _ -> Prim.arity_error p ~expected:0 xs
  ;;

  let () =
    List.iter
      elementwise
      B.
        [ Add
        ; Sub
        ; Mul
        ; Div
        ; Max
        ; Greater
        ; Less
        ; Neg
        ; Sin
        ; Cos
        ; Exp
        ; Log
        ; Sqrt
        ; Integer_pow 0
        ; Select
        ];
    shape_rules ();
    dot_rule ();
    register_rule (Staging.Compile.Call { name = ""; graph = U.empty }) call_rule;
    register_rule (C.Cond { then_ = U.empty; else_ = U.empty }) cond_rule;
    register_rule (C.While { cond = U.empty; body = U.empty }) while_rule
  ;;
end

module Vmap = struct
  module Bt = Batching

  let normalize_axis ~rank a = if a < 0 then a + rank else a

  let batch_dims ?in_axes ~in_axis xs =
    match in_axes with
    | None -> List.map (fun x -> Some (normalize_axis ~rank:(Ops.rank x) in_axis)) xs
    | Some ds when List.length ds <> List.length xs ->
      Error.type_error "vmap: %d in_axes for %d leaves" (List.length ds) (List.length xs)
    | Some ds ->
      List.map2 (fun x d -> Option.map (normalize_axis ~rank:(Ops.rank x)) d) xs ds
  ;;

  let batch_size xs ds =
    let sizes =
      List.filter_map
        (fun (x, d) ->
           match d with
           | Some d when d >= 0 && d < Ops.rank x -> Some (Ops.shape x).(d)
           | Some d -> Error.type_error "vmap: %s has no axis %d" (Ops.describe x) d
           | None -> None)
        (List.combine xs ds)
    in
    match sizes with
    | [] -> Error.type_error "vmap: no input is batched"
    | n :: rest when List.exists (( <> ) n) rest ->
      Error.type_error
        "vmap: batched inputs have different sizes (%s)"
        (String.concat ", " (List.map string_of_int sizes))
    | n :: _ -> n
  ;;

  let vmap_tree
        ?in_axes
        ?(in_axis = 0)
        ?(out_axis = 0)
        ~input:in_tree
        ~output:out_tree
        f
        x
    =
    let xs, s = Tree.flatten in_tree x in
    let ds = batch_dims ?in_axes ~in_axis xs in
    let size = batch_size xs ds in
    let ys, dys, out = Bt.vmap_flat ~size (Tree.flat_fn in_tree out_tree f s) xs ds in
    let place y d =
      let y = Bt.to_front ~size y d in
      Bt.move_axis ~from:0 ~to_:(normalize_axis ~rank:(Ops.rank y) out_axis) y
    in
    Tree.unflatten out_tree out (List.map2 place ys dys)
  ;;

  let vmap ?in_axis ?out_axis f x =
    vmap_tree ?in_axis ?out_axis ~input:Tree.value ~output:Tree.value f x
  ;;
end
