open Common
open Lang

module Maybe_zero = struct
  type t =
    | Zero of Aval.t
    | Nonzero of Interp.value

  let zero_like v = Zero (Interp.aval v)
  let nonzero v = Nonzero v

  let is_zero = function
    | Zero _ -> true
    | Nonzero _ -> false
  ;;

  let aval = function
    | Zero a -> a
    | Nonzero v -> Interp.aval v
  ;;

  let check ~rule ~what ~of_ prim expected got =
    List.iter2
      (fun e g ->
         if not (Aval.equal (aval g) e)
         then
           Error.type_error
             "%s rule for %s: %s %s for %s %s"
             rule
             (Prim.name prim)
             what
             (Aval.to_string (aval g))
             of_
             (Aval.to_string e))
      expected
      got
  ;;

  let instantiate = function
    | Zero a -> Ops.zeros a.shape
    | Nonzero v -> v
  ;;

  let map ~like f = function
    | Zero _ -> zero_like like
    | Nonzero v -> Nonzero (f v)
  ;;

  let add ~like a b =
    match a, b with
    | Zero _, Zero _ -> zero_like like
    | t, Zero _ | Zero _, t -> t
    | Nonzero a, Nonzero b -> Nonzero (Ops.add a b)
  ;;

  let sub ~like a b =
    match a, b with
    | Zero _, Zero _ -> zero_like like
    | t, Zero _ -> t
    | Zero _, Nonzero b -> Nonzero (Ops.neg b)
    | Nonzero a, Nonzero b -> Nonzero (Ops.sub a b)
  ;;
end

module Jvp = struct
  open Interp
  module Z = Maybe_zero

  type rule = Prim.t -> value list -> Z.t list -> value list * Z.t list

  let rules : rule Prim.Table.t = Prim.Table.create "jvp"
  let register_rule p rule = Prim.Table.add rules p rule
  let missing_rules () = Prim.Table.missing rules

  type dual =
    { primal : value
    ; tangent : value
    }

  let unpack self v =
    match payload self v with
    | Some { primal; tangent } -> primal, Z.Nonzero tangent
    | None -> v, Z.zero_like v
  ;;

  let pack self primal = function
    | Z.Zero _ -> primal
    | Z.Nonzero tangent -> tracer self (aval primal) { primal; tangent }
  ;;

  let check_rule prim outs touts =
    if List.length outs <> List.length touts
    then
      Error.type_error
        "jvp rule for %s returned %d outputs and %d tangents"
        (Prim.name prim)
        (List.length outs)
        (List.length touts);
    if !Debug.enabled
    then
      Z.check ~rule:"jvp" ~what:"tangent" ~of_:"an output" prim (List.map aval outs) touts
  ;;

  let process self prim args =
    let primals, tangents = args |> List.map (unpack self) |> List.split in
    if List.for_all Z.is_zero tangents
    then bind prim primals
    else (
      let rule = Prim.Table.find rules prim in
      let outs, touts = rule prim primals tangents in
      check_rule prim outs touts;
      List.map2 (pack self) outs touts)
  ;;

  let jvp_flat f primals tangents =
    let self = create ~name:"jvp" ~concrete:(fun d -> Some d.primal) process in
    let outs, aux = handle self (fun () -> List.map2 (pack self) primals tangents |> f) in
    let ys, dys = outs |> List.map (unpack self) |> List.split in
    ys, dys, aux
  ;;
end

module Linear = struct
  type var =
    { id : int
    ; aval : Aval.t
    }

  type atom =
    | Var of var
    | Known of Interp.value

  type eqn =
    { prim : Prim.t
    ; inputs : atom list
    ; outputs : var list
    }

  type t =
    { inputs : var list
    ; eqns : eqn list
    ; outputs : atom list
    ; num_vars : int
    }

  type operand =
    | Linear of Aval.t
    | Fixed of Interp.value

  let var_ids atoms =
    List.filter_map
      (function
        | Var x -> Some x.id
        | Known _ -> None)
      atoms
  ;;
end

module Partial_eval = struct
  open Interp
  open Linear

  type rule =
    stage:(Prim.t -> value list -> value list)
    -> Prim.t
    -> value list
    -> known:bool list
    -> value list

  let rules : rule Prim.Table.t = Prim.Table.create "partial_eval"
  let register_rule p rule = Prim.Table.add rules p rule

  let prune g =
    let live =
      Graph.live
        ~inputs:(fun (e : eqn) -> var_ids e.inputs)
        ~outputs:(fun (e : eqn) -> List.map (fun x -> x.id) e.outputs)
        ~roots:(var_ids g.outputs)
        g.eqns
    in
    { g with eqns = Lists.pick live g.eqns }
  ;;

  let trace f operands =
    let ids = Graph.counter () in
    let fresh aval = { id = Graph.fresh ids; aval } in
    let eqns = Tape.create () in
    let atom self v =
      match payload self v with
      | Some x -> Var x
      | None -> Known v
    in
    let stage self prim args =
      let outputs = args |> List.map aval |> Prim.abstract_eval prim |> List.map fresh in
      Tape.record eqns { prim; inputs = List.map (atom self) args; outputs };
      List.map (fun x -> tracer self x.aval x) outputs
    in
    let process self prim args =
      let known = List.map (fun v -> Option.is_none (payload self v)) args in
      if List.for_all Fun.id known
      then bind prim args
      else (
        match Prim.Table.find_opt rules prim with
        | Some rule -> rule ~stage:(stage self) prim args ~known
        | None -> stage self prim args)
    in
    let self = create ~name:"partial_eval" process in
    let input = function
      | Fixed v -> v, None
      | Linear a ->
        let x = fresh a in
        tracer self a x, Some x
    in
    let ins = List.map input operands in
    let outs, aux = handle self (fun () -> List.map fst ins |> f) in
    let g =
      { inputs = List.filter_map snd ins
      ; eqns = Tape.to_list eqns
      ; outputs = List.map (atom self) outs
      ; num_vars = Graph.count ids
      }
    in
    prune g, aux
  ;;
end

module Transpose = struct
  open Interp
  open Linear
  module Z = Maybe_zero

  type rule = Prim.t -> operand list -> Z.t list -> Z.t list

  let rules : rule Prim.Table.t = Prim.Table.create "transpose"
  let register_rule p rule = Prim.Table.add rules p rule
  let missing_rules () = Prim.Table.missing rules

  let operand_of = function
    | Var x -> Linear x.aval
    | Known v -> Fixed v
  ;;

  let check_rule (e : eqn) cin =
    if List.length cin <> List.length e.inputs
    then
      Error.type_error
        "transpose rule for %s returned %d cotangents for %d operands"
        (Prim.name e.prim)
        (List.length cin)
        (List.length e.inputs);
    if !Debug.enabled
    then (
      let expected, got =
        List.combine e.inputs cin
        |> List.filter_map (function
          | Var x, c -> Some (x.aval, c)
          | Known _, _ -> None)
        |> List.split
      in
      Z.check ~rule:"transpose" ~what:"cotangent" ~of_:"an operand" e.prim expected got)
  ;;

  let cotangent cts (x : var) =
    match Env.find_opt x.id cts with
    | Some c -> Z.Nonzero c
    | None -> Z.Zero x.aval
  ;;

  let accumulate cts atom c =
    match atom, c with
    | Var x, Z.Nonzero c ->
      Env.update
        x.id
        (function
          | None -> Some c
          | Some c0 -> Some (Ops.add c0 c))
        cts
    | _ -> cts
  ;;

  let backprop cts (e : eqn) =
    let couts = List.map (cotangent cts) e.outputs in
    if List.for_all Z.is_zero couts
    then cts
    else (
      let rule = Prim.Table.find rules e.prim in
      let cin = rule e.prim (List.map operand_of e.inputs) couts in
      check_rule e cin;
      List.fold_left2 accumulate cts e.inputs cin)
  ;;

  let transpose g cts =
    let seeded =
      List.map Z.nonzero cts |> List.fold_left2 accumulate Env.empty g.outputs
    in
    let cts = List.rev g.eqns |> List.fold_left backprop seeded in
    List.map (fun x -> cotangent cts x |> Z.instantiate) g.inputs
  ;;
end

module Vjp = struct
  open Interp
  module Z = Maybe_zero

  let linearize f xs =
    let jvp_on_unknown_tangents dxs =
      let ys, dys, aux = Jvp.jvp_flat f xs (List.map Z.nonzero dxs) in
      List.map Z.instantiate dys, (ys, aux)
    in
    let operands = List.map (fun x -> Linear.Linear (aval x)) xs in
    let lin, (ys, aux) = Partial_eval.trace jvp_on_unknown_tangents operands in
    if not (List.for_all alive ys)
    then
      Error.type_error
        "linearize: an output depends on the tangents; a primitive needs a partial_eval \
         rule";
    ys, aux, lin
  ;;

  let vjp_flat f xs =
    let ys, aux, lin = linearize f xs in
    ys, aux, Transpose.transpose lin
  ;;
end

module Derivatives = struct
  open Interp
  module B = Builtin
  module Z = Maybe_zero

  type partials = Prim.t -> value list -> value -> (value -> value) option list

  type operand = Linear.operand =
    | Linear of Aval.t
    | Fixed of value

  type transpose = Prim.t -> operand list -> value -> Z.t list

  let add_term y acc = function
    | None -> acc
    | Some t -> Z.add ~like:y acc (Z.Nonzero t)
  ;;

  let partials_of p (partials : partials) xs y =
    let ds = partials p xs y in
    if List.length ds <> List.length xs
    then
      Error.type_error
        "%s: %d partials for %d operands"
        (Prim.name p)
        (List.length ds)
        (List.length xs);
    ds
  ;;

  let elementwise_jvp partials p xs dxs =
    let y = bind1 p xs in
    let term d dx =
      match d, dx with
      | Some d, Z.Nonzero dx -> Some (d dx)
      | None, _ | _, Z.Zero _ -> None
    in
    let ds = partials_of p partials xs y in
    let t =
      List.fold_left2 (fun acc d dx -> add_term y acc (term d dx)) (Z.zero_like y) ds dxs
    in
    [ y ], [ t ]
  ;;

  let register_elementwise p ~partials = Jvp.register_rule p (elementwise_jvp partials)

  let operand_aval = function
    | Linear a -> a
    | Fixed v -> aval v
  ;;

  let linear_jvp groups p xs dxs =
    let y = bind1 p xs in
    let term group =
      let in_group = List.mapi (fun i _ -> List.mem i group) dxs in
      let group_dxs = Lists.pick in_group dxs in
      if List.for_all Z.is_zero group_dxs
      then None
      else (
        let args =
          Lists.merge
            in_group
            (List.map Z.instantiate group_dxs)
            (Lists.reject in_group xs)
        in
        Some (bind1 p args))
    in
    let groups = groups (List.length dxs) in
    let t =
      List.fold_left (fun acc group -> add_term y acc (term group)) (Z.zero_like y) groups
    in
    [ y ], [ t ]
  ;;

  let transpose_rule (transpose : transpose) p operands cts =
    match cts with
    | [ Z.Nonzero c ] -> transpose p operands c
    | _ -> Error.type_error "transpose of %s: expected 1 cotangent" (Prim.name p)
  ;;

  let register_groups p groups transpose =
    Jvp.register_rule p (linear_jvp groups);
    Transpose.register_rule p (transpose_rule transpose)
  ;;

  let register_linear ?(fixed = []) p ~transpose =
    let not_fixed n = Lists.range n |> List.filter (fun i -> not (List.mem i fixed)) in
    register_groups p (fun n -> [ not_fixed n ]) transpose
  ;;

  let register_multilinear p ~transpose =
    register_groups p (fun n -> Lists.range n |> List.map (fun i -> [ i ])) transpose
  ;;

  open Ops

  let nz = Z.nonzero
  let bad p = Error.type_error "transpose of %s: wrong operands" (Prim.name p)

  let with1 p xs f =
    match xs with
    | [ x ] -> f x
    | _ -> Prim.arity_error p ~expected:1 xs
  ;;

  let with2 p xs f =
    match xs with
    | [ x; y ] -> f x y
    | _ -> Prim.arity_error p ~expected:2 xs
  ;;

  let jvp2 p xs dxs f =
    match xs, dxs with
    | [ x; y ], [ dx; dy ] -> f (x, dx) (y, dy)
    | _ -> Prim.arity_error p ~expected:2 xs
  ;;

  let unary p f =
    register_elementwise p ~partials:(fun p xs y -> with1 p xs (fun x -> [ f x y ]))
  ;;

  let binary p f =
    register_elementwise p ~partials:(fun p xs z -> with2 p xs (fun x y -> f x y z))
  ;;

  let linear1 p f =
    register_linear p ~transpose:(fun p operands c ->
      match operands with
      | [ Linear a ] -> [ nz (f p a c) ]
      | _ -> bad p)
  ;;

  let bilinear p ~left ~right =
    register_multilinear p ~transpose:(fun p operands c ->
      match operands with
      | [ Linear _; Fixed y ] -> [ nz (left c y); Z.zero_like y ]
      | [ Fixed x; Linear _ ] -> [ Z.zero_like x; nz (right x c) ]
      | _ -> bad p)
  ;;

  let elementwise_rules () =
    (* [open Ops] brings [Ops.Infix] into scope: [t * y] is [mul t y]. *)
    binary B.Div (fun _ y z -> [ Some (fun t -> t / y); Some (fun t -> -(t * z / y)) ]);
    binary B.Max (fun x y _ ->
      let wx () = (x > y) + (scalar 0.5 * equal x y) in
      [ Some (fun t -> wx () * t); Some (fun t -> (scalar 1. - wx ()) * t) ]);
    binary B.Greater (fun _ _ _ -> [ None; None ]);
    binary B.Less (fun _ _ _ -> [ None; None ]);
    unary B.Sin (fun x _ -> Some (fun t -> t * cos x));
    unary B.Cos (fun x _ -> Some (fun t -> -(t * sin x)));
    unary B.Exp (fun _ y -> Some (fun t -> t * y));
    unary B.Log (fun x _ -> Some (fun t -> t / x));
    unary B.Sqrt (fun _ y -> Some (fun t -> t / (scalar 2. * y)));
    register_elementwise (B.Integer_pow 0) ~partials:(fun p xs _ ->
      with1 p xs (fun x ->
        match p with
        | B.Integer_pow 0 -> [ None ]
        | B.Integer_pow n ->
          let power = integer_pow ~exponent:(Int.pred n) in
          [ Some (fun t -> t * (scalar (float n) * power x)) ]
        | _ -> Prim.arity_error p ~expected:1 xs))
  ;;

  let slice_transpose p (a : Aval.t) c =
    match p with
    | B.Slice { axis; start; stop } ->
      let zeros_of len = zeros (Shape.set_dim ~axis len a.shape) in
      let before = if Stdlib.(start > 0) then [ zeros_of start ] else [] in
      let after =
        let n = a.shape.(axis) in
        if Stdlib.(stop < n) then [ zeros_of (Int.sub n stop) ] else []
      in
      (match List.concat [ before; [ c ]; after ] with
       | [ c ] -> c
       | parts -> concatenate ~axis parts)
    | _ -> bad p
  ;;

  let concatenate_transpose p operands c =
    match p with
    | B.Concatenate axis ->
      let piece start x =
        let stop = Int.add start (operand_aval x).shape.(axis) in
        stop, nz (slice ~axis ~start ~stop c)
      in
      List.fold_left_map piece 0 operands |> snd
    | _ -> bad p
  ;;

  let linear_rules () =
    register_linear B.Add ~transpose:(fun _ _ c -> [ nz c; nz c ]);
    register_linear B.Sub ~transpose:(fun _ _ c -> [ nz c; nz (neg c) ]);
    register_linear B.Neg ~transpose:(fun _ _ c -> [ nz (neg c) ]);
    Jvp.register_rule B.Add (fun p xs dxs ->
      jvp2 p xs dxs (fun (x, dx) (y, dy) ->
        let z = add x y in
        [ z ], [ Z.add ~like:z dx dy ]));
    Jvp.register_rule B.Sub (fun p xs dxs ->
      jvp2 p xs dxs (fun (x, dx) (y, dy) ->
        let z = sub x y in
        [ z ], [ Z.sub ~like:z dx dy ]));
    register_linear B.Select ~fixed:[ 0 ] ~transpose:(fun p operands c ->
      match operands with
      | [ Fixed q; x; _ ] ->
        let zs = zeros (operand_aval x).shape in
        [ Z.zero_like q; nz (select q c zs); nz (select q zs c) ]
      | _ -> bad p);
    linear1 (B.Reduce_sum []) (fun p a c ->
      match p with
      | B.Reduce_sum axes -> broadcast ~shape:a.shape ~axes c
      | _ -> bad p);
    linear1
      (B.Broadcast { shape = [||]; axes = [] })
      (fun p _ c ->
         match p with
         | B.Broadcast { axes; _ } -> reduce_sum ~axes c
         | _ -> bad p);
    linear1 (B.Transpose [||]) (fun p _ c ->
      match p with
      | B.Transpose perm -> transpose ~perm:(Shape.inverse_perm perm) c
      | _ -> bad p);
    linear1 (B.Reshape [||]) (fun _ a c -> reshape ~shape:a.shape c);
    register_linear (B.Concatenate 0) ~transpose:concatenate_transpose;
    linear1 (B.Slice { axis = 0; start = 0; stop = 0 }) slice_transpose
  ;;

  let multilinear_rules () =
    bilinear B.Mul ~left:mul ~right:mul;
    Transpose.register_rule B.Div (fun p operands cts ->
      match operands, cts with
      | [ Linear _; Fixed y ], [ Z.Nonzero c ] -> [ nz (div c y); Z.zero_like y ]
      | _ -> bad p);
    bilinear
      B.Dot
      ~left:(fun c y -> dot c (transpose y))
      ~right:(fun x c -> dot (transpose x) c)
  ;;

  let reduce_max_rule () =
    Jvp.register_rule (B.Reduce_max []) (fun p xs dxs ->
      match p, xs, dxs with
      | B.Reduce_max axes, [ x ], [ dx ] ->
        let y = bind1 p [ x ] in
        let mean_at_max dx =
          let at_max = equal x (broadcast ~shape:(shape x) ~axes y) in
          reduce_sum ~axes (at_max * dx) / reduce_sum ~axes at_max
        in
        [ y ], [ Z.map ~like:y mean_at_max dx ]
      | _ -> Prim.arity_error p ~expected:1 xs)
  ;;

  let () =
    elementwise_rules ();
    linear_rules ();
    multilinear_rules ();
    reduce_max_rule ()
  ;;
end

module Ad = struct
  open Interp
  module Z = Maybe_zero

  let flat_fn = Tree.flat_fn

  let check_leaves ~fn ~what expected got =
    let ne = List.length expected
    and ng = List.length got in
    if ne <> ng then Error.type_error "%s: expected %d %ss, got %d" fn ne what ng;
    List.iter2
      (fun e g ->
         if not (Aval.equal (aval e) (aval g))
         then
           Error.type_error
             "%s: %s %s does not match %s"
             fn
             what
             (Aval.to_string (aval g))
             (Aval.to_string (aval e)))
      expected
      got
  ;;

  let jvp_tree ~input:in_tree ~output:out_tree f primals tangents =
    let xs, s = Tree.flatten in_tree primals in
    let dxs, ds = Tree.flatten in_tree tangents in
    if not (Tree.equal_structure ds s)
    then Error.type_error "jvp: tangents have another structure than primals";
    check_leaves ~fn:"jvp" ~what:"tangent" xs dxs;
    let flat_f = flat_fn in_tree out_tree f s in
    let ys, dys, out = Jvp.jvp_flat flat_f xs (List.map Z.nonzero dxs) in
    let rebuild = Tree.unflatten out_tree out in
    rebuild ys, List.map Z.instantiate dys |> rebuild
  ;;

  let jvp f x dx = jvp_tree ~input:Tree.value ~output:Tree.value f x dx
  let deriv f x = jvp f x (Ops.ones_like x) |> snd

  let basis x i =
    Array.init (Ops.size x) (fun j -> if i = j then 1. else 0.)
    |> Ops.array ~shape:(Ops.shape x)
  ;;

  let jacobian ~axis ~out x rows =
    let shape = Array.append out (Ops.shape x) in
    match rows with
    | [] -> Ops.zeros shape
    | rows -> Ops.stack ~axis rows |> Ops.reshape ~shape
  ;;

  let jacfwd f x =
    let cols = List.init (Ops.size x) (fun i -> jvp f x (basis x i) |> snd) in
    let out =
      match cols with
      | col :: _ -> Ops.shape col
      | [] -> Ops.shape (f x)
    in
    jacobian ~axis:(-1) ~out x cols
  ;;

  let vjp_tree ~input:in_tree ~output:out_tree f x =
    let xs, s = Tree.flatten in_tree x in
    let ys, out, back = Vjp.vjp_flat (flat_fn in_tree out_tree f s) xs in
    let back ct =
      let cts, cs = Tree.flatten out_tree ct in
      if not (Tree.equal_structure cs out)
      then Error.type_error "vjp: cotangents have another structure than outputs";
      check_leaves ~fn:"vjp" ~what:"cotangent" ys cts;
      back cts |> Tree.unflatten in_tree s
    in
    Tree.unflatten out_tree out ys, back
  ;;

  let vjp f x = vjp_tree ~input:Tree.value ~output:Tree.value f x

  let value_and_grad_tree ~wrt f x =
    let y, back = vjp_tree ~input:wrt ~output:Tree.value f x in
    if Ops.rank y <> 0
    then Error.type_error "grad: f must return a scalar, got %s" (Aval.to_string (aval y));
    y, back (Ops.ones_like y)
  ;;

  let value_and_grad f x = value_and_grad_tree ~wrt:Tree.value f x
  let grad_tree ~wrt f x = value_and_grad_tree ~wrt f x |> snd
  let grad f x = grad_tree ~wrt:Tree.value f x

  let jacrev f x =
    let y, back = vjp f x in
    let rows = List.init (Ops.size y) (fun i -> back (basis y i)) in
    jacobian ~axis:0 ~out:(Ops.shape y) x rows
  ;;

  let hessian f x = jacfwd (grad f) x
end
