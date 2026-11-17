open Common
open Lang

module Maybe_zero = struct
  type t =
    | Zero of Aval.t
    | Nonzero of Interp.value

  let zero_like v = Zero (Interp.aval v)

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
         let ag = aval g in
         if ag <> e
         then
           Error.type_error
             "%s rule for %s: %s %s for %s %s"
             rule
             (Prim.name prim)
             what
             (Aval.to_string ag)
             of_
             (Aval.to_string e))
      expected
      got
  ;;

  let instantiate = function
    | Zero a -> Ops.zeros a.shape
    | Nonzero v -> v
  ;;

  let map out f = function
    | Zero _ -> zero_like out
    | Nonzero v -> Nonzero (f v)
  ;;

  let add out a b =
    match a, b with
    | Zero _, Zero _ -> zero_like out
    | t, Zero _ | Zero _, t -> t
    | Nonzero a, Nonzero b -> Nonzero (Ops.add a b)
  ;;

  let sub out a b =
    match a, b with
    | Zero _, Zero _ -> zero_like out
    | t, Zero _ -> t
    | Zero _, Nonzero b -> Nonzero (Ops.neg b)
    | Nonzero a, Nonzero b -> Nonzero (Ops.sub a b)
  ;;
end

module Jvp = struct
  (* Values with a zero tangent are not wrapped: zeros stay symbolic. *)

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

  let lift self v =
    match payload self v with
    | Some { primal; tangent } -> primal, Z.Nonzero tangent
    | None -> v, Z.zero_like v
  ;;

  let dual self primal = function
    | Z.Zero _ -> primal
    | Z.Nonzero tangent -> tracer self (aval primal) { primal; tangent }
  ;;

  let process self prim args =
    let primals, tangents = List.split (List.map (lift self) args) in
    if List.for_all Z.is_zero tangents
    then bind prim primals
    else (
      let outs, touts = (Prim.Table.find rules prim) prim primals tangents in
      if List.length outs <> List.length touts
      then
        Error.type_error
          "jvp rule for %s returned %d outputs and %d tangents"
          (Prim.name prim)
          (List.length outs)
          (List.length touts);
      if !Debug.enabled
      then
        Z.check
          ~rule:"jvp"
          ~what:"tangent"
          ~of_:"an output"
          prim
          (List.map aval outs)
          touts;
      List.map2 (dual self) outs touts)
  ;;

  let concrete { primal; _ } = Some primal

  let jvp_flat f primals tangents =
    let self = create ~name:"jvp" ~concrete process in
    let ins = List.map2 (dual self) primals tangents in
    let outs, aux = handle self (fun () -> f ins) in
    let ys, dys = List.split (List.map (lift self) outs) in
    ys, dys, aux
  ;;
end

module Vjp = struct
  (* Dynamic tape. [back] runs after [handle] returns, so rule ops bind to
     the outer interpreters: that is what makes nesting work. *)

  open Interp
  module Z = Maybe_zero

  type rule = Prim.t -> value list -> value list -> Z.t list -> Z.t list

  (* [needed]: which operands need a cotangent. *)
  type sparse_rule = needed:bool list -> rule

  let rules : sparse_rule Prim.Table.t = Prim.Table.create "vjp"
  let register_sparse_rule p (rule : sparse_rule) = Prim.Table.add rules p rule
  let register_rule p (rule : rule) = register_sparse_rule p (fun ~needed:_ -> rule)
  let missing_rules () = Prim.Table.missing rules

  type node =
    { primal : value
    ; id : int
    }

  type entry =
    { prim : Prim.t
    ; back : Z.t list -> Z.t list
    ; in_avals : Aval.t list
    ; out_avals : Aval.t list
    ; in_ids : int option list (* [None]: a constant *)
    ; out_ids : int list
    }

  let vjp_flat f xs =
    let ids = Graph.counter () in
    let fresh () = Graph.fresh ids in
    let tape = ref [] in
    let node self primal id = tracer self (aval primal) { primal; id } in
    let lift self v =
      match payload self v with
      | Some { primal; id } -> primal, Some id
      | None -> v, None
    in
    let process self prim args =
      let primals, in_ids = List.split (List.map (lift self) args) in
      let outs = bind prim primals in
      if List.for_all Option.is_none in_ids
      then outs
      else (
        let needed = List.map Option.is_some in_ids in
        let back = (Prim.Table.find rules prim) ~needed prim primals outs in
        let out_ids = List.map (fun _ -> fresh ()) outs in
        tape
        := { prim
           ; back
           ; in_avals = List.map aval primals
           ; out_avals = List.map aval outs
           ; in_ids
           ; out_ids
           }
           :: !tape;
        List.map2 (node self) outs out_ids)
    in
    let self = create ~name:"vjp" ~concrete:(fun { primal; _ } -> Some primal) process in
    let in_ids = List.map (fun _ -> fresh ()) xs in
    let ins = List.map2 (node self) xs in_ids in
    let outs, aux = handle self (fun () -> f ins) in
    let ys, out_ids = List.split (List.map (lift self) outs) in
    let in_avals = List.map aval xs in
    let num_ids = Graph.count ids in
    let tape =
      List.combine
        (Graph.reaches
           ~num_ids
           ~inputs:(fun e -> List.filter_map Fun.id e.in_ids)
           ~outputs:(fun e -> e.out_ids)
           ~roots:(List.filter_map Fun.id out_ids)
           !tape)
        !tape
      |> List.filter_map (fun (live, e) -> if live then Some e else None)
    in
    let back cts_out =
      let acc = Array.make num_ids None in
      let add id c =
        acc.(id)
        <- (match acc.(id) with
            | None -> Some c
            | Some c0 -> Some (Ops.add c0 c))
      in
      (* Missing id: a zero cotangent. *)
      List.iter2 (fun id c -> Option.iter (fun id -> add id c) id) out_ids cts_out;
      List.iter
        (fun e ->
           let cts =
             List.map2
               (fun a id ->
                  match acc.(id) with
                  | Some c -> Z.Nonzero c
                  | None -> Z.Zero a)
               e.out_avals
               e.out_ids
           in
           if not (List.for_all Z.is_zero cts)
           then (
             let cin = e.back cts in
             if List.length cin <> List.length e.in_ids
             then
               Error.type_error
                 "vjp rule for %s returned %d cotangents for %d operands"
                 (Prim.name e.prim)
                 (List.length cin)
                 (List.length e.in_ids);
             if !Debug.enabled
             then
               Z.check
                 ~rule:"vjp"
                 ~what:"cotangent"
                 ~of_:"an operand"
                 e.prim
                 e.in_avals
                 cin;
             List.iter2
               (fun id c ->
                  match id, c with
                  | Some id, Z.Nonzero c -> add id c
                  | _ -> ())
               e.in_ids
               cin))
        tape;
      List.map2
        (fun a id ->
           match acc.(id) with
           | Some c -> c
           | None -> Ops.zeros a.Aval.shape)
        in_avals
        in_ids
    in
    ys, aux, back
  ;;
end

module Derivatives = struct
  open Interp
  module B = Builtin
  module Z = Maybe_zero

  type partials = Prim.t -> value list -> value -> (value -> value) option list

  let one_output p = function
    | [ y ] -> y
    | ys ->
      Error.type_error "%s: expected 1 output, got %d" (Prim.name p) (List.length ys)
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
    let t =
      List.fold_left2
        (fun acc d dx ->
           match d, dx with
           | None, _ | _, Z.Zero _ -> acc
           | Some d, Z.Nonzero dx ->
             let term = d dx in
             Z.add y acc (Z.Nonzero term))
        (Z.zero_like y)
        (partials_of p partials xs y)
        dxs
    in
    [ y ], [ t ]
  ;;

  (* [Ops] broadcasts before binding, so no cotangent is un-broadcast. *)
  let elementwise_vjp partials ~needed p xs outs =
    let ds =
      List.map2
        (fun d n -> if n then d else None)
        (partials_of p partials xs (one_output p outs))
        needed
    in
    let avals = List.map aval xs in
    fun cts ->
      match cts with
      | [ Z.Nonzero c ] ->
        List.map2
          (fun d a ->
             match d with
             | None -> Z.Zero a
             | Some d -> Z.Nonzero (d c))
          ds
          avals
      | _ -> Error.type_error "vjp of %s: expected 1 cotangent" (Prim.name p)
  ;;

  let register_elementwise p ~partials =
    Jvp.register_rule p (elementwise_jvp partials);
    Vjp.register_sparse_rule p (elementwise_vjp partials)
  ;;

  type operand =
    | Linear of Aval.t
    | Fixed of value

  type transpose = Prim.t -> operand list -> value -> Z.t list
  type groups = int -> int list list

  let operand_aval = function
    | Linear a -> a
    | Fixed v -> aval v
  ;;

  let linear_jvp (groups : groups) p xs dxs =
    let y = bind1 p xs in
    let dxs = Array.of_list dxs in
    let t =
      List.fold_left
        (fun acc g ->
           if List.for_all (fun i -> Z.is_zero dxs.(i)) g
           then acc
           else (
             let args =
               List.mapi (fun i x -> if List.mem i g then Z.instantiate dxs.(i) else x) xs
             in
             let term = bind1 p args in
             Z.add y acc (Z.Nonzero term)))
        (Z.zero_like y)
        (groups (Array.length dxs))
    in
    [ y ], [ t ]
  ;;

  let linear_vjp (groups : groups) (transpose : transpose) ~needed p xs _ =
    let avals = List.map aval xs in
    let needed = Array.of_list needed in
    let calls =
      List.filter_map
        (fun g ->
           if List.exists (fun i -> needed.(i)) g
           then
             Some
               ( g
               , List.mapi
                   (fun i x -> if List.mem i g then Linear (aval x) else Fixed x)
                   xs )
           else None)
        (groups (List.length xs))
    in
    fun cts ->
      match cts with
      | [ Z.Nonzero c ] ->
        let cin = Array.of_list (List.map (fun a -> Z.Zero a) avals) in
        List.iter
          (fun (g, operands) ->
             let cs = transpose p operands c in
             if List.length cs <> Array.length cin
             then
               Error.type_error
                 "transpose rule for %s returned %d cotangents for %d operands"
                 (Prim.name p)
                 (List.length cs)
                 (Array.length cin);
             List.iteri (fun i ct -> if List.mem i g then cin.(i) <- ct) cs)
          calls;
        Array.to_list cin
      | _ -> Error.type_error "vjp of %s: expected 1 cotangent" (Prim.name p)
  ;;

  let register_groups p groups transpose =
    Jvp.register_rule p (linear_jvp groups);
    Vjp.register_sparse_rule p (linear_vjp groups transpose)
  ;;

  let register_linear ?(fixed = []) p ~transpose =
    register_groups
      p
      (fun n -> [ List.filter (fun i -> not (List.mem i fixed)) (List.init n Fun.id) ])
      transpose
  ;;

  let register_multilinear p ~transpose =
    register_groups p (fun n -> List.init n (fun i -> [ i ])) transpose
  ;;

  let inverse_perm perm =
    let inv = Array.make (Array.length perm) 0 in
    Array.iteri (fun i q -> inv.(q) <- i) perm;
    inv
  ;;

  let at_max ~axes x y = Ops.(equal x (broadcast ~shape:(shape x) ~axes y))

  let () =
    let open Ops in
    let unary p f =
      register_elementwise p ~partials:(fun p xs y ->
        match xs with
        | [ x ] -> [ f x y ]
        | _ -> Prim.arity_error p ~expected:1 xs)
    in
    let binary p f =
      register_elementwise p ~partials:(fun p xs z ->
        match xs with
        | [ x; y ] -> f x y z
        | _ -> Prim.arity_error p ~expected:2 xs)
    in
    binary B.Mul (fun x y _ -> [ Some (fun t -> mul t y); Some (fun t -> mul x t) ]);
    binary B.Div (fun _ y z ->
      [ Some (fun t -> div t y); Some (fun t -> neg (div (mul t z) y)) ]);
    (* Ties split evenly, like JAX. *)
    binary B.Max (fun x y _ ->
      let wx () = add (greater x y) (mul (scalar 0.5) (equal x y)) in
      [ Some (fun t -> mul (wx ()) t); Some (fun t -> mul (sub (scalar 1.) (wx ())) t) ]);
    binary B.Greater (fun _ _ _ -> [ None; None ]);
    binary B.Less (fun _ _ _ -> [ None; None ]);
    unary B.Sin (fun x _ -> Some (fun t -> mul t (cos x)));
    unary B.Cos (fun x _ -> Some (fun t -> neg (mul t (sin x))));
    unary B.Exp (fun _ y -> Some (fun t -> mul t y));
    unary B.Log (fun x _ -> Some (fun t -> div t x));
    unary B.Sqrt (fun _ y -> Some (fun t -> div t (mul (scalar 2.) y)));
    register_elementwise (B.Integer_pow 0) ~partials:(fun p xs _ ->
      match p, xs with
      | B.Integer_pow 0, [ _ ] -> [ None ]
      | B.Integer_pow n, [ x ] ->
        [ Some
            (fun t ->
              mul t (mul (scalar (float n)) (integer_pow ~exponent:(Int.pred n) x)))
        ]
      | _ -> Prim.arity_error p ~expected:1 xs);
    let bad p = Error.type_error "transpose of %s: wrong operands" (Prim.name p) in
    let nz c = Z.Nonzero c in
    let linear1 p f =
      register_linear p ~transpose:(fun p operands c ->
        match operands with
        | [ Linear a ] -> [ nz (f p a c) ]
        | _ -> bad p)
    in
    register_linear B.Add ~transpose:(fun _ _ c -> [ nz c; nz c ]);
    register_linear B.Sub ~transpose:(fun _ _ c -> [ nz c; nz (neg c) ]);
    register_linear B.Neg ~transpose:(fun _ _ c -> [ nz (neg c) ]);
    (* Skips instantiating a zero, unlike the generic rule. *)
    Jvp.register_rule B.Add (fun p xs dxs ->
      match xs, dxs with
      | [ x; y ], [ dx; dy ] ->
        let z = add x y in
        [ z ], [ Z.add z dx dy ]
      | _ -> Prim.arity_error p ~expected:2 xs);
    Jvp.register_rule B.Sub (fun p xs dxs ->
      match xs, dxs with
      | [ x; y ], [ dx; dy ] ->
        let z = sub x y in
        [ z ], [ Z.sub z dx dy ]
      | _ -> Prim.arity_error p ~expected:2 xs);
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
      | B.Transpose perm -> transpose ~perm:(inverse_perm perm) c
      | _ -> bad p);
    linear1 (B.Reshape [||]) (fun _ a c -> reshape ~shape:a.shape c);
    register_linear (B.Concatenate 0) ~transpose:(fun p operands c ->
      match p with
      | B.Concatenate axis ->
        snd
          (List.fold_left_map
             (fun start x ->
                let stop = Int.add start (operand_aval x).shape.(axis) in
                stop, nz (slice ~axis ~start ~stop c))
             0
             operands)
      | _ -> bad p);
    linear1
      (B.Slice { axis = 0; start = 0; stop = 0 })
      (fun p a c ->
         match p with
         | B.Slice { axis; start; stop } ->
           let n = a.shape.(axis) in
           let zeros_of len =
             let s = Array.copy a.shape in
             s.(axis) <- len;
             zeros s
           in
           let parts =
             (if Stdlib.(start > 0) then [ zeros_of start ] else [])
             @ [ c ]
             @ if Stdlib.(stop < n) then [ zeros_of (Int.sub n stop) ] else []
           in
           (match parts with
            | [ c ] -> c
            | parts -> concatenate ~axis parts)
         | _ -> bad p);
    register_multilinear B.Dot ~transpose:(fun p operands c ->
      match operands with
      | [ Linear _; Fixed y ] -> [ nz (dot c (transpose y)); Z.zero_like y ]
      | [ Fixed x; Linear _ ] -> [ Z.zero_like x; nz (dot (transpose x) c) ]
      | _ -> bad p);
    Jvp.register_rule (B.Reduce_max []) (fun p xs dxs ->
      match p, xs, dxs with
      | B.Reduce_max axes, [ x ], [ dx ] ->
        let y = bind1 p [ x ] in
        let t =
          Z.map
            y
            (fun dx ->
               let at_max = at_max ~axes x y in
               div (reduce_sum ~axes (mul at_max dx)) (reduce_sum ~axes at_max))
            dx
        in
        [ y ], [ t ]
      | _ -> Prim.arity_error p ~expected:1 xs);
    Vjp.register_rule (B.Reduce_max []) (fun p xs outs ->
      match p, xs, outs with
      | B.Reduce_max axes, [ x ], [ y ] ->
        fun cts ->
          (match cts with
           | [ Z.Nonzero c ] ->
             let up v = broadcast ~shape:(shape x) ~axes v in
             let at_max = at_max ~axes x y in
             [ nz (div (mul at_max (up c)) (up (reduce_sum ~axes at_max))) ]
           | _ -> bad p)
      | _ -> Prim.arity_error p ~expected:1 xs)
  ;;
end

module Ad = struct
  open Interp
  module Z = Maybe_zero

  let flat_fn in_tree out_tree f s leaves =
    Tree.flatten out_tree (f (Tree.unflatten in_tree s leaves))
  ;;

  let check_leaves ~fn ~what expected got =
    let ne = List.length expected
    and ng = List.length got in
    if ne <> ng then Error.type_error "%s: expected %d %ss, got %d" fn ne what ng;
    List.iter2
      (fun e g ->
         let ae = aval e
         and ag = aval g in
         if ae <> ag
         then
           Error.type_error
             "%s: %s %s does not match %s"
             fn
             what
             (Aval.to_string ag)
             (Aval.to_string ae))
      expected
      got
  ;;

  let jvp in_tree out_tree f primals tangents =
    let xs, s = Tree.flatten in_tree primals in
    let dxs, ds = Tree.flatten in_tree tangents in
    if ds <> s then Error.type_error "jvp: tangents have another structure than primals";
    check_leaves ~fn:"jvp" ~what:"tangent" xs dxs;
    let ys, dys, out =
      Jvp.jvp_flat (flat_fn in_tree out_tree f s) xs (List.map (fun t -> Z.Nonzero t) dxs)
    in
    ( Tree.unflatten out_tree out ys
    , Tree.unflatten out_tree out (List.map Z.instantiate dys) )
  ;;

  let jvp1 f x dx = jvp Tree.value Tree.value f x dx
  let deriv f x = snd (jvp1 f x (Ops.ones_like x))

  let basis x i =
    Ops.array
      ~shape:(Ops.shape x)
      (Array.init (Ops.size x) (fun j -> if i = j then 1. else 0.))
  ;;

  let jacfwd f x =
    let cols = List.init (Ops.size x) (fun i -> snd (jvp1 f x (basis x i))) in
    let out = Ops.stack ~axis:(-1) cols in
    Ops.reshape ~shape:(Array.append (Ops.shape (List.hd cols)) (Ops.shape x)) out
  ;;

  let vjp in_tree out_tree f x =
    let xs, s = Tree.flatten in_tree x in
    let ys, out, back = Vjp.vjp_flat (flat_fn in_tree out_tree f s) xs in
    let back ct =
      let cts, cs = Tree.flatten out_tree ct in
      if cs <> out
      then Error.type_error "vjp: cotangents have another structure than outputs";
      check_leaves ~fn:"vjp" ~what:"cotangent" ys cts;
      Tree.unflatten in_tree s (back cts)
    in
    Tree.unflatten out_tree out ys, back
  ;;

  let vjp1 f x = vjp Tree.value Tree.value f x

  let value_and_grad tree f x =
    let y, back = vjp tree Tree.value f x in
    if Ops.rank y <> 0
    then Error.type_error "grad: f must return a scalar, got %s" (Aval.to_string (aval y));
    y, back (Ops.ones_like y)
  ;;

  let value_and_grad1 f x = value_and_grad Tree.value f x
  let grad tree f x = snd (value_and_grad tree f x)
  let grad1 f x = grad Tree.value f x

  let jacrev f x =
    let y, back = vjp1 f x in
    let rows = List.init (Ops.size y) (fun i -> back (basis y i)) in
    Ops.reshape ~shape:(Array.append (Ops.shape y) (Ops.shape x)) (Ops.stack rows)
  ;;

  let hessian f x = jacfwd (grad1 f) x
end
