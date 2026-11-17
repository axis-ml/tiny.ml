open Common

module Prim = struct
  type t = ..

  type info =
    { name : string
    ; abstract_eval : t -> Aval.t list -> Aval.t list
    ; impl : t -> Ndarray.t list -> Ndarray.t list
    }

  (* Constructors with arguments share one key. *)
  let key (p : t) = Obj.Extension_constructor.(id (of_val p))
  let registry : (int, info) Hashtbl.t = Hashtbl.create 32

  let register p ~name ~abstract_eval ~impl =
    let k = key p in
    Hashtbl.iter
      (fun k' i ->
         if k' <> k && String.equal i.name name
         then invalid_arg ("Prim.register: another primitive is already named " ^ name))
      registry;
    Hashtbl.replace registry k { name; abstract_eval; impl }
  ;;

  let info p =
    match Hashtbl.find_opt registry (key p) with
    | Some i -> i
    | None ->
      invalid_arg
        ("Prim: unregistered primitive " ^ Obj.Extension_constructor.(name (of_val p)))
  ;;

  let name p = (info p).name
  let abstract_eval p avals = (info p).abstract_eval p avals
  let impl p arrays = (info p).impl p arrays

  let names () =
    List.sort compare (Hashtbl.fold (fun _ i acc -> i.name :: acc) registry [])
  ;;

  let arity_error p ~expected got =
    Error.type_error
      "%s: expected %d operand(s), got %d"
      (name p)
      expected
      (List.length got)
  ;;

  module Table = struct
    type prim = t

    type 'a t =
      { transformation : string
      ; rules : (int, 'a) Hashtbl.t
      }

    let create transformation = { transformation; rules = Hashtbl.create 32 }
    let add t p rule = Hashtbl.replace t.rules (key p) rule

    let find t (p : prim) =
      match Hashtbl.find_opt t.rules (key p) with
      | Some r -> r
      | None ->
        raise
          (Error.Missing_rule
             (Printf.sprintf "%s has no rule for %s" t.transformation (name p)))
    ;;

    let find_opt t (p : prim) = Hashtbl.find_opt t.rules (key p)

    let missing t =
      Hashtbl.fold
        (fun k i acc -> if Hashtbl.mem t.rules k then acc else i.name :: acc)
        registry
        []
      |> List.sort compare
    ;;
  end
end

module Interp = struct
  (* A per-interpreter witness recovers ['p] without a cast ([Type.Id] by
     hand, for OCaml 5.0). *)

  open Error

  type (_, _) eq = Refl : ('a, 'a) eq
  type _ witness = ..

  type value =
    | Arr of Ndarray.t
    | Tracer of tracer

  and tracer =
    | T :
        { interp : 'p t
        ; aval : Aval.t
        ; payload : 'p
        }
        -> tracer

  and 'p t =
    { name : string
    ; id : int
    ; witness : 'p witness
    ; same : 'b. 'b witness -> ('p, 'b) eq option
    ; process : 'p t -> Prim.t -> value list -> value list
    ; concrete : 'p -> value option
    ; mutable active : int (* how many [handle]s of it are running *)
    }

  type any = Any : _ t -> any

  let of_ndarray a = Arr a

  let aval = function
    | Arr a -> Aval.of_ndarray a
    | Tracer (T t) -> t.aval
  ;;

  let label (Any i) = Printf.sprintf "%s#%d" i.name i.id

  let describe = function
    | Arr a -> "array " ^ Aval.to_string (Aval.of_ndarray a)
    | Tracer (T t) ->
      Printf.sprintf "%s tracer #%d %s" t.interp.name t.interp.id (Aval.to_string t.aval)
  ;;

  let to_ndarray_opt = function
    | Arr a -> Some a
    | Tracer _ -> None
  ;;

  let next_id = Atomic.make 1

  let create (type p) ~name ?(concrete = fun _ -> None) process : p t =
    let module W = struct
      type _ witness += W : p witness
    end
    in
    let same : type b. b witness -> (p, b) eq option = function
      | W.W -> Some Refl
      | _ -> None
    in
    { name
    ; id = Atomic.fetch_and_add next_id 1
    ; witness = W.W
    ; same
    ; process
    ; concrete
    ; active = 0
    }
  ;;

  let name i = i.name
  let id i = i.id
  let tracer interp aval payload = Tracer (T { interp; aval; payload })

  let payload (type p) (i : p t) v : p option =
    match v with
    | Arr _ -> None
    | Tracer (T t) ->
      (match i.same t.interp.witness with
       | Some Refl -> Some t.payload
       | None -> None)
  ;;

  let eval prim args =
    let concrete = function
      | Arr a -> a
      | v ->
        raise
          (Escaped_tracer
             (Printf.sprintf
                "%s received %s at the eval interpreter: the tracer escaped the \
                 transformation that created it"
                (Prim.name prim)
                (describe v)))
    in
    let arrays = List.map concrete args in
    let expected = Prim.abstract_eval prim (List.map Aval.of_ndarray arrays) in
    let outs = Prim.impl prim arrays in
    if List.map Aval.of_ndarray outs <> expected
    then
      failwith
        (Printf.sprintf
           "%s: impl and abstract_eval disagree on the output type"
           (Prim.name prim));
    List.map (fun a -> Arr a) outs
  ;;

  type _ Effect.t +=
    | Bind : Prim.t * value list -> value list Effect.t
    | Stack : any list Effect.t (** innermost first, eval excluded *)

  let bind prim args =
    try Effect.perform (Bind (prim, args)) with
    | Effect.Unhandled (Bind _) -> eval prim args
  ;;

  let bind1 prim args =
    match bind prim args with
    | [ y ] -> y
    | ys -> type_error "%s: expected 1 result, got %d" (Prim.name prim) (List.length ys)
  ;;

  let stack () =
    try Effect.perform Stack with
    | Effect.Unhandled Stack -> []
  ;;

  let handle i f =
    let open Effect.Deep in
    i.active <- i.active + 1;
    match_with
      f
      ()
      { retc =
          (fun x ->
            i.active <- i.active - 1;
            x)
      ; exnc =
          (fun e ->
            let bt = Printexc.get_raw_backtrace () in
            i.active <- i.active - 1;
            Printexc.raise_with_backtrace e bt)
      ; effc =
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Bind (prim, args) ->
              Some
                (fun (k : (a, _) continuation) ->
                  (* Re-raise at the [bind] in [f], so it unwinds [f]. *)
                  match i.process i prim args with
                  | out -> continue k out
                  | exception e ->
                    discontinue_with_backtrace k e (Printexc.get_raw_backtrace ()))
            | Stack ->
              Some (fun (k : (a, _) continuation) -> continue k (Any i :: stack ()))
            | _ -> None)
      }
  ;;

  let alive = function
    | Arr _ -> true
    | Tracer (T t) -> t.interp.active > 0
  ;;

  let rec to_ndarray = function
    | Arr a -> a
    | Tracer (T t) as v ->
      if t.interp.active = 0
      then
        raise
          (Escaped_tracer
             (Printf.sprintf "%s was read after its transformation returned" (describe v)));
      (match t.interp.concrete t.payload with
       | Some under -> to_ndarray under
       | None ->
         raise
           (Concretization_error
              (Printf.sprintf
                 "%s has no concrete value: %s only knows its type, so its data cannot \
                  be read (e.g. for an if on a value)"
                 (describe v)
                 t.interp.name)))
  ;;

  let stack_to_string () = String.concat " < " ("eval" :: List.rev_map label (stack ()))
end

module Builtin = struct
  open Error
  open Aval

  let shape_str = Shape.describe

  type Prim.t +=
    | Add
    | Sub
    | Mul
    | Div
    | Max
    | Greater
    | Less
    | Neg
    | Sin
    | Cos
    | Exp
    | Log
    | Sqrt
    | Integer_pow of int
    | Select
    | Dot
    | Reduce_sum of int list
    | Reduce_max of int list
    | Broadcast of
        { shape : int array
        ; axes : int list
        }
    | Transpose of int array
    | Reshape of int array
    | Concatenate of int
    | Slice of
        { axis : int
        ; start : int
        ; stop : int
        }

  let check_axes p ~rank axes =
    Option.iter (type_error "%s: %s" (Prim.name p)) (Shape.axes_error ~rank axes)
  ;;

  let same_types p = function
    | [] -> Prim.arity_error p ~expected:1 []
    | x :: rest as all ->
      if List.exists (fun y -> y <> x) rest
      then
        type_error
          "%s: operands must have the same type, got %s"
          (Prim.name p)
          (String.concat " and " (List.map Aval.to_string all));
      x
  ;;

  let one p = function
    | [ x ] -> x
    | l -> Prim.arity_error p ~expected:1 l
  ;;

  let n_ary p n avals =
    if List.length avals <> n then Prim.arity_error p ~expected:n avals;
    same_types p avals
  ;;

  let reduce p axes x =
    let rank = Array.length x.shape in
    check_axes p ~rank axes;
    { x with shape = Shape.remove_dims ~axes x.shape }
  ;;

  let abstract_eval p avals =
    let name = Prim.name p in
    let out =
      match p with
      | Add | Sub | Mul | Div | Max | Greater | Less -> n_ary p 2 avals
      | Neg | Sin | Cos | Exp | Log | Sqrt | Integer_pow _ -> one p avals
      | Select -> n_ary p 3 avals
      | Dot ->
        (match avals with
         | [ ({ shape = [| m; k |]; _ } as x); ({ shape = [| k'; n |]; _ } as y) ]
           when k = k' && x.dtype = y.dtype -> { x with shape = [| m; n |] }
         | [ _; _ ] ->
           type_error
             "%s: expected [m; k] and [k; n], got %s"
             name
             (String.concat " and " (List.map Aval.to_string avals))
         | _ -> Prim.arity_error p ~expected:2 avals)
      | Reduce_sum axes | Reduce_max axes -> reduce p axes (one p avals)
      | Broadcast { shape; axes } ->
        let x = one p avals in
        let rank = Array.length shape in
        if Array.length x.shape + List.length axes <> rank
        then
          type_error
            "%s: inserting %d axes into %s cannot give %s"
            name
            (List.length axes)
            (Aval.to_string x)
            (shape_str shape);
        check_axes p ~rank axes;
        List.iteri
          (fun k d ->
             if x.shape.(k) <> shape.(d)
             then
               type_error
                 "%s: operand %s does not match %s at dimension %d"
                 name
                 (Aval.to_string x)
                 (shape_str shape)
                 d)
          (Shape.kept_dims ~rank axes);
        { x with shape }
      | Transpose perm ->
        let x = one p avals in
        if not (Shape.is_perm ~rank:(Array.length x.shape) perm)
        then type_error "%s: invalid permutation for %s" name (Aval.to_string x);
        { x with shape = Array.map (fun i -> x.shape.(i)) perm }
      | Reshape shape ->
        let x = one p avals in
        if Shape.numel shape <> Shape.numel x.shape
        then
          type_error
            "%s: cannot reshape %s to %s"
            name
            (Aval.to_string x)
            (shape_str shape);
        { x with shape }
      | Concatenate axis ->
        (match avals with
         | [] -> Prim.arity_error p ~expected:1 []
         | x :: _ ->
           let rank = Array.length x.shape in
           if axis < 0 || axis >= rank
           then type_error "%s: axis %d out of range for %s" name axis (Aval.to_string x);
           let off_axis a = Array.mapi (fun d n -> if d = axis then 0 else n) a.shape in
           if
             List.exists
               (fun a ->
                  Array.length a.shape <> rank
                  || off_axis a <> off_axis x
                  || a.dtype <> x.dtype)
               avals
           then
             type_error
               "%s: shapes must match except on axis %d, got %s"
               name
               axis
               (String.concat " and " (List.map Aval.to_string avals));
           let shape = Array.copy x.shape in
           shape.(axis) <- List.fold_left (fun n a -> n + a.shape.(axis)) 0 avals;
           { x with shape })
      | Slice { axis; start; stop } ->
        let x = one p avals in
        let rank = Array.length x.shape in
        if axis < 0 || axis >= rank
        then type_error "%s: axis %d out of range for %s" name axis (Aval.to_string x);
        if start < 0 || stop < start || stop > x.shape.(axis)
        then
          type_error
            "%s: [%d, %d) out of range for %s on axis %d"
            name
            start
            stop
            (Aval.to_string x)
            axis;
        let shape = Array.copy x.shape in
        shape.(axis) <- stop - start;
        { x with shape }
      | _ -> invalid_arg ("Builtin.abstract_eval: not a built-in primitive: " ^ name)
    in
    [ out ]
  ;;

  let impl p args =
    let out =
      match p, args with
      | Add, [ x; y ] -> Ndarray.map2 ( +. ) x y
      | Sub, [ x; y ] -> Ndarray.map2 ( -. ) x y
      | Mul, [ x; y ] -> Ndarray.map2 ( *. ) x y
      | Div, [ x; y ] -> Ndarray.map2 ( /. ) x y
      | Max, [ x; y ] -> Ndarray.maximum x y
      | Greater, [ x; y ] -> Ndarray.greater x y
      | Less, [ x; y ] -> Ndarray.less x y
      | Neg, [ x ] -> Ndarray.map Float.neg x
      | Sin, [ x ] -> Ndarray.map Float.sin x
      | Cos, [ x ] -> Ndarray.map Float.cos x
      | Exp, [ x ] -> Ndarray.map Float.exp x
      | Log, [ x ] -> Ndarray.map Float.log x
      | Sqrt, [ x ] -> Ndarray.map Float.sqrt x
      | Integer_pow n, [ x ] -> Ndarray.map (fun v -> Float.pow v (float n)) x
      | Select, [ c; x; y ] -> Ndarray.select c x y
      | Dot, [ x; y ] -> Ndarray.dot x y
      | Reduce_sum axes, [ x ] -> Ndarray.reduce_sum ~axes x
      | Reduce_max axes, [ x ] -> Ndarray.reduce_max ~axes x
      | Broadcast { shape; axes }, [ x ] -> Ndarray.broadcast ~shape ~axes x
      | Transpose perm, [ x ] -> Ndarray.transpose ~perm x
      | Reshape shape, [ x ] -> Ndarray.reshape x shape
      | Concatenate axis, xs -> Ndarray.concatenate ~axis xs
      | Slice { axis; start; stop }, [ x ] -> Ndarray.slice ~axis ~start ~stop x
      | p, args ->
        type_error "%s: wrong operand count (%d)" (Prim.name p) (List.length args)
    in
    [ out ]
  ;;

  let all =
    [ Add, "add"
    ; Sub, "sub"
    ; Mul, "mul"
    ; Div, "div"
    ; Max, "max"
    ; Greater, "greater"
    ; Less, "less"
    ; Neg, "neg"
    ; Sin, "sin"
    ; Cos, "cos"
    ; Exp, "exp"
    ; Log, "log"
    ; Sqrt, "sqrt"
    ; Integer_pow 0, "integer_pow"
    ; Select, "select"
    ; Dot, "dot"
    ; Reduce_sum [], "reduce_sum"
    ; Reduce_max [], "reduce_max"
    ; Broadcast { shape = [||]; axes = [] }, "broadcast"
    ; Transpose [||], "transpose"
    ; Reshape [||], "reshape"
    ; Concatenate 0, "concatenate"
    ; Slice { axis = 0; start = 0; stop = 0 }, "slice"
    ]
  ;;

  let () = List.iter (fun (p, name) -> Prim.register p ~name ~abstract_eval ~impl) all
  let all = List.map fst all
end

module Ops = struct
  (* Arguments evaluate right to left. *)

  open Error
  module B = Builtin

  type value = Interp.value

  let shape_str = Shape.describe
  let of_ndarray = Interp.of_ndarray
  let scalar x = of_ndarray (Ndarray.scalar x)

  let array ?shape data =
    let shape = Option.value shape ~default:[| Array.length data |] in
    if Array.length data <> Shape.numel shape
    then
      type_error
        "Ops.array: %d elements cannot fill shape %s"
        (Array.length data)
        (Shape.to_string shape);
    of_ndarray (Ndarray.of_array shape data)
  ;;

  let full shape x = of_ndarray (Ndarray.full shape x)
  let zeros shape = full shape 0.
  let ones shape = full shape 1.
  let arange n = of_ndarray (Ndarray.arange n)
  let linspace lo hi n = of_ndarray (Ndarray.linspace lo hi n)
  let eye n = of_ndarray (Ndarray.eye n)
  let shape x = (Interp.aval x).shape
  let dtype x = (Interp.aval x).dtype
  let rank x = Array.length (shape x)
  let size x = Shape.numel (shape x)
  let zeros_like x = zeros (shape x)
  let ones_like x = ones (shape x)
  let to_ndarray = Interp.to_ndarray
  let describe = Interp.describe

  let to_float v =
    let a = to_ndarray v in
    if Ndarray.rank a <> 0
    then type_error "Ops.to_float: expected a scalar, got %s" (describe v);
    Ndarray.to_float a
  ;;

  let to_array v = Ndarray.to_array (to_ndarray v)

  let allclose ?rtol ?atol x y =
    Ndarray.allclose ?rtol ?atol (to_ndarray x) (to_ndarray y)
  ;;

  let to_string v =
    match Interp.to_ndarray_opt v with
    | Some a -> Ndarray.to_string a
    | None -> describe v
  ;;

  let pp fmt v = Format.pp_print_string fmt (to_string v)
  let bind1 = Interp.bind1
  let unop p x = bind1 p [ x ]
  let broadcast ~shape ~axes x = unop (B.Broadcast { shape; axes }) x
  let reshape ~shape:target x = if shape x = target then x else unop (B.Reshape target) x

  let transpose ?perm x =
    let perm =
      match perm with
      | Some p -> p
      | None ->
        let r = rank x in
        Array.init r (fun i -> r - 1 - i)
    in
    unop (B.Transpose perm) x
  ;;

  let expand_dims ~axes x =
    let s = shape x in
    let r = Array.length s + List.length axes in
    let axes = Shape.normalize_axes ~rank:r axes in
    Option.iter (type_error "expand_dims: %s") (Shape.axes_error ~rank:r axes);
    reshape ~shape:(Shape.insert_ones ~axes s) x
  ;;

  let squeeze ?axes x =
    let s = shape x in
    let r = Array.length s in
    let axes =
      match axes with
      | Some axes -> Shape.normalize_axes ~rank:r axes
      | None -> List.filter (fun d -> s.(d) = 1) (List.init r Fun.id)
    in
    List.iter
      (fun d ->
         if d < 0 || d >= r || s.(d) <> 1
         then type_error "squeeze: axis %d of %s is not size 1" d (shape_str s))
      axes;
    reshape ~shape:(Shape.remove_dims ~axes s) x
  ;;

  (* Size-1 dims are dropped, then [Broadcast] reinserts them at full size. *)
  let broadcast_to ~shape:target x =
    let s = shape x in
    let n = Array.length target
    and k = Array.length s in
    let fail () =
      type_error "cannot broadcast %s to %s" (shape_str s) (shape_str target)
    in
    if k > n then fail ();
    let lead = n - k in
    let stretched = ref [] in
    Array.iteri
      (fun i d ->
         let t = target.(lead + i) in
         if d = 1 && t <> 1
         then stretched := (lead + i) :: !stretched
         else if d <> t
         then fail ())
      s;
    let stretched = List.rev !stretched in
    if lead = 0 && stretched = []
    then x
    else (
      let x =
        if stretched = []
        then x
        else
          reshape
            ~shape:
              (Array.of_list
                 (List.filteri
                    (fun i _ -> not (List.mem (lead + i) stretched))
                    (Array.to_list s)))
            x
      in
      broadcast ~shape:target ~axes:(List.init lead Fun.id @ stretched) x)
  ;;

  let broadcast_operands p xs =
    let shapes = List.map shape xs in
    let common =
      List.fold_left
        (fun acc s -> Option.bind acc (Shape.broadcast_shapes s))
        (Some (List.hd shapes))
        (List.tl shapes)
    in
    match common with
    | Some s ->
      List.map
        (fun (x, sx) -> if sx = s then x else broadcast_to ~shape:s x)
        (List.combine xs shapes)
    | None ->
      type_error
        "%s: cannot broadcast %s together (dims must match from the right, or be 1)"
        (Prim.name p)
        (String.concat " and " (List.map shape_str shapes))
  ;;

  let binop p x y = bind1 p (broadcast_operands p [ x; y ])
  let add = binop B.Add
  let sub = binop B.Sub
  let mul = binop B.Mul
  let div = binop B.Div
  let maximum = binop B.Max
  let greater = binop B.Greater
  let less = binop B.Less
  let neg = unop B.Neg
  let sin = unop B.Sin
  let cos = unop B.Cos
  let exp = unop B.Exp
  let log = unop B.Log
  let sqrt = unop B.Sqrt
  let integer_pow ~exponent:n x = unop (B.Integer_pow n) x
  let select p x y = bind1 B.Select (broadcast_operands B.Select [ p; x; y ])
  let one_minus m = sub (scalar 1.) m
  let greater_equal x y = one_minus (less x y)
  let less_equal x y = one_minus (greater x y)
  let equal x y = mul (one_minus (greater x y)) (one_minus (less x y))
  let not_equal x y = one_minus (equal x y)
  let minimum x y = neg (maximum (neg x) (neg y))
  let square x = integer_pow ~exponent:2 x
  let abs x = select (less x (scalar 0.)) (neg x) x
  let relu x = maximum x (scalar 0.)
  let logistic x = div (scalar 1.) (add (scalar 1.) (exp (neg x)))

  let tanh x =
    let e = exp (mul (scalar (-2.)) (abs x)) in
    let t = div (one_minus e) (add (scalar 1.) e) in
    select (less x (scalar 0.)) (neg t) t
  ;;

  let tan x = div (sin x) (cos x)
  let sinh x = mul (scalar 0.5) (sub (exp x) (exp (neg x)))
  let cosh x = mul (scalar 0.5) (add (exp x) (exp (neg x)))

  (* Avoids cancellation for large negative [x]. *)
  let asinh x =
    let a = abs x in
    let t = log (add a (sqrt (add (square a) (scalar 1.)))) in
    select (less x (scalar 0.)) (neg t) t
  ;;

  let acosh x = log (add x (sqrt (sub (square x) (scalar 1.))))
  let atanh x = mul (scalar 0.5) (log (div (add (scalar 1.) x) (one_minus x)))

  let pow ~exponent:p x =
    if Float.is_integer p && Float.abs p <= 64.
    then integer_pow ~exponent:(int_of_float p) x
    else exp (mul (scalar p) (log x))
  ;;

  let reduce p_of ?axes ?(keepdims = false) x =
    let r = rank x in
    let axes =
      match axes with
      | None -> List.init r Fun.id
      | Some axes -> Shape.normalize_axes ~rank:r axes
    in
    let y = unop (p_of axes) x in
    if keepdims
    then (
      let s = Array.copy (shape x) in
      List.iter (fun d -> s.(d) <- 1) axes;
      reshape ~shape:s y)
    else y
  ;;

  let reduce_sum ?axes ?keepdims x = reduce (fun a -> B.Reduce_sum a) ?axes ?keepdims x
  let reduce_max ?axes ?keepdims x = reduce (fun a -> B.Reduce_max a) ?axes ?keepdims x
  let reduce_min ?axes ?keepdims x = neg (reduce_max ?axes ?keepdims (neg x))

  let mean ?axes ?keepdims x =
    let total = reduce_sum ?axes ?keepdims x in
    div total (scalar (float (size x / max 1 (size total))))
  ;;

  let concatenate ?(axis = 0) xs =
    match xs with
    | [] -> type_error "concatenate: no arrays"
    | x :: _ ->
      let axis = if axis < 0 then axis + rank x else axis in
      bind1 (B.Concatenate axis) xs
  ;;

  let slice ?(axis = 0) ~start ~stop x =
    let r = rank x in
    let axis = if axis < 0 then axis + r else axis in
    if axis < 0 || axis >= r
    then type_error "slice: axis out of range for %s" (shape_str (shape x));
    let n = (shape x).(axis) in
    let norm i = if i < 0 then i + n else i in
    bind1 (B.Slice { axis; start = norm start; stop = norm stop }) [ x ]
  ;;

  let stack ?(axis = 0) xs =
    match xs with
    | [] -> type_error "stack: no arrays"
    | x :: _ ->
      let axis = if axis < 0 then axis + rank x + 1 else axis in
      concatenate ~axis (List.map (expand_dims ~axes:[ axis ]) xs)
  ;;

  let dot x y =
    let sx = shape x
    and sy = shape y in
    let mismatch () =
      type_error "dot: shapes %s and %s are not aligned" (shape_str sx) (shape_str sy)
    in
    let mm x y = bind1 B.Dot [ x; y ] in
    match sx, sy with
    | [| k |], [| k' |] -> if k <> k' then mismatch () else reduce_sum (mul x y)
    | [| m; k |], [| k' |] ->
      if k <> k'
      then mismatch ()
      else reshape ~shape:[| m |] (mm x (reshape ~shape:[| k; 1 |] y))
    | [| k |], [| k'; n |] ->
      if k <> k'
      then mismatch ()
      else reshape ~shape:[| n |] (mm (reshape ~shape:[| 1; k |] x) y)
    | [| _; k |], [| k'; _ |] -> if k <> k' then mismatch () else mm x y
    | _ ->
      type_error
        "dot: only rank 1 and 2 are supported, got %s and %s"
        (shape_str sx)
        (shape_str sy)
  ;;

  let matmul = dot

  (* Last, so the int operators above are not shadowed. *)
  module Infix = struct
    let ( + ) = add
    let ( - ) = sub
    let ( * ) = mul
    let ( / ) = div
    let ( ~- ) = neg
    let ( ** ) x p = pow ~exponent:p x
    let ( > ) = greater
    let ( < ) = less
    let ( >= ) = greater_equal
    let ( <= ) = less_equal
  end

  include Infix
end

module Tree = struct
  type value = Interp.value

  type _ t =
    | Value : value t
    | Unit : unit t
    | Pair : 'a t * 'b t -> ('a * 'b) t
    | List : 'a t -> 'a list t
    | Array : 'a t -> 'a array t
    | Conv :
        { to_tree : 'b -> 'a
        ; of_tree : 'a -> 'b
        ; tree : 'a t
        }
        -> 'b t

  type structure =
    | Leaf
    | Node of structure list

  let value = Value
  let unit = Unit
  let pair a b = Pair (a, b)
  let list a = List a
  let array a = Array a
  let conv ~to_tree ~of_tree tree = Conv { to_tree; of_tree; tree }

  let triple a b c =
    conv
      ~to_tree:(fun (x, y, z) -> x, (y, z))
      ~of_tree:(fun (x, (y, z)) -> x, y, z)
      (pair a (pair b c))
  ;;

  (* Leaves are accumulated in reverse. *)
  let rec flat : type a. a t -> value list -> a -> value list * structure =
    fun t acc x ->
    match t with
    | Value -> x :: acc, Leaf
    | Unit -> acc, Node []
    | Pair (a, b) ->
      let x, y = x in
      let acc, sx = flat a acc x in
      let acc, sy = flat b acc y in
      acc, Node [ sx; sy ]
    | List a ->
      let acc, ss = List.fold_left_map (flat a) acc x in
      acc, Node ss
    | Array a ->
      let acc, ss = List.fold_left_map (flat a) acc (Array.to_list x) in
      acc, Node ss
    | Conv { to_tree; tree; _ } -> flat tree acc (to_tree x)
  ;;

  let flatten t x =
    let acc, s = flat t [] x in
    List.rev acc, s
  ;;

  let leaves t x = fst (flatten t x)
  let structure t x = snd (flatten t x)

  let rec num_leaves = function
    | Leaf -> 1
    | Node ss -> List.fold_left (fun n s -> n + num_leaves s) 0 ss
  ;;

  let mismatch () = Error.type_error "Tree: the structure does not fit the description"

  (* Returns the value and the leaves left over. *)
  let rec unflat : type a. a t -> structure -> value list -> a * value list =
    fun t s leaves ->
    let many a ss leaves =
      let rest, xs =
        List.fold_left_map
          (fun leaves s ->
             let x, rest = unflat a s leaves in
             rest, x)
          leaves
          ss
      in
      xs, rest
    in
    match t, s with
    | Value, Leaf ->
      (match leaves with
       | v :: rest -> v, rest
       | [] -> mismatch ())
    | Unit, Node [] -> (), leaves
    | Pair (a, b), Node [ sx; sy ] ->
      let x, leaves = unflat a sx leaves in
      let y, leaves = unflat b sy leaves in
      (x, y), leaves
    | List a, Node ss -> many a ss leaves
    | Array a, Node ss ->
      let xs, rest = many a ss leaves in
      Array.of_list xs, rest
    | Conv { of_tree; tree; _ }, s ->
      let x, rest = unflat tree s leaves in
      of_tree x, rest
    | _ -> mismatch ()
  ;;

  let unflatten t s leaves =
    let expected = num_leaves s in
    let got = List.length leaves in
    if expected <> got
    then Error.type_error "Tree: expected %d leaves, got %d" expected got;
    fst (unflat t s leaves)
  ;;

  let flat_fn in_tree out_tree f s leaves =
    flatten out_tree (f (unflatten in_tree s leaves))
  ;;

  let map t f x =
    let leaves, s = flatten t x in
    unflatten t s (List.map f leaves)
  ;;

  let map2 t f x y =
    let xs, s = flatten t x in
    let ys, s' = flatten t y in
    if s <> s' then Error.type_error "Tree.map2: the two values have different structures";
    unflatten t s (List.map2 f xs ys)
  ;;
end
