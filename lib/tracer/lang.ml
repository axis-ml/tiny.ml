open Common

module Prim = struct
  type t = ..

  type info =
    { name : string
    ; abstract_eval : t -> Aval.t list -> Aval.t list
    ; impl : t -> Ndarray.t list -> Ndarray.t list
    }

  let key (p : t) = Obj.Extension_constructor.(id (of_val p))
  let registry : (int, info) Hashtbl.t = Hashtbl.create 32

  let register p ~name ~abstract_eval ~impl =
    let k = key p in
    let taken_by_another =
      Hashtbl.to_seq registry
      |> Seq.exists (fun (k', i) -> k' <> k && String.equal i.name name)
    in
    if taken_by_another
    then invalid_arg @@ "Prim.register: another primitive is already named " ^ name;
    Hashtbl.replace registry k { name; abstract_eval; impl }
  ;;

  let info p =
    match Hashtbl.find_opt registry (key p) with
    | Some i -> i
    | None ->
      let constructor = Obj.Extension_constructor.(name (of_val p)) in
      invalid_arg @@ "Prim: unregistered primitive " ^ constructor
  ;;

  let name p = (info p).name
  let abstract_eval p avals = (info p).abstract_eval p avals
  let impl p arrays = (info p).impl p arrays

  let names () =
    Hashtbl.to_seq_values registry
    |> Seq.map (fun i -> i.name)
    |> List.of_seq
    |> List.sort compare
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
      | None -> Error.missing_rule "%s has no rule for %s" t.transformation (name p)
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
    ; mutable active : int
    }

  type any = Any : _ t -> any

  let of_ndarray a = Arr a

  let aval = function
    | Arr a -> Aval.of_ndarray a
    | Tracer (T t) -> Aval.create ~dtype:t.aval.dtype (Array.copy t.aval.shape)
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

  let tracer interp (aval : Aval.t) payload =
    let aval = Aval.create ~dtype:aval.dtype (Array.copy aval.shape) in
    Tracer (T { interp; aval; payload })
  ;;

  let payload (type p) (i : p t) v : p option =
    match v with
    | Arr _ -> None
    | Tracer (T t) ->
      (match i.same t.interp.witness with
       | Some Refl -> Some t.payload
       | None -> None)
  ;;

  let eval prim args =
    let expect_array = function
      | Arr a -> a
      | v ->
        Error.escaped_tracer
          "%s received %s at the eval interpreter: the tracer escaped the transformation \
           that created it"
          (Prim.name prim)
          (describe v)
    in
    let arrays = List.map expect_array args in
    let expected = Prim.abstract_eval prim (List.map Aval.of_ndarray arrays) in
    let outs = Prim.impl prim arrays in
    if not (List.map Aval.of_ndarray outs |> List.equal Aval.equal expected)
    then
      Printf.ksprintf
        failwith
        "%s: impl and abstract_eval disagree on the output type"
        (Prim.name prim);
    List.map of_ndarray outs
  ;;

  type _ Effect.t +=
    | Bind : Prim.t * value list -> value list Effect.t
    | Stack : any list Effect.t

  let bind prim args =
    try Effect.perform (Bind (prim, args)) with
    | Effect.Unhandled (Bind _) -> eval prim args
  ;;

  let bind1 prim args =
    match bind prim args with
    | [ y ] -> y
    | ys ->
      Error.type_error "%s: expected 1 result, got %d" (Prim.name prim) (List.length ys)
  ;;

  let stack () =
    try Effect.perform Stack with
    | Effect.Unhandled Stack -> []
  ;;

  let handle i f =
    let open Effect.Deep in
    let leave () = i.active <- i.active - 1 in
    i.active <- i.active + 1;
    match_with
      f
      ()
      { retc =
          (fun x ->
            leave ();
            x)
      ; exnc =
          (fun e ->
            let bt = Printexc.get_raw_backtrace () in
            leave ();
            Printexc.raise_with_backtrace e bt)
      ; effc =
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Bind (prim, args) ->
              Some
                (fun (k : (a, _) continuation) ->
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
        Error.escaped_tracer "%s was read after its transformation returned" (describe v);
      (match t.interp.concrete t.payload with
       | Some under -> to_ndarray under
       | None ->
         Error.concretization_error
           "%s has no concrete value: %s only knows its type, so its data cannot be read \
            (e.g. for an if on a value)"
           (describe v)
           t.interp.name)
  ;;

  let stack_to_string () =
    stack () |> List.rev_map label |> List.cons "eval" |> String.concat " < "
  ;;
end

module Builtin = struct
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
    Shape.axes_error ~rank axes |> Option.iter (Error.type_error "%s: %s" (Prim.name p))
  ;;

  let avals_to_string avals = List.map Aval.to_string avals |> String.concat " and "

  let same_types p = function
    | [] -> Prim.arity_error p ~expected:1 []
    | x :: rest as all ->
      if List.exists (fun y -> y <> x) rest
      then
        Error.type_error
          "%s: operands must have the same type, got %s"
          (Prim.name p)
          (avals_to_string all);
      x
  ;;

  let unary p f = function
    | [ x ] -> f x
    | xs -> Prim.arity_error p ~expected:1 xs
  ;;

  let n_ary p n avals =
    if List.compare_length_with avals n <> 0 then Prim.arity_error p ~expected:n avals;
    same_types p avals
  ;;

  let reduce_type p axes x =
    check_axes p ~rank:(Array.length x.shape) axes;
    Shape.remove_dims ~axes x.shape |> Aval.create ~dtype:x.dtype
  ;;

  let dot_type p avals =
    match avals with
    | [ ({ shape = [| m; k |]; _ } as x); ({ shape = [| k'; n |]; _ } as y) ]
      when k = k' && x.dtype = y.dtype -> Aval.create ~dtype:x.dtype [| m; n |]
    | [ _; _ ] ->
      Error.type_error
        "%s: expected [m; k] and [k; n], got %s"
        (Prim.name p)
        (avals_to_string avals)
    | _ -> Prim.arity_error p ~expected:2 avals
  ;;

  let broadcast_type p ~shape ~axes x =
    let name = Prim.name p in
    let rank = Array.length shape in
    if Array.length x.shape + List.length axes <> rank
    then
      Error.type_error
        "%s: inserting %d axes into %s cannot give %s"
        name
        (List.length axes)
        (Aval.to_string x)
        (shape_str shape);
    check_axes p ~rank axes;
    Shape.kept_dims ~rank axes
    |> List.mapi (fun k d -> x.shape.(k), d)
    |> List.find_opt (fun (n, d) -> n <> shape.(d))
    |> Option.iter (fun (_, d) ->
      Error.type_error
        "%s: operand %s does not match %s at dimension %d"
        name
        (Aval.to_string x)
        (shape_str shape)
        d);
    Aval.create ~dtype:x.dtype shape
  ;;

  let transpose_type p perm x =
    if not (Shape.is_perm ~rank:(Array.length x.shape) perm)
    then
      Error.type_error "%s: invalid permutation for %s" (Prim.name p) (Aval.to_string x);
    Array.map (Array.get x.shape) perm |> Aval.create ~dtype:x.dtype
  ;;

  let reshape_type p shape x =
    if Shape.numel shape <> Shape.numel x.shape
    then
      Error.type_error
        "%s: cannot reshape %s to %s"
        (Prim.name p)
        (Aval.to_string x)
        (shape_str shape);
    Aval.create ~dtype:x.dtype shape
  ;;

  let check_axis p axis x =
    if axis < 0 || axis >= Array.length x.shape
    then
      Error.type_error
        "%s: axis %d out of range for %s"
        (Prim.name p)
        axis
        (Aval.to_string x)
  ;;

  let concatenate_type p axis avals =
    match avals with
    | [] -> Prim.arity_error p ~expected:1 []
    | x :: _ ->
      check_axis p axis x;
      let off_axis a = Shape.set_dim ~axis 0 a.shape in
      if List.exists (fun a -> off_axis a <> off_axis x || a.dtype <> x.dtype) avals
      then
        Error.type_error
          "%s: shapes must match except on axis %d, got %s"
          (Prim.name p)
          axis
          (avals_to_string avals);
      let total = List.fold_left (fun n a -> n + a.shape.(axis)) 0 avals in
      Shape.set_dim ~axis total x.shape |> Aval.create ~dtype:x.dtype
  ;;

  let slice_type p ~axis ~start ~stop x =
    check_axis p axis x;
    if start < 0 || stop < start || stop > x.shape.(axis)
    then
      Error.type_error
        "%s: [%d, %d) out of range for %s on axis %d"
        (Prim.name p)
        start
        stop
        (Aval.to_string x)
        axis;
    Shape.set_dim ~axis (stop - start) x.shape |> Aval.create ~dtype:x.dtype
  ;;

  let abstract_eval p avals =
    let out =
      match p with
      | Add | Sub | Mul | Div | Max | Greater | Less -> n_ary p 2 avals
      | Neg | Sin | Cos | Exp | Log | Sqrt | Integer_pow _ -> unary p Fun.id avals
      | Select -> n_ary p 3 avals
      | Dot -> dot_type p avals
      | Reduce_sum axes | Reduce_max axes -> unary p (reduce_type p axes) avals
      | Broadcast { shape; axes } -> unary p (broadcast_type p ~shape ~axes) avals
      | Transpose perm -> unary p (transpose_type p perm) avals
      | Reshape shape -> unary p (reshape_type p shape) avals
      | Concatenate axis -> concatenate_type p axis avals
      | Slice { axis; start; stop } -> unary p (slice_type p ~axis ~start ~stop) avals
      | _ ->
        invalid_arg @@ "Builtin.abstract_eval: not a built-in primitive: " ^ Prim.name p
    in
    [ out ]
  ;;

  let impl p args =
    let wrong_count () =
      Error.type_error "%s: wrong operand count (%d)" (Prim.name p) (List.length args)
    in
    let unary f =
      match args with
      | [ x ] -> f x
      | _ -> wrong_count ()
    in
    let binary f =
      match args with
      | [ x; y ] -> f x y
      | _ -> wrong_count ()
    in
    let pointwise f = unary (Ndarray.map f) in
    let out =
      match p with
      | Add -> binary (Ndarray.map2 ( +. ))
      | Sub -> binary (Ndarray.map2 ( -. ))
      | Mul -> binary (Ndarray.map2 ( *. ))
      | Div -> binary (Ndarray.map2 ( /. ))
      | Max -> binary Ndarray.maximum
      | Greater -> binary Ndarray.greater
      | Less -> binary Ndarray.less
      | Neg -> pointwise Float.neg
      | Sin -> pointwise Float.sin
      | Cos -> pointwise Float.cos
      | Exp -> pointwise Float.exp
      | Log -> pointwise Float.log
      | Sqrt -> pointwise Float.sqrt
      | Integer_pow n -> pointwise (fun v -> Float.pow v (float n))
      | Select ->
        (match args with
         | [ c; x; y ] -> Ndarray.select c x y
         | _ -> wrong_count ())
      | Dot -> binary Ndarray.dot
      | Reduce_sum axes -> unary (Ndarray.reduce_sum ~axes)
      | Reduce_max axes -> unary (Ndarray.reduce_max ~axes)
      | Broadcast { shape; axes } -> unary (Ndarray.broadcast ~shape ~axes)
      | Transpose perm -> unary (Ndarray.transpose ~perm)
      | Reshape shape -> unary (Ndarray.reshape ~shape)
      | Concatenate axis -> Ndarray.concatenate ~axis args
      | Slice { axis; start; stop } -> unary (Ndarray.slice ~axis ~start ~stop)
      | _ -> wrong_count ()
    in
    [ out ]
  ;;

  let table =
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

  let () = List.iter (fun (p, name) -> Prim.register p ~name ~abstract_eval ~impl) table
  let all = List.map fst table
end

module Ops = struct
  module B = Builtin

  type value = Interp.value

  let shape_str = Shape.describe
  let of_ndarray = Interp.of_ndarray
  let scalar x = of_ndarray (Ndarray.scalar x)

  let array ?shape data =
    let shape = Option.value shape ~default:[| Array.length data |] in
    if Array.length data <> Shape.numel shape
    then
      Error.type_error
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
    then Error.type_error "Ops.to_float: expected a scalar, got %s" (describe v);
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
  let normalize_axis ~rank axis = if axis < 0 then axis + rank else axis
  let broadcast ~shape ~axes x = unop (B.Broadcast { shape; axes }) x
  let reshape ~shape:target x = if shape x = target then x else unop (B.Reshape target) x

  let transpose ?perm x =
    let reversed r = Array.init r (fun i -> r - 1 - i) in
    let perm =
      match perm with
      | Some p -> p
      | None -> reversed (rank x)
    in
    unop (B.Transpose perm) x
  ;;

  let expand_dims ~axes x =
    let s = shape x in
    let r = Array.length s + List.length axes in
    let axes = Shape.normalize_axes ~rank:r axes in
    Shape.axes_error ~rank:r axes |> Option.iter (Error.type_error "expand_dims: %s");
    reshape ~shape:(Shape.insert_ones ~axes s) x
  ;;

  let squeeze ?axes x =
    let s = shape x in
    let r = Array.length s in
    let axes =
      match axes with
      | Some axes -> Shape.normalize_axes ~rank:r axes
      | None -> Lists.range r |> List.filter (fun d -> s.(d) = 1)
    in
    let not_unit d = d < 0 || d >= r || s.(d) <> 1 in
    List.find_opt not_unit axes
    |> Option.iter (fun d ->
      Error.type_error "squeeze: axis %d of %s is not size 1" d (shape_str s));
    reshape ~shape:(Shape.remove_dims ~axes s) x
  ;;

  let broadcast_to ~shape:target x =
    let s = shape x in
    let lead = Array.length target - Array.length s in
    let fail () =
      Error.type_error "cannot broadcast %s to %s" (shape_str s) (shape_str target)
    in
    if lead < 0 then fail ();
    let dims = Array.to_list s |> List.mapi (fun i d -> lead + i, d, target.(lead + i)) in
    if List.exists (fun (_, d, t) -> d <> t && d <> 1) dims then fail ();
    let new_axes, kept =
      List.partition_map
        (fun (i, d, t) -> if d = 1 && t <> 1 then Either.Left i else Either.Right d)
        dims
    in
    if lead = 0 && new_axes = []
    then x
    else
      x
      |> reshape ~shape:(Array.of_list kept)
      |> broadcast ~shape:target ~axes:(Lists.range lead @ new_axes)
  ;;

  let common_shape = function
    | [] -> Some [||]
    | s :: rest ->
      List.fold_left
        (fun acc s -> Option.bind acc (Shape.broadcast_shapes s))
        (Some s)
        rest
  ;;

  let broadcast_operands p xs =
    let shapes = List.map shape xs in
    let common = common_shape shapes in
    match common with
    | Some s -> List.map (fun x -> if shape x = s then x else broadcast_to ~shape:s x) xs
    | None ->
      Error.type_error
        "%s: cannot broadcast %s together (dims must match from the right, or be 1)"
        (Prim.name p)
        (List.map shape_str shapes |> String.concat " and ")
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

  let pow ~exponent:p x =
    if Float.is_integer p && Float.abs p <= 64.
    then integer_pow ~exponent:(int_of_float p) x
    else exp (mul (scalar p) (log x))
  ;;

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

  (* The formulas below read as math: [Infix] is opened locally, so [+] is [add]. *)
  let is_negative x = less x (scalar 0.)
  let equal x y = Infix.(one_minus (x > y) * one_minus (x < y))
  let not_equal x y = one_minus (equal x y)
  let minimum x y = Infix.(-(maximum (-x) (-y)))
  let square x = integer_pow ~exponent:2 x
  let abs x = select (is_negative x) (neg x) x
  let relu x = maximum x (scalar 0.)
  let logistic x = Infix.(scalar 1. / (scalar 1. + exp (-x)))

  let tanh x =
    let open Infix in
    let e = exp (scalar (-2.) * abs x) in
    let t = one_minus e / (scalar 1. + e) in
    select (is_negative x) (-t) t
  ;;

  let tan x = Infix.(sin x / cos x)
  let sinh x = Infix.(scalar 0.5 * (exp x - exp (-x)))
  let cosh x = Infix.(scalar 0.5 * (exp x + exp (-x)))

  let asinh x =
    let open Infix in
    let a = abs x in
    let t = log (a + sqrt (square a + scalar 1.)) in
    select (is_negative x) (-t) t
  ;;

  let acosh x = Infix.(log (x + sqrt (square x - scalar 1.)))
  let atanh x = Infix.(scalar 0.5 * log ((scalar 1. + x) / one_minus x))

  let reduce p_of ?axes ?(keepdims = false) x =
    let r = rank x in
    let axes =
      match axes with
      | None -> Lists.range r
      | Some axes -> Shape.normalize_axes ~rank:r axes
    in
    let y = unop (p_of axes) x in
    if keepdims
    then (
      let kept = shape x |> Array.mapi (fun d n -> if List.mem d axes then 1 else n) in
      reshape ~shape:kept y)
    else y
  ;;

  let reduce_sum ?axes ?keepdims x = reduce (fun a -> B.Reduce_sum a) ?axes ?keepdims x
  let reduce_max ?axes ?keepdims x = reduce (fun a -> B.Reduce_max a) ?axes ?keepdims x
  let reduce_min ?axes ?keepdims x = x |> neg |> reduce_max ?axes ?keepdims |> neg

  let mean ?axes ?keepdims x =
    let total = reduce_sum ?axes ?keepdims x in
    let count = size x / max 1 (size total) in
    div total (scalar (float count))
  ;;

  let concatenate ?(axis = 0) xs =
    match xs with
    | [] -> Error.type_error "concatenate: no arrays"
    | x :: _ ->
      let axis = normalize_axis ~rank:(rank x) axis in
      bind1 (B.Concatenate axis) xs
  ;;

  let slice ?(axis = 0) ~start ~stop x =
    let r = rank x in
    let axis = normalize_axis ~rank:r axis in
    if axis < 0 || axis >= r
    then Error.type_error "slice: axis out of range for %s" (shape_str (shape x));
    let n = (shape x).(axis) in
    let norm i = if i < 0 then i + n else i in
    bind1 (B.Slice { axis; start = norm start; stop = norm stop }) [ x ]
  ;;

  let stack ?(axis = 0) xs =
    match xs with
    | [] -> Error.type_error "stack: no arrays"
    | x :: _ ->
      let axis = normalize_axis ~rank:(rank x + 1) axis in
      xs |> List.map (expand_dims ~axes:[ axis ]) |> concatenate ~axis
  ;;

  let dot x y =
    let sx = shape x
    and sy = shape y in
    let mm x y = bind1 B.Dot [ x; y ] in
    match sx, sy with
    | [| k |], [| k' |] when k = k' -> reduce_sum (mul x y)
    | [| m; k |], [| k' |] when k = k' ->
      y |> reshape ~shape:[| k; 1 |] |> mm x |> reshape ~shape:[| m |]
    | [| k |], [| k'; n |] when k = k' ->
      mm (reshape ~shape:[| 1; k |] x) y |> reshape ~shape:[| n |]
    | [| _; k |], [| k'; _ |] when k = k' -> mm x y
    | ([| _ |] | [| _; _ |]), ([| _ |] | [| _; _ |]) ->
      Error.type_error
        "dot: shapes %s and %s are not aligned"
        (shape_str sx)
        (shape_str sy)
    | _ ->
      Error.type_error
        "dot: only rank 1 and 2 are supported, got %s and %s"
        (shape_str sx)
        (shape_str sy)
  ;;

  let matmul = dot

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

  let rec equal_structure a b =
    match a, b with
    | Leaf, Leaf -> true
    | Node xs, Node ys -> List.equal equal_structure xs ys
    | Leaf, Node _ | Node _, Leaf -> false
  ;;

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

  let leaves t x = flatten t x |> fst
  let structure t x = flatten t x |> snd

  let rec num_leaves = function
    | Leaf -> 1
    | Node ss -> List.fold_left (fun n s -> n + num_leaves s) 0 ss
  ;;

  let mismatch () = Error.type_error "Tree: the structure does not fit the description"

  let rec unflat : type a. a t -> structure -> value list -> a * value list =
    fun t s leaves ->
    let many a ss leaves =
      let step leaves s =
        let x, rest = unflat a s leaves in
        rest, x
      in
      let rest, xs = List.fold_left_map step leaves ss in
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
    unflat t s leaves |> fst
  ;;

  let flat_fn in_tree out_tree f s leaves =
    unflatten in_tree s leaves |> f |> flatten out_tree
  ;;

  let map t f x =
    let leaves, s = flatten t x in
    List.map f leaves |> unflatten t s
  ;;

  let map2 t f x y =
    let xs, s = flatten t x in
    let ys, s' = flatten t y in
    if not (equal_structure s s')
    then Error.type_error "Tree.map2: the two values have different structures";
    List.map2 f xs ys |> unflatten t s
  ;;
end
