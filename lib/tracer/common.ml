module Error = struct
  type t =
    | Type_error of string
    | Escaped_tracer of string
    | Concretization_error of string
    | Missing_rule of string

  exception Error of t

  let to_string = function
    | Type_error msg -> "Type_error: " ^ msg
    | Escaped_tracer msg -> "Escaped_tracer: " ^ msg
    | Concretization_error msg -> "Concretization_error: " ^ msg
    | Missing_rule msg -> "Missing_rule: " ^ msg
  ;;

  let () =
    Printexc.register_printer (function
      | Error e -> Some (to_string e)
      | _ -> None)
  ;;

  let catch f =
    match f () with
    | v -> Ok v
    | exception Error e -> Stdlib.Error e
  ;;

  let fail kind fmt = Printf.ksprintf (fun msg -> raise (Error (kind msg))) fmt
  let type_error fmt = fail (fun m -> Type_error m) fmt
  let escaped_tracer fmt = fail (fun m -> Escaped_tracer m) fmt
  let concretization_error fmt = fail (fun m -> Concretization_error m) fmt
  let missing_rule fmt = fail (fun m -> Missing_rule m) fmt
end

module Debug = struct
  let enabled =
    ref
      (match Sys.getenv_opt "TINY_ML_DEBUG" with
       | None | Some ("" | "0" | "false") -> false
       | Some _ -> true)
  ;;
end

module Lists = struct
  let range n = List.init n Fun.id

  let rec split_at n l =
    match n, l with
    | 0, _ -> [], l
    | n, x :: rest ->
      let front, back = split_at (n - 1) rest in
      x :: front, back
    | _, [] -> invalid_arg "Lists.split_at: list too short"
  ;;

  let pick mask l =
    List.combine mask l
    |> List.filter_map (fun (keep, x) -> if keep then Some x else None)
  ;;

  let reject mask l = pick (List.map not mask) l

  let rec merge mask yes no =
    match mask, yes, no with
    | [], [], [] -> []
    | true :: mask, y :: yes, no -> y :: merge mask yes no
    | false :: mask, yes, n :: no -> n :: merge mask yes no
    | _ -> invalid_arg "Lists.merge: lengths do not fit the mask"
  ;;

  let all_some l =
    let cons x acc =
      match x, acc with
      | Some x, Some xs -> Some (x :: xs)
      | _ -> None
    in
    List.fold_right cons l (Some [])
  ;;
end

module Env = struct
  module M = Map.Make (Int)

  type 'a t = 'a M.t

  let empty = M.empty
  let add = M.add

  let add_all ids values env =
    List.fold_left2 (fun env id v -> M.add id v env) env ids values
  ;;

  let find_opt = M.find_opt
  let mem = M.mem
  let update = M.update
end

module Tape = struct
  type 'a t = { mutable entries : 'a list }

  let create () = { entries = [] }
  let record t x = t.entries <- x :: t.entries
  let to_list t = List.rev t.entries
  let find_map f t = List.find_map f t.entries
end

module Graph = struct
  type counter = { mutable next : int }

  let counter () = { next = 0 }

  let fresh c =
    let i = c.next in
    c.next <- i + 1;
    i
  ;;

  let count c = c.next

  module Ids = Set.Make (Int)

  let live ~inputs ~outputs ~roots nodes =
    let step (live_ids, marks) n =
      let is_live = List.exists (fun i -> Ids.mem i live_ids) (outputs n) in
      let live_ids =
        if is_live then List.fold_right Ids.add (inputs n) live_ids else live_ids
      in
      live_ids, is_live :: marks
    in
    List.rev nodes |> List.fold_left step (Ids.of_list roots, []) |> snd
  ;;
end

module Shape = struct
  let numel shape = Array.fold_left ( * ) 1 shape

  let to_string s =
    let dims = s |> Array.to_list |> List.map string_of_int |> String.concat "," in
    "[" ^ dims ^ "]"
  ;;

  let describe s = "f64" ^ to_string s

  let strides shape =
    let _, strides =
      Array.fold_right (fun d (stride, acc) -> stride * d, stride :: acc) shape (1, [])
    in
    Array.of_list strides
  ;;

  let scatter ~rank ~default pairs =
    Array.init rank (fun d -> Option.value (List.assoc_opt d pairs) ~default)
  ;;

  let kept_dims ~rank axes =
    Lists.range rank |> List.filter (fun d -> not (List.mem d axes))
  ;;

  let remove_dims ~axes s =
    kept_dims ~rank:(Array.length s) axes |> List.map (Array.get s) |> Array.of_list
  ;;

  let insert_ones ~axes s =
    let rank = Array.length s + List.length axes in
    scatter ~rank ~default:1 (List.combine (kept_dims ~rank axes) (Array.to_list s))
  ;;

  let set_dim ~axis n s = Array.mapi (fun d x -> if d = axis then n else x) s
  let has_duplicates l = List.length (List.sort_uniq compare l) <> List.length l

  let axes_error ~rank axes =
    let out_of_range ax = ax < 0 || ax >= rank in
    match List.find_opt out_of_range axes with
    | Some ax -> Some (Printf.sprintf "axis %d out of range for rank %d" ax rank)
    | None when has_duplicates axes -> Some "duplicate axes"
    | None -> None
  ;;

  let is_perm ~rank perm =
    Array.length perm = rank && List.sort compare (Array.to_list perm) = Lists.range rank
  ;;

  let inverse_perm perm =
    perm
    |> Array.to_list
    |> List.mapi (fun i p -> p, i)
    |> List.sort compare
    |> List.map snd
    |> Array.of_list
  ;;

  let normalize_axes ~rank axes =
    axes |> List.map (fun a -> if a < 0 then a + rank else a) |> List.sort compare
  ;;

  let broadcast_shapes a b =
    let n = max (Array.length a) (Array.length b) in
    let pad s = Array.to_list (Array.append (Array.make (n - Array.length s) 1) s) in
    let dim x y =
      match x, y with
      | x, y when x = y -> Some x
      | x, 1 -> Some x
      | 1, y -> Some y
      | _ -> None
    in
    List.map2 dim (pad a) (pad b) |> Lists.all_some |> Option.map Array.of_list
  ;;
end

module Ndarray = struct
  open Bigarray

  type t = (float, float64_elt, c_layout) Genarray.t

  let shape (a : t) = Genarray.dims a
  let rank (a : t) = Genarray.num_dims a
  let numel = Shape.numel
  let size a = numel (shape a)
  let shape_to_string = Shape.to_string
  let invalid fmt = Printf.ksprintf invalid_arg fmt
  let flat (a : t) = reshape_1 a (size a)
  let create shape : t = Genarray.create float64 c_layout shape

  let init shape f : t =
    let a = create shape in
    let fa = flat a in
    for i = 0 to Array1.dim fa - 1 do
      Array1.unsafe_set fa i (f i)
    done;
    a
  ;;

  let iter_offsets ~shape ~strides f =
    let rank = Array.length shape in
    if rank = 0
    then f 0 0
    else (
      let total = numel shape in
      let last = shape.(rank - 1)
      and last_stride = strides.(rank - 1) in
      let idx = Array.make rank 0 in
      let rec carry k off =
        if k < 0
        then off
        else (
          idx.(k) <- idx.(k) + 1;
          if idx.(k) < shape.(k)
          then off + strides.(k)
          else (
            idx.(k) <- 0;
            carry (k - 1) (off + strides.(k) - (shape.(k) * strides.(k)))))
      in
      let rec runs i off =
        if i < total
        then (
          for j = 0 to last - 1 do
            f (i + j) (off + (j * last_stride))
          done;
          runs (i + last) (carry (rank - 2) off))
      in
      runs 0 0)
  ;;

  let gather ~shape ~strides src : t =
    let out = create shape in
    let fo = flat out
    and fs = flat src in
    iter_offsets ~shape ~strides (fun i off ->
      Array1.unsafe_set fo i (Array1.unsafe_get fs off));
    out
  ;;

  let copy_run (src : (float, float64_elt, c_layout) Array1.t) so dst d len =
    if len >= 64
    then Array1.blit (Array1.sub src so len) (Array1.sub dst d len)
    else
      for j = 0 to len - 1 do
        Array1.unsafe_set dst (d + j) (Array1.unsafe_get src (so + j))
      done
  ;;

  let of_array shape data =
    if Array.length data <> numel shape
    then
      invalid
        "Ndarray.of_array: %d elements for shape %s"
        (Array.length data)
        (shape_to_string shape);
    init shape (Array.get data)
  ;;

  let to_array a =
    let fa = flat a in
    Array.init (Array1.dim fa) (Array1.get fa)
  ;;

  let full shape v = init shape (fun _ -> v)
  let zeros shape = full shape 0.
  let scalar v = full [||] v
  let arange n = init [| n |] float

  let linspace lo hi n =
    if n = 1
    then full [| 1 |] lo
    else init [| n |] (fun i -> lo +. ((hi -. lo) *. float i /. float (n - 1)))
  ;;

  let eye n = init [| n; n |] (fun i -> if i / n = i mod n then 1. else 0.)

  let to_float a =
    if rank a <> 0
    then
      invalid
        "Ndarray.to_float: expected a scalar, got shape %s"
        (shape_to_string (shape a));
    Genarray.get a [||]
  ;;

  let check_axes ~fn ~rank axes =
    Option.iter (invalid "%s: %s" fn) (Shape.axes_error ~rank axes)
  ;;

  let map f a =
    let fa = flat a in
    init (shape a) (fun i -> f (Array1.unsafe_get fa i))
  ;;

  let map2 f a b =
    if shape a <> shape b
    then
      invalid
        "Ndarray.map2: shapes %s and %s differ"
        (shape_to_string (shape a))
        (shape_to_string (shape b));
    let fa = flat a
    and fb = flat b in
    init (shape a) (fun i -> f (Array1.unsafe_get fa i) (Array1.unsafe_get fb i))
  ;;

  let mask p x y = if p x y then 1. else 0.
  let greater = map2 (mask ( > ))
  let less = map2 (mask ( < ))
  let maximum = map2 Float.max_num

  let select p a b =
    if shape p <> shape a || shape a <> shape b
    then invalid "Ndarray.select: shapes differ";
    let fp = flat p
    and fa = flat a
    and fb = flat b in
    init (shape a) (fun i ->
      if Array1.unsafe_get fp i > 0.
      then Array1.unsafe_get fa i
      else Array1.unsafe_get fb i)
  ;;

  let dot a b =
    match shape a, shape b with
    | [| m; k |], [| k'; n |] when k = k' ->
      let fa = flat a
      and fb = flat b in
      let entry i j =
        let rec sum p acc =
          if p = k
          then acc
          else
            sum
              (p + 1)
              (acc
               +. (Array1.unsafe_get fa ((i * k) + p) *. Array1.unsafe_get fb ((p * n) + j)
                  ))
        in
        sum 0 0.
      in
      init [| m; n |] (fun o -> entry (o / n) (o mod n))
    | sa, sb ->
      invalid
        "Ndarray.dot: expected [m; k] and [k; n], got %s and %s"
        (shape_to_string sa)
        (shape_to_string sb)
  ;;

  let reduce ~fn ~init:z ~op ~axes a =
    let ishape = shape a in
    let rank = Array.length ishape in
    check_axes ~fn ~rank axes;
    let oshape = Shape.remove_dims ~axes ishape in
    let into =
      Shape.scatter
        ~rank
        ~default:0
        (List.combine (Shape.kept_dims ~rank axes) (Array.to_list (Shape.strides oshape)))
    in
    let out = full oshape z in
    let fo = flat out
    and fa = flat a in
    iter_offsets ~shape:ishape ~strides:into (fun i o ->
      Array1.unsafe_set fo o (op (Array1.unsafe_get fo o) (Array1.unsafe_get fa i)));
    out
  ;;

  let reduce_sum = reduce ~fn:"Ndarray.reduce_sum" ~init:0. ~op:( +. )

  let reduce_max =
    reduce ~fn:"Ndarray.reduce_max" ~init:Float.neg_infinity ~op:Float.max_num
  ;;

  let reshape ~shape:target a =
    if numel target <> size a
    then
      invalid
        "Ndarray.reshape: cannot reshape %s to %s"
        (shape_to_string (shape a))
        (shape_to_string target);
    let fa = flat a in
    init target (Array1.get fa)
  ;;

  let expand_dims ~axes a =
    check_axes ~fn:"Ndarray.expand_dims" ~rank:(rank a + List.length axes) axes;
    reshape ~shape:(Shape.insert_ones ~axes (shape a)) a
  ;;

  let outer_inner s axis =
    let prod lo hi = numel (Array.sub s lo (hi - lo)) in
    prod 0 axis, prod (axis + 1) (Array.length s)
  ;;

  let concatenate ~axis arrays =
    match arrays with
    | [] -> invalid "Ndarray.concatenate: no arrays"
    | first :: _ ->
      let s0 = shape first in
      let rank = Array.length s0 in
      if axis < 0 || axis >= rank
      then invalid "Ndarray.concatenate: axis %d out of range" axis;
      let off_axis s = Shape.set_dim ~axis 0 s in
      if List.exists (fun a -> off_axis (shape a) <> off_axis s0) arrays
      then invalid "Ndarray.concatenate: shapes differ off the axis";
      let total = List.fold_left (fun n a -> n + (shape a).(axis)) 0 arrays in
      let oshape = Shape.set_dim ~axis total s0 in
      let outer, inner = outer_inner oshape axis in
      let runs = List.map (fun a -> flat a, (shape a).(axis) * inner) arrays in
      let row = total * inner in
      let out = create oshape in
      let fo = flat out in
      let copy_row o =
        let copy_from d (src, len) =
          copy_run src (o * len) fo d len;
          d + len
        in
        ignore (List.fold_left copy_from (o * row) runs : int)
      in
      List.iter copy_row (Lists.range outer);
      out
  ;;

  let slice ~axis ~start ~stop a =
    let s = shape a in
    if axis < 0 || axis >= Array.length s
    then invalid "Ndarray.slice: axis %d out of range" axis;
    if start < 0 || stop < start || stop > s.(axis)
    then invalid "Ndarray.slice: [%d, %d) out of range for size %d" start stop s.(axis);
    let outer, inner = outer_inner s axis in
    let len = (stop - start) * inner in
    let out = create (Shape.set_dim ~axis (stop - start) s) in
    let fo = flat out
    and fa = flat a in
    List.iter
      (fun o -> copy_run fa (((o * s.(axis)) + start) * inner) fo (o * len) len)
      (Lists.range outer);
    out
  ;;

  let broadcast_gen ~fn ~axes target a =
    let s = shape a in
    let m = Array.length target in
    let fail () =
      invalid
        "%s: cannot broadcast %s to %s"
        fn
        (shape_to_string s)
        (shape_to_string target)
    in
    if Array.length s + List.length axes <> m then fail ();
    check_axes ~fn ~rank:m axes;
    let source =
      List.combine (Shape.kept_dims ~rank:m axes) (Lists.range (Array.length s))
    in
    let sst = Shape.strides s in
    let stride d =
      match List.assoc_opt d source with
      | None -> 0
      | Some k when s.(k) = 1 -> 0
      | Some k when s.(k) = target.(d) -> sst.(k)
      | Some _ -> fail ()
    in
    gather ~shape:target ~strides:(Array.init m stride) a
  ;;

  let broadcast_to ~shape:target a =
    broadcast_gen ~fn:"Ndarray.broadcast_to" ~axes:[] target a
  ;;

  let broadcast ~shape:target ~axes a =
    broadcast_gen ~fn:"Ndarray.broadcast" ~axes target a
  ;;

  let transpose ~perm a =
    let s = shape a in
    if not (Shape.is_perm ~rank:(Array.length s) perm)
    then invalid "Ndarray.transpose: invalid permutation for rank %d" (Array.length s);
    let st = Shape.strides s in
    gather
      ~shape:(Array.map (Array.get s) perm)
      ~strides:(Array.map (Array.get st) perm)
      a
  ;;

  let allclose ?(rtol = 1e-5) ?(atol = 1e-8) a b =
    shape a = shape b
    && Array.for_all2
         (fun x y -> Float.abs (x -. y) <= atol +. (rtol *. Float.abs y))
         (to_array a)
         (to_array b)
  ;;

  let float_repr x =
    let rec shortest p =
      let s = Printf.sprintf "%.*g" p x in
      if p >= 17 || float_of_string s = x then s else shortest (p + 1)
    in
    match Float.classify_float x with
    | FP_nan -> "nan"
    | FP_infinite -> if x > 0. then "inf" else "-inf"
    | _ when Float.is_integer x && Float.abs x < 1e16 -> Printf.sprintf "%.1f" x
    | _ -> shortest 1
  ;;

  let to_string a =
    let s = shape a in
    let st = Shape.strides s
    and fa = flat a in
    let rec show d off =
      if d = Array.length s
      then float_repr (Array1.get fa off)
      else
        "["
        ^ String.concat " " (List.init s.(d) (fun i -> show (d + 1) (off + (i * st.(d)))))
        ^ "]"
    in
    show 0 0
  ;;

  let pp fmt a = Format.pp_print_string fmt (to_string a)
end

module Aval = struct
  type dtype = F64

  type t =
    { shape : int array
    ; dtype : dtype
    }

  let create ?(dtype = F64) shape = { shape; dtype }

  let equal a b =
    a.dtype = b.dtype
    && Array.length a.shape = Array.length b.shape
    && Array.for_all2 Int.equal a.shape b.shape
  ;;

  let of_ndarray a = create (Ndarray.shape a)

  let dtype_name = function
    | F64 -> "f64"
  ;;

  let to_string a = dtype_name a.dtype ^ Shape.to_string a.shape
end
