module Error = struct
  exception Type_error of string
  exception Escaped_tracer of string
  exception Missing_rule of string
  exception Concretization_error of string

  let () =
    Printexc.register_printer (function
      | Type_error msg -> Some ("Type_error: " ^ msg)
      | Escaped_tracer msg -> Some ("Escaped_tracer: " ^ msg)
      | Missing_rule msg -> Some ("Missing_rule: " ^ msg)
      | Concretization_error msg -> Some ("Concretization_error: " ^ msg)
      | _ -> None)
  ;;

  let type_error fmt = Printf.ksprintf (fun msg -> raise (Type_error msg)) fmt
end

module Debug = struct
  let enabled =
    ref
      (match Sys.getenv_opt "TINY_ML_DEBUG" with
       | None | Some ("" | "0" | "false") -> false
       | Some _ -> true)
  ;;
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

  let reaches ~num_ids ~inputs ~outputs ~roots nodes =
    let live = Array.make num_ids false in
    List.iter (fun i -> live.(i) <- true) roots;
    List.map
      (fun n ->
         let r = List.exists (fun i -> live.(i)) (outputs n) in
         if r then List.iter (fun i -> live.(i) <- true) (inputs n);
         r)
      nodes
  ;;
end

module Shape = struct
  let numel shape = Array.fold_left ( * ) 1 shape

  let to_string s =
    "[" ^ String.concat "," (Array.to_list (Array.map string_of_int s)) ^ "]"
  ;;

  (* e.g. "f64[2,3]"; f64 is the only dtype. *)
  let describe s = "f64" ^ to_string s

  let strides shape =
    let n = Array.length shape in
    let s = Array.make n 1 in
    for d = n - 2 downto 0 do
      s.(d) <- s.(d + 1) * shape.(d + 1)
    done;
    s
  ;;

  let kept_dims ~rank axes =
    List.filter (fun d -> not (List.mem d axes)) (List.init rank Fun.id)
  ;;

  let remove_dims ~axes s =
    Array.of_list (List.map (fun d -> s.(d)) (kept_dims ~rank:(Array.length s) axes))
  ;;

  let insert_ones ~axes s =
    let rank = Array.length s + List.length axes in
    let out = Array.make rank 1 in
    List.iteri (fun k d -> out.(d) <- s.(k)) (kept_dims ~rank axes);
    out
  ;;

  let axes_error ~rank axes =
    match List.find_opt (fun ax -> ax < 0 || ax >= rank) axes with
    | Some ax -> Some (Printf.sprintf "axis %d out of range for rank %d" ax rank)
    | None ->
      if List.length (List.sort_uniq compare axes) <> List.length axes
      then Some "duplicate axes"
      else None
  ;;

  let is_perm ~rank perm =
    Array.length perm = rank
    && List.sort compare (Array.to_list perm) = List.init rank Fun.id
  ;;

  let normalize_axes ~rank axes =
    List.sort compare (List.map (fun a -> if a < 0 then a + rank else a) axes)
  ;;

  let broadcast_shapes a b =
    let n = max (Array.length a) (Array.length b) in
    let dim s i =
      let k = i - (n - Array.length s) in
      if k < 0 then 1 else s.(k)
    in
    let out = Array.make n 0 in
    let ok = ref true in
    for i = 0 to n - 1 do
      let x = dim a i
      and y = dim b i in
      if x = y || y = 1 then out.(i) <- x else if x = 1 then out.(i) <- y else ok := false
    done;
    if !ok then Some out else None
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

  let init shape f : t =
    let a = Genarray.create float64 c_layout shape in
    let fa = flat a in
    for i = 0 to Array1.dim fa - 1 do
      Array1.unsafe_set fa i (f i)
    done;
    a
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

  let strides = Shape.strides

  (* Calls [f i off] in row-major order;
     an odometer, so no division per element. *)
  let iter_offsets ~shape ~strides f =
    let n = Array.length shape in
    let total = numel shape in
    if n = 0
    then f 0 0
    else if total > 0
    then (
      let last = shape.(n - 1)
      and sl = strides.(n - 1) in
      let idx = Array.make n 0 in
      let off = ref 0 in
      let i = ref 0 in
      while !i < total do
        let o = !off
        and i0 = !i in
        for j = 0 to last - 1 do
          f (i0 + j) (o + (j * sl))
        done;
        i := i0 + last;
        let d = ref (n - 2) in
        while !d >= 0 do
          let k = !d in
          off := !off + strides.(k);
          idx.(k) <- idx.(k) + 1;
          if idx.(k) < shape.(k)
          then d := -1
          else (
            off := !off - (shape.(k) * strides.(k));
            idx.(k) <- 0;
            decr d)
        done
      done)
  ;;

  let gather ~shape ~strides src : t =
    let out = Genarray.create float64 c_layout shape in
    let fo = flat out
    and fs = flat src in
    iter_offsets ~shape ~strides (fun i off ->
      Array1.unsafe_set fo i (Array1.unsafe_get fs off));
    out
  ;;

  (* Short runs loop: [Array1.sub] allocates. *)
  let copy_run (src : (float, float64_elt, c_layout) Array1.t) so dst d len =
    if len >= 64
    then Array1.blit (Array1.sub src so len) (Array1.sub dst d len)
    else
      for j = 0 to len - 1 do
        Array1.unsafe_set dst (d + j) (Array1.unsafe_get src (so + j))
      done
  ;;

  let outer_inner s axis =
    let prod lo hi = Array.fold_left ( * ) 1 (Array.sub s lo (hi - lo)) in
    prod 0 axis, prod (axis + 1) (Array.length s)
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

  let greater = map2 (fun x y -> if x > y then 1. else 0.)
  let less = map2 (fun x y -> if x < y then 1. else 0.)
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
      init [| m; n |] (fun o ->
        let i = o / n
        and j = o mod n in
        let acc = ref 0. in
        for p = 0 to k - 1 do
          acc
          := !acc
             +. (Array1.unsafe_get fa ((i * k) + p) *. Array1.unsafe_get fb ((p * n) + j))
        done;
        !acc)
    | sa, sb ->
      invalid
        "Ndarray.dot: expected [m; k] and [k; n], got %s and %s"
        (shape_to_string sa)
        (shape_to_string sb)
  ;;

  let reduce ~fn ~init:z ~op ~axes a =
    let ishape = shape a in
    let n = Array.length ishape in
    check_axes ~fn ~rank:n axes;
    let kept = Shape.kept_dims ~rank:n axes in
    let oshape = Shape.remove_dims ~axes ishape in
    let ostrides = strides oshape in
    let st = Array.make n 0 in
    List.iteri (fun k d -> st.(d) <- ostrides.(k)) kept;
    let out = full oshape z in
    let fo = flat out
    and fa = flat a in
    iter_offsets ~shape:ishape ~strides:st (fun i o ->
      Array1.unsafe_set fo o (op (Array1.unsafe_get fo o) (Array1.unsafe_get fa i)));
    out
  ;;

  let reduce_sum = reduce ~fn:"Ndarray.reduce_sum" ~init:0. ~op:( +. )

  let reduce_max =
    reduce ~fn:"Ndarray.reduce_max" ~init:Float.neg_infinity ~op:Float.max_num
  ;;

  let reshape a target =
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
    reshape a (Shape.insert_ones ~axes (shape a))
  ;;

  let concatenate ~axis arrays =
    match arrays with
    | [] -> invalid "Ndarray.concatenate: no arrays"
    | first :: _ ->
      let s0 = shape first in
      let n = Array.length s0 in
      if axis < 0 || axis >= n
      then invalid "Ndarray.concatenate: axis %d out of range" axis;
      List.iter
        (fun a ->
           let s = shape a in
           if
             Array.length s <> n
             || Array.exists Fun.id (Array.mapi (fun d x -> d <> axis && x <> s0.(d)) s)
           then invalid "Ndarray.concatenate: shapes differ off the axis")
        arrays;
      let sizes = List.map (fun a -> (shape a).(axis)) arrays in
      let oshape = Array.copy s0 in
      oshape.(axis) <- List.fold_left ( + ) 0 sizes;
      let outer, inner = outer_inner oshape axis in
      let out = Genarray.create float64 c_layout oshape in
      let fo = flat out
      and runs = List.map (fun a -> flat a, (shape a).(axis) * inner) arrays in
      let d = ref 0 in
      for o = 0 to outer - 1 do
        List.iter
          (fun (fp, len) ->
             copy_run fp (o * len) fo !d len;
             d := !d + len)
          runs
      done;
      out
  ;;

  let slice ~axis ~start ~stop a =
    let s = shape a in
    let n = Array.length s in
    if axis < 0 || axis >= n then invalid "Ndarray.slice: axis %d out of range" axis;
    if start < 0 || stop < start || stop > s.(axis)
    then invalid "Ndarray.slice: [%d, %d) out of range for size %d" start stop s.(axis);
    let oshape = Array.copy s in
    oshape.(axis) <- stop - start;
    let outer, inner = outer_inner s axis in
    let len = (stop - start) * inner in
    let out = Genarray.create float64 c_layout oshape in
    let fo = flat out
    and fa = flat a in
    for o = 0 to outer - 1 do
      copy_run fa (((o * s.(axis)) + start) * inner) fo (o * len) len
    done;
    out
  ;;

  (* New axes and size-1 dims get stride 0. *)
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
    let sst = strides s in
    let st = Array.make m 0 in
    List.iteri
      (fun k d ->
         if s.(k) <> 1 then if s.(k) = target.(d) then st.(d) <- sst.(k) else fail ())
      (Shape.kept_dims ~rank:m axes);
    gather ~shape:target ~strides:st a
  ;;

  let broadcast_to target a = broadcast_gen ~fn:"Ndarray.broadcast_to" ~axes:[] target a

  let broadcast ~shape:target ~axes a =
    broadcast_gen ~fn:"Ndarray.broadcast" ~axes target a
  ;;

  let transpose ~perm a =
    let s = shape a in
    let n = Array.length s in
    if not (Shape.is_perm ~rank:n perm)
    then invalid "Ndarray.transpose: invalid permutation for rank %d" n;
    let oshape = Array.map (fun p -> s.(p)) perm in
    let st = strides s in
    let st = Array.map (fun p -> st.(p)) perm in
    gather ~shape:oshape ~strides:st a
  ;;

  let allclose ?(rtol = 1e-5) ?(atol = 1e-8) a b =
    shape a = shape b
    && Array.for_all2
         (fun x y -> Float.abs (x -. y) <= atol +. (rtol *. Float.abs y))
         (to_array a)
         (to_array b)
  ;;

  let float_repr x =
    if Float.is_nan x
    then "nan"
    else if Float.is_integer x && Float.abs x < 1e16
    then Printf.sprintf "%.1f" x
    else if Float.is_finite x
    then (
      let rec go p =
        let s = Printf.sprintf "%.*g" p x in
        if p >= 17 || float_of_string s = x then s else go (p + 1)
      in
      go 1)
    else if x > 0.
    then "inf"
    else "-inf"
  ;;

  let to_string a =
    let s = shape a in
    let st = strides s
    and fa = flat a in
    let buf = Buffer.create 64 in
    let rec go d off =
      if d = Array.length s
      then Buffer.add_string buf (float_repr (Array1.get fa off))
      else (
        Buffer.add_char buf '[';
        for i = 0 to s.(d) - 1 do
          if i > 0 then Buffer.add_char buf ' ';
          go (d + 1) (off + (i * st.(d)))
        done;
        Buffer.add_char buf ']')
    in
    go 0 0;
    Buffer.contents buf
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
  let of_ndarray a = create (Ndarray.shape a)

  let dtype_name = function
    | F64 -> "f64"
  ;;

  let to_string a = dtype_name a.dtype ^ Shape.to_string a.shape
end
