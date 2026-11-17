open Tracer

let nd shape data = Ndarray.of_array shape data
let a23 = nd [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]

let unravel shape i =
  let idx = Array.make (Array.length shape) 0
  and rest = ref i in
  for d = Array.length shape - 1 downto 0 do
    idx.(d) <- !rest mod shape.(d);
    rest := !rest / shape.(d)
  done;
  idx
;;

let ref_init shape f = Ndarray.init shape (fun i -> f (unravel shape i))
let get = Bigarray.Genarray.get
let iota shape = Ndarray.init shape (fun i -> float (i + 1))

let ref_transpose ~perm a =
  let s = Ndarray.shape a in
  ref_init
    (Array.map (fun p -> s.(p)) perm)
    (fun idx ->
       let src = Array.make (Array.length s) 0 in
       Array.iteri (fun k p -> src.(p) <- idx.(k)) perm;
       get a src)
;;

let ref_reduce_sum ~axes a =
  let s = Ndarray.shape a in
  let n = Array.length s in
  let kept = List.filter (fun d -> not (List.mem d axes)) (List.init n Fun.id) in
  let oshape = Array.of_list (List.map (fun d -> s.(d)) kept) in
  let out = Ndarray.to_array (Ndarray.zeros oshape) in
  for i = 0 to Ndarray.size a - 1 do
    let idx = unravel s i in
    let o = ref 0 in
    List.iter (fun d -> o := (!o * s.(d)) + idx.(d)) kept;
    out.(!o) <- out.(!o) +. get a idx
  done;
  Ndarray.of_array oshape out
;;

let ref_slice ~axis ~start ~stop a =
  let s = Array.copy (Ndarray.shape a) in
  s.(axis) <- stop - start;
  ref_init s (fun idx ->
    let src = Array.copy idx in
    src.(axis) <- src.(axis) + start;
    get a src)
;;

let ref_concatenate ~axis arrays =
  let s = Array.copy (Ndarray.shape (List.hd arrays)) in
  s.(axis) <- List.fold_left (fun acc a -> acc + (Ndarray.shape a).(axis)) 0 arrays;
  ref_init s (fun idx ->
    let rec find j = function
      | a :: rest ->
        let k = (Ndarray.shape a).(axis) in
        if j < k then a, j else find (j - k) rest
      | [] -> assert false
    in
    let a, j = find idx.(axis) arrays in
    let src = Array.copy idx in
    src.(axis) <- j;
    get a src)
;;

let exact name ~expected got =
  Check.bool
    name
    (Ndarray.shape got = Ndarray.shape expected
     && Ndarray.to_array got = Ndarray.to_array expected)
;;

let () =
  Check.bool "shape" (Ndarray.shape a23 = [| 2; 3 |]);
  Check.bool
    "scalar is rank 0, size 1"
    (Ndarray.rank (Ndarray.scalar 1.) = 0 && Ndarray.size (Ndarray.scalar 1.) = 1);
  Check.float "to_float" ~expected:3.5 (Ndarray.to_float (Ndarray.scalar 3.5));
  Check.raises "to_float rejects non-scalar" (fun () -> Ndarray.to_float a23);
  Check.raises "of_array rejects wrong size" (fun () -> nd [| 2; 2 |] [| 1. |]);
  Check.array
    "zeros"
    ~expected:(nd [| 2; 2 |] [| 0.; 0.; 0.; 0. |])
    (Ndarray.zeros [| 2; 2 |]);
  Check.array
    "map"
    ~expected:(nd [| 2; 3 |] [| 2.; 4.; 6.; 8.; 10.; 12. |])
    (Ndarray.map (fun x -> 2. *. x) a23);
  Check.array
    "map2"
    ~expected:(nd [| 2; 3 |] [| 2.; 4.; 6.; 8.; 10.; 12. |])
    (Ndarray.map2 ( +. ) a23 a23);
  Check.raises "map2 rejects shape mismatch" (fun () ->
    Ndarray.map2 ( +. ) a23 (Ndarray.zeros [| 3; 2 |]));
  Check.array
    "greater"
    ~expected:(nd [| 3 |] [| 0.; 1.; 0. |])
    (Ndarray.greater (nd [| 3 |] [| 1.; 5.; 3. |]) (nd [| 3 |] [| 2.; 2.; 3. |]));
  Check.array
    "less"
    ~expected:(nd [| 3 |] [| 1.; 0.; 0. |])
    (Ndarray.less (nd [| 3 |] [| 1.; 5.; 3. |]) (nd [| 3 |] [| 2.; 2.; 3. |]));
  Check.array
    "reduce_sum axes=[0]"
    ~expected:(nd [| 3 |] [| 5.; 7.; 9. |])
    (Ndarray.reduce_sum ~axes:[ 0 ] a23);
  Check.array
    "reduce_sum axes=[1]"
    ~expected:(nd [| 2 |] [| 6.; 15. |])
    (Ndarray.reduce_sum ~axes:[ 1 ] a23);
  Check.array
    "reduce_sum axes=[0;1]"
    ~expected:(Ndarray.scalar 21.)
    (Ndarray.reduce_sum ~axes:[ 0; 1 ] a23);
  Check.array "reduce_sum axes=[]" ~expected:a23 (Ndarray.reduce_sum ~axes:[] a23);
  Check.array
    "reduce_sum 3D middle axis"
    ~expected:(nd [| 2; 2 |] [| 4.; 6.; 12.; 14. |])
    (Ndarray.reduce_sum
       ~axes:[ 1 ]
       (nd [| 2; 2; 2 |] (Array.init 8 (fun i -> float (i + 1)))));
  Check.array
    "dot [2,3] x [3,2]"
    ~expected:(nd [| 2; 2 |] [| 4.; 5.; 10.; 11. |])
    (Ndarray.dot a23 (nd [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |]));
  Check.raises "dot rejects misaligned shapes" (fun () -> Ndarray.dot a23 a23);
  Check.raises "dot rejects a vector" (fun () ->
    Ndarray.dot a23 (nd [| 3 |] [| 1.; 2.; 3. |]));
  Check.raises "reduce_sum rejects bad axis" (fun () ->
    Ndarray.reduce_sum ~axes:[ 2 ] a23);
  Check.bool
    "expand_dims shape"
    (Ndarray.shape (Ndarray.expand_dims ~axes:[ 0; 2 ] (nd [| 3 |] [| 1.; 2.; 3. |]))
     = [| 1; 3; 1 |]);
  Check.array
    "broadcast scalar"
    ~expected:(Ndarray.full [| 2; 3 |] 7.)
    (Ndarray.broadcast ~shape:[| 2; 3 |] ~axes:[ 0; 1 ] (Ndarray.scalar 7.));
  Check.array
    "broadcast new leading axis"
    ~expected:(nd [| 2; 3 |] [| 1.; 2.; 3.; 1.; 2.; 3. |])
    (Ndarray.broadcast ~shape:[| 2; 3 |] ~axes:[ 0 ] (nd [| 3 |] [| 1.; 2.; 3. |]));
  Check.array
    "broadcast new trailing axis"
    ~expected:(nd [| 2; 3 |] [| 1.; 1.; 1.; 2.; 2.; 2. |])
    (Ndarray.broadcast ~shape:[| 2; 3 |] ~axes:[ 1 ] (nd [| 2 |] [| 1.; 2. |]));
  Check.array
    "broadcast_to stretches size-1 dim"
    ~expected:(nd [| 2; 3 |] [| 1.; 1.; 1.; 2.; 2.; 2. |])
    (Ndarray.broadcast_to [| 2; 3 |] (nd [| 2; 1 |] [| 1.; 2. |]));
  Check.raises "broadcast_to rejects incompatible shape" (fun () ->
    Ndarray.broadcast_to [| 2; 3 |] (nd [| 2; 2 |] [| 1.; 2.; 3.; 4. |]));
  Check.array
    "transpose 2D"
    ~expected:(nd [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |])
    (Ndarray.transpose ~perm:[| 1; 0 |] a23);
  (let x = nd [| 2; 3; 4 |] (Array.init 24 float) in
   let t = Ndarray.transpose ~perm:[| 2; 0; 1 |] x in
   let ok = ref (Ndarray.shape t = [| 4; 2; 3 |]) in
   for i = 0 to 1 do
     for j = 0 to 2 do
       for k = 0 to 3 do
         if Bigarray.Genarray.get t [| k; i; j |] <> float ((i * 12) + (j * 4) + k)
         then ok := false
       done
     done
   done;
   Check.bool "transpose 3D perm=[2;0;1]: out[k][i][j] = in[i][j][k]" !ok);
  Check.raises "transpose rejects non-permutation" (fun () ->
    Ndarray.transpose ~perm:[| 0; 0 |] a23);
  let x4 = iota [| 3; 4; 5; 6 |] in
  List.iter
    (fun perm ->
       exact
         (Printf.sprintf "transpose 4D perm=%s" (Ndarray.shape_to_string perm))
         ~expected:(ref_transpose ~perm x4)
         (Ndarray.transpose ~perm x4))
    [ [| 3; 1; 0; 2 |]; [| 0; 1; 3; 2 |]; [| 0; 1; 2; 3 |]; [| 3; 2; 1; 0 |] ];
  List.iter
    (fun axes ->
       exact
         (Printf.sprintf
            "reduce_sum 4D axes=%s"
            (Ndarray.shape_to_string (Array.of_list axes)))
         ~expected:(ref_reduce_sum ~axes x4)
         (Ndarray.reduce_sum ~axes x4))
    [ [ 0 ]; [ 3 ]; [ 0; 2 ]; [ 1; 3 ]; [ 1; 2 ]; [ 0; 1; 2; 3 ] ];
  exact
    "broadcast 4D new axes and size-1 dim"
    ~expected:
      (ref_init [| 2; 3; 4; 5 |] (fun idx -> float ((idx.(1) * 10) + idx.(3) + 1)))
    (Ndarray.broadcast
       ~shape:[| 2; 3; 4; 5 |]
       ~axes:[ 0; 2 ]
       (Ndarray.init [| 3; 5 |] (fun i -> float ((i / 5 * 10) + (i mod 5) + 1))));
  let big = iota [| 4; 50; 30 |] in
  List.iter
    (fun (axis, start, stop) ->
       let name = Printf.sprintf "slice axis=%d [%d, %d)" axis start stop in
       exact
         name
         ~expected:(ref_slice ~axis ~start ~stop x4)
         (Ndarray.slice ~axis ~start ~stop x4);
       exact
         (name ^ " long runs")
         ~expected:(ref_slice ~axis ~start ~stop big)
         (Ndarray.slice ~axis ~start ~stop big))
    [ 0, 1, 3; 1, 1, 3; 2, 2, 4; 2, 0, 0; 1, 0, 4 ];
  let parts axis sizes =
    List.mapi
      (fun k m ->
         let s = Array.copy (Ndarray.shape x4) in
         s.(axis) <- m;
         Ndarray.map (fun x -> x +. (1000. *. float k)) (iota s))
      sizes
  in
  List.iter
    (fun (axis, sizes) ->
       let arrays = parts axis sizes in
       exact
         (Printf.sprintf
            "concatenate axis=%d sizes=%s"
            axis
            (Ndarray.shape_to_string (Array.of_list sizes)))
         ~expected:(ref_concatenate ~axis arrays)
         (Ndarray.concatenate ~axis arrays))
    [ 0, [ 1; 2; 3 ]; 1, [ 2; 0; 3 ]; 2, [ 4; 1 ]; 3, [ 6; 6; 1 ]; 1, [ 40; 25 ] ];
  let s = Ndarray.scalar 2.5 in
  exact "transpose rank 0" ~expected:s (Ndarray.transpose ~perm:[||] s);
  exact "reduce_sum rank 0" ~expected:s (Ndarray.reduce_sum ~axes:[] s);
  exact "broadcast_to rank 0" ~expected:s (Ndarray.broadcast_to [||] s);
  exact
    "transpose zero-size"
    ~expected:(Ndarray.zeros [| 3; 0; 2 |])
    (Ndarray.transpose ~perm:[| 2; 1; 0 |] (Ndarray.zeros [| 2; 0; 3 |]));
  exact
    "reduce_sum over a zero-size axis"
    ~expected:(Ndarray.zeros [| 2; 3 |])
    (Ndarray.reduce_sum ~axes:[ 1 ] (Ndarray.zeros [| 2; 0; 3 |]));
  exact
    "reduce_sum of zero-size keeping it"
    ~expected:(Ndarray.zeros [| 0 |])
    (Ndarray.reduce_sum ~axes:[ 1 ] (Ndarray.zeros [| 0; 3 |]));
  exact
    "broadcast to zero-size"
    ~expected:(Ndarray.zeros [| 0; 3 |])
    (Ndarray.broadcast ~shape:[| 0; 3 |] ~axes:[ 0 ] (iota [| 3 |]));
  exact
    "slice of zero-size"
    ~expected:(Ndarray.zeros [| 0; 2 |])
    (Ndarray.slice ~axis:1 ~start:1 ~stop:3 (Ndarray.zeros [| 0; 4 |]));
  exact
    "concatenate zero-size"
    ~expected:(Ndarray.zeros [| 0; 5 |])
    (Ndarray.concatenate ~axis:1 [ Ndarray.zeros [| 0; 2 |]; Ndarray.zeros [| 0; 3 |] ]);
  (let y = Ndarray.slice ~axis:1 ~start:0 ~stop:50 big in
   Bigarray.Genarray.set y [| 0; 0; 0 |] (-1.);
   Check.bool "slice returns a fresh array" (get big [| 0; 0; 0 |] = 1.));
  List.iter
    (fun (x, s) ->
       Check.bool (Printf.sprintf "float_repr %s" s) (Ndarray.float_repr x = s))
    [ 2., "2.0"
    ; 0.1, "0.1"
    ; -0.5, "-0.5"
    ; 1e-5, "1e-05"
    ; 2.7177599838802657, "2.7177599838802657"
    ];
  Check.bool "to_string" (Ndarray.to_string a23 = "[[1.0 2.0 3.0] [4.0 5.0 6.0]]");
  Check.finish ()
;;
