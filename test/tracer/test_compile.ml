open Tracer

let s = Ops.scalar
let f x = Ops.(-(sin x * scalar 2.) + x)

let () =
  let g = Compile.compile1 f in
  Check.value "eager compile1 = f" ~expected:(f (s 3.)) (g (s 3.));
  let x = Ops.array [| 0.; 1.; 2. |] in
  Check.value "eager compile1 on arrays" ~expected:(f x) (g x);
  Check.value
    "deriv of compiled = deriv"
    ~expected:(Ad.deriv f (s 3.))
    (Ad.deriv g (s 3.));
  Check.value
    "nested deriv of compiled"
    ~expected:(Ad.deriv (Ad.deriv f) (s 3.))
    (Ad.deriv (Ad.deriv g) (s 3.))
;;

let () =
  let h (x, ys) = Ops.(List.fold_left ( + ) x ys), List.map Ops.sin ys in
  let g =
    Compile.compile Tree.(pair value (list value)) Tree.(pair value (list value)) h
  in
  List.iter
    (fun n ->
       let ys = List.init n (fun i -> s (float_of_int i)) in
       let sum, sins = g (s 10., ys) in
       let esum, esins = h (s 10., ys) in
       Check.value (Printf.sprintf "pytree sum, %d leaves" n) ~expected:esum sum;
       Check.bool
         (Printf.sprintf "pytree list, %d leaves" n)
         (List.length sins = n && List.for_all2 Ops.allclose sins esins))
    [ 0; 2; 5 ]
;;

let () =
  let prepared = ref 0
  and calls = ref 0 in
  let module Counting = struct
    let name = "counting"

    let compile f =
      incr prepared;
      fun xs ->
        incr calls;
        f xs
    ;;
  end
  in
  let g = Compile.compile1 ~backend:(module Counting) f in
  ignore (g (s 1.));
  ignore (g (s 2.));
  Check.bool "backend prepares once" (!prepared = 1);
  Check.bool "backend runs per call" (!calls = 2);
  Check.value "backend result" ~expected:(f (s 2.)) (g (s 2.))
;;

let () =
  let module Broken = struct
    let name = "broken"

    let compile f xs =
      let ys, out = f xs in
      List.tl ys, out
    ;;
  end
  in
  let g = Compile.compile1 ~backend:(module Broken) f in
  Check.bool
    "backend that drops a leaf raises Invalid_argument"
    (match g (s 1.) with
     | _ -> false
     | exception Invalid_argument _ -> true)
;;

let () = Check.finish ()
