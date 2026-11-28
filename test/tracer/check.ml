open Tracer

let failures = ref 0

let report name ok detail =
  if ok
  then Printf.printf "ok   %s\n" name
  else (
    incr failures;
    Printf.printf "FAIL %s: %s\n" name detail)
;;

let bool name b = report name b "condition is false"

let float ?(rtol = 1e-12) ?(atol = 1e-12) name ~expected got =
  report
    name
    (Float.abs (got -. expected) <= atol +. (rtol *. Float.abs expected))
    (Printf.sprintf "expected %.17g, got %.17g" expected got)
;;

let array ?(rtol = 1e-12) ?(atol = 1e-12) name ~expected got =
  report
    name
    (Ndarray.allclose ~rtol ~atol got expected)
    (Printf.sprintf
       "expected %s, got %s"
       (Ndarray.to_string expected)
       (Ndarray.to_string got))
;;

let value ?rtol ?atol name ~expected got =
  array ?rtol ?atol name ~expected:(Ops.to_ndarray expected) (Ops.to_ndarray got)
;;

let rand ?(lo = 0.5) ?(hi = 2.0) shape =
  Ndarray.init shape (fun _ -> lo +. Random.float (hi -. lo)) |> Ops.of_ndarray
;;

let fd_check ?(eps = 1e-6) name f xs =
  let dxs = List.map (fun x -> rand ~lo:(-1.) ~hi:1. (Ops.shape x)) xs in
  let vs = Tree.(list value) in
  let _, tangents = Ad.jvp_tree ~input:vs ~output:vs f xs dxs in
  let at t = List.map2 (fun x dx -> Ops.(x + (scalar t * dx))) xs dxs |> f in
  let central p m = Ops.((p - m) / scalar (2. *. eps)) in
  let fd = List.map2 central (at eps) (at (-.eps)) in
  List.iter2 (fun t e -> value ~rtol:1e-6 ~atol:1e-6 name ~expected:e t) tangents fd
;;

let un name op x =
  fd_check
    ("jvp " ^ name)
    (function
      | [ x ] -> [ op x ]
      | _ -> assert false)
    [ x ]
;;

let bin name op x y =
  fd_check
    ("jvp " ^ name)
    (function
      | [ x; y ] -> [ op x y ]
      | _ -> assert false)
    [ x; y ]
;;

let raises ?(matches = fun _ -> true) name f =
  match f () with
  | _ -> report name false "expected an exception"
  | exception e -> report name (matches e) ("unexpected exception " ^ Printexc.to_string e)
;;

let is_type_error = function
  | Error (Type_error _) -> true
  | _ -> false
;;

let is_escaped = function
  | Error (Escaped_tracer _) -> true
  | _ -> false
;;

let is_concretization = function
  | Error (Concretization_error _) -> true
  | _ -> false
;;

let contains s sub =
  let n = String.length sub in
  let rec go i = i + n <= String.length s && (String.sub s i n = sub || go (i + 1)) in
  go 0
;;

let is_missing_rule = function
  | Error (Missing_rule _) -> true
  | _ -> false
;;

let finish () =
  if !failures > 0
  then (
    Printf.printf "%d check(s) failed\n" !failures;
    exit 1)
;;
