open Tracer
open Common

let%ops f x = (x *. sin x) +. 1.
let%ops norm2 x = reduce_sum (x *. x)
let%ops newton_step g x = x -. (g x /. Ad.deriv g x)
let%ops leaky x = if to_float x > 0. then x else 0.1 *. x

let arrays () =
  title "arrays";
  let x = Ops.array [| 0.; 1.; 2. |] in
  let m = Ops.array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  print "x" x;
  print "m" m;
  print "m *. x (broadcast)" [%ops m *. x];
  print "sum of m over axis 0" (Ops.reduce_sum ~axes:[ 0 ] m);
  print "matmul m x" (Ops.matmul m x);
  x, m
;;

let derivatives x =
  title "f x = x sin x + 1, and its derivatives";
  print "f x" (f x);
  print "deriv f x" (Ad.deriv f x);
  print "by hand: sin x + x cos x" [%ops sin x +. (x *. cos x)];
  print "deriv (deriv f) x" (Ad.deriv (Ad.deriv f) x)
;;

let gradient x =
  title "gradient of a scalar function";
  print "norm2 x" (norm2 x);
  print "grad norm2 x = 2x" (Ad.grad norm2 x)
;;

let batching m =
  title "vmap: map a function over a batch dim";
  print "vmap norm2 m (per row)" (Vmap.vmap norm2 m);
  print "vmap (grad norm2) m = 2m" (Vmap.vmap (Ad.grad norm2) m)
;;

let newton () =
  title "Newton's method for cos x = x";
  let%ops g x = cos x -. x in
  let n = 4 in
  let iterates = Seq.iterate (newton_step g) (Ops.scalar 1.) in
  let steps = iterates |> Seq.drop 1 |> Seq.take n |> List.of_seq in
  steps |> List.iteri (fun i x -> print (Printf.sprintf "step %d" (i + 1)) x);
  List.nth steps (n - 1) |> g |> print "cos x - x"
;;

let branching () =
  title "f may branch on its argument's value";
  Ops.scalar 2. |> Ad.deriv leaky |> print "deriv leaky 2";
  Ops.scalar (-2.) |> Ad.deriv leaky |> print "deriv leaky (-2)"
;;

let run () =
  let x, m = arrays () in
  derivatives x;
  gradient x;
  batching m;
  newton ();
  branching ()
;;

let example = { name = "basics"; summary = "arrays, deriv, grad, Newton's method"; run }
