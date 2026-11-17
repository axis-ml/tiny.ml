(* Arrays, a function written once, and its derivatives. *)

open Tracer
open Common

(* Inside let%ops, float syntax works on arrays. *)
let%ops f x = (x *. sin x) +. 1.

let run () =
  title "arrays";
  let x = Ops.array [| 0.; 1.; 2. |] in
  let m = Ops.array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  print "x" x;
  print "m" m;
  print "m *. x (broadcast)" [%ops m *. x];
  print "sum of m over axis 0" (Ops.reduce_sum ~axes:[ 0 ] m);
  print "matmul m x" (Ops.matmul m x);
  title "f x = x sin x + 1, and its derivatives";
  print "f x" (f x);
  print "deriv f x" (Ad.deriv f x);
  print "by hand: sin x + x cos x" [%ops sin x +. (x *. cos x)];
  print "deriv (deriv f) x" (Ad.deriv (Ad.deriv f) x);
  title "gradient of a scalar function";
  let%ops norm2 x = reduce_sum (x *. x) in
  print "norm2 x" (norm2 x);
  print "grad norm2 x = 2x" (Ad.grad1 norm2 x);
  title "Newton's method for cos x = x";
  let%ops g x = cos x -. x in
  let x = ref (Ops.scalar 1.) in
  for i = 1 to 4 do
    x := [%ops !x -. (g !x /. Ad.deriv g !x)];
    print (Printf.sprintf "step %d" i) !x
  done;
  print "cos x - x" (g !x);
  title "f may branch on its argument's value";
  let%ops leaky x = if to_float x > 0. then x else 0.1 *. x in
  print "deriv leaky 2" (Ad.deriv leaky (Ops.scalar 2.));
  print "deriv leaky (-2)" (Ad.deriv leaky (Ops.scalar (-2.)))
;;

let example = { name = "basics"; summary = "arrays, deriv, grad, Newton's method"; run }
