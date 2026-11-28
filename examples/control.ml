open Tracer
open Common

let%ops huber x =
  Control.cond
    (Float.abs x <=. 1.)
    ~then_:(fun x -> 0.5 *. x *. x)
    ~else_:(fun x -> Float.abs x -. 0.5)
    x
;;

let%ops newton_sqrt a =
  Control.while_loop_tree
    ~carry:Tree.(pair value value)
    ~cond:(fun (x, _) -> Float.abs ((x *. x) -. a) >. 1e-12)
    ~body:(fun (x, n) -> 0.5 *. (x +. (a /. x)), n +. 1.)
    (a, Ops.scalar 0.)
;;

let root a = newton_sqrt a |> fst

let conds () =
  title "cond: both branches are traced, one runs";
  [ 0.5; -3. ]
  |> List.iter (fun v ->
    let x = Ops.scalar v in
    print (Printf.sprintf "huber %g" v) (huber x);
    print (Printf.sprintf "grad huber %g" v) (Ad.grad huber x));
  title "cond under jit: one trace serves both branches";
  let fast = Compile.compile ~backend:noisy_jit huber in
  Ops.scalar 0.5 |> fast |> print "fast 0.5";
  Ops.scalar (-3.) |> fast |> print "fast -3";
  Ops.scalar (-3.) |> Ad.grad fast |> print "grad fast -3";
  title "the graph: cond holds its branches";
  Ir.make huber (Ops.scalar 0.) |> show_graph
;;

let loops () =
  title "while_loop: Newton's method for sqrt";
  [ 2.; 100. ]
  |> List.iter (fun v ->
    let r, steps = newton_sqrt (Ops.scalar v) in
    print (Printf.sprintf "sqrt %g" v) r;
    print "  steps" steps);
  title "a loop under jit: the trip count is decided at run time";
  let fast_root = Compile.compile ~backend:noisy_jit root in
  Ops.scalar 2. |> fast_root |> print "fast_root 2";
  Ops.scalar 100. |> fast_root |> print "fast_root 100";
  title "derivatives of a loop: forward mode";
  Ops.scalar 2. |> Ad.deriv root |> print "deriv root 2";
  Ops.scalar (1. /. (2. *. Float.sqrt 2.)) |> print "by hand: 1 / (2 sqrt 2)";
  Ops.scalar 2. |> Ad.deriv fast_root |> print "deriv fast_root 2";
  title "reverse mode needs a fixed trip count";
  match catch (fun () -> Ad.grad root (Ops.scalar 2.)) with
  | Ok _ -> ()
  | Error (Missing_rule msg) -> Printf.printf "  grad root: %s\n" msg
  | Error e -> failwith @@ error_to_string e
;;

let run () =
  conds ();
  loops ()
;;

let example = { name = "control"; summary = "cond and while_loop on data"; run }
