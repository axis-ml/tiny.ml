(* Control flow on data: Control.cond and Control.while_loop. *)

open Tracer
open Common

(* Quadratic near 0, linear far out. *)
let%ops huber x =
  Control.cond1
    (Float.abs x <=. 1.)
    ~then_:(fun x -> 0.5 *. x *. x)
    ~else_:(fun x -> Float.abs x -. 0.5)
    x
;;

(* Newton's method for sqrt a; the carry is (x, steps). *)
let%ops newton_sqrt a =
  Control.while_loop
    Tree.(pair value value)
    ~cond:(fun (x, _) -> Float.abs ((x *. x) -. a) >. 1e-12)
    ~body:(fun (x, n) -> 0.5 *. (x +. (a /. x)), n +. 1.)
    (a, Ops.scalar 0.)
;;

let root a = fst (newton_sqrt a)

let run () =
  title "cond: both branches are traced, one runs";
  List.iter
    (fun v ->
       let x = Ops.scalar v in
       print (Printf.sprintf "huber %g" v) (huber x);
       print (Printf.sprintf "grad huber %g" v) (Ad.grad1 huber x))
    [ 0.5; -3. ];
  title "cond under jit: one trace serves both branches";
  let fast =
    Compile.compile1
      ~backend:(Compile.jit_with ~on_trace:(fun _ -> print_endline "  [traced]"))
      huber
  in
  print "fast 0.5" (fast (Ops.scalar 0.5));
  print "fast -3" (fast (Ops.scalar (-3.)));
  print "grad fast -3" (Ad.grad1 fast (Ops.scalar (-3.)));
  title "the graph: cond holds its branches";
  String.split_on_char '\n' (Ir.to_string (Ir.make1 huber (Ops.scalar 0.)))
  |> List.iter (fun l -> if l <> "" then Printf.printf "    %s\n" l);
  title "while_loop: Newton's method for sqrt";
  List.iter
    (fun v ->
       let r, steps = newton_sqrt (Ops.scalar v) in
       print (Printf.sprintf "sqrt %g" v) r;
       print "  steps" steps)
    [ 2.; 100. ];
  title "a loop under jit: the trip count is decided at run time";
  let fast_root =
    Compile.compile1
      ~backend:(Compile.jit_with ~on_trace:(fun _ -> print_endline "  [traced]"))
      root
  in
  print "fast_root 2" (fast_root (Ops.scalar 2.));
  print "fast_root 100" (fast_root (Ops.scalar 100.));
  title "derivatives of a loop: forward mode";
  print "deriv root 2" (Ad.deriv root (Ops.scalar 2.));
  print "by hand: 1 / (2 sqrt 2)" (Ops.scalar (1. /. (2. *. Float.sqrt 2.)));
  print "deriv fast_root 2" (Ad.deriv fast_root (Ops.scalar 2.));
  title "reverse mode needs a fixed trip count";
  match Ad.grad1 root (Ops.scalar 2.) with
  | _ -> ()
  | exception Missing_rule msg -> Printf.printf "  grad root: %s\n" msg
;;

let example = { name = "control"; summary = "cond and while_loop on data"; run }
