(* Staging: trace a function into a graph, and compile it with jit. *)

open Tracer
open Common

let%ops f x = (x *. sin x) +. 1.

let show_graph g =
  String.split_on_char '\n' (Ir.to_string g)
  |> List.iter (fun l -> if l <> "" then Printf.printf "    %s\n" l)
;;

let run () =
  title "the graph of f, for a scalar";
  show_graph (Ir.make1 f (Ops.scalar 0.));
  title "the graph of grad f: derivatives are ops too";
  show_graph (Ir.make1 (Ad.grad1 f) (Ops.scalar 0.));
  title "jit: traced on the first call for each input type";
  let fast =
    Compile.compile1
      ~backend:(Compile.jit_with ~on_trace:(fun _ -> print_endline "  [traced]"))
      f
  in
  print "fast 1" (fast (Ops.scalar 1.));
  print "fast 2 (cached)" (fast (Ops.scalar 2.));
  print "fast [1 2] (new shape)" (fast (Ops.array [| 1.; 2. |]));
  print "deriv fast 1" (Ad.deriv fast (Ops.scalar 1.));
  title "while tracing there is no data to branch on";
  let%ops leaky x = if to_float x > 0. then x else 0.1 *. x in
  print "leaky 2, eager" (leaky (Ops.scalar 2.));
  (match Ir.make1 leaky (Ops.scalar 2.) with
   | _ -> ()
   | exception Concretization_error msg -> Printf.printf "  Ir.make1 leaky: %s\n" msg);
  title "Control.cond traces both branches, so it stages";
  let leaky x =
    Control.cond1 Ops.(x > scalar 0.) ~then_:Fun.id ~else_:Ops.(fun x -> scalar 0.1 * x) x
  in
  show_graph (Ir.make1 leaky (Ops.scalar 2.));
  let fast = Compile.compile1 ~backend:Compile.jit leaky in
  print "fast leaky -3" (fast (Ops.scalar (-3.)));
  print "grad fast leaky -3" (Ad.grad1 fast (Ops.scalar (-3.)))
;;

let example = { name = "jit"; summary = "trace to a graph, compile with jit"; run }
