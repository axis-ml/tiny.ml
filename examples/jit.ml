open Tracer
open Common

let%ops f x = (x *. sin x) +. 1.
let%ops leaky x = if to_float x > 0. then x else 0.1 *. x

let staged_leaky x =
  Control.cond Ops.(x > scalar 0.) ~then_:Fun.id ~else_:Ops.(fun x -> scalar 0.1 * x) x
;;

let graphs () =
  title "the graph of f, for a scalar";
  show_graph (Ir.make f (Ops.scalar 0.));
  title "the graph of grad f: derivatives are ops too";
  show_graph (Ir.make (Ad.grad f) (Ops.scalar 0.))
;;

let jit () =
  title "jit: traced on the first call for each input type";
  let fast = Compile.compile ~backend:noisy_jit f in
  print "fast 1" (fast (Ops.scalar 1.));
  print "fast 2 (cached)" (fast (Ops.scalar 2.));
  print "fast [1 2] (new shape)" (fast (Ops.array [| 1.; 2. |]));
  print "deriv fast 1" (Ad.deriv fast (Ops.scalar 1.));
  title "under a transformation, a jitted function stays one call";
  show_graph (Ir.make (Ad.grad fast) (Ops.scalar 0.))
;;

let branching () =
  title "while tracing there is no data to branch on";
  print "leaky 2, eager" (leaky (Ops.scalar 2.));
  (match catch (fun () -> Ir.make leaky (Ops.scalar 2.)) with
   | Ok _ -> ()
   | Error (Concretization_error msg) -> Printf.printf "  Ir.make leaky: %s\n" msg
   | Error e -> failwith (error_to_string e));
  title "Control.cond traces both branches, so it stages";
  show_graph (Ir.make staged_leaky (Ops.scalar 2.));
  let fast = Compile.compile ~backend:Compile.jit staged_leaky in
  print "fast leaky -3" (fast (Ops.scalar (-3.)));
  print "grad fast leaky -3" (Ad.grad fast (Ops.scalar (-3.)))
;;

let run () =
  graphs ();
  jit ();
  branching ()
;;

let example = { name = "jit"; summary = "trace to a graph, compile with jit"; run }
