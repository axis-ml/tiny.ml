open Tracer
open Common

let%ops f x = (x *. sin x) +. 1.
let%ops leaky x = if to_float x > 0. then x else 0.1 *. x

let staged_leaky x =
  Control.cond Ops.(x > scalar 0.) ~then_:Fun.id ~else_:Ops.(fun x -> scalar 0.1 * x) x
;;

let graphs () =
  title "the graph of f, for a scalar";
  Ir.make f (Ops.scalar 0.) |> show_graph;
  title "the graph of grad f: derivatives are ops too";
  Ir.make (Ad.grad f) (Ops.scalar 0.) |> show_graph
;;

let jit () =
  title "jit: traced on the first call for each input type";
  let fast = Compile.compile ~backend:noisy_jit f in
  Ops.scalar 1. |> fast |> print "fast 1";
  Ops.scalar 2. |> fast |> print "fast 2 (cached)";
  Ops.array [| 1.; 2. |] |> fast |> print "fast [1 2] (new shape)";
  Ops.scalar 1. |> Ad.deriv fast |> print "deriv fast 1";
  title "under a transformation, a jitted function stays one call";
  Ir.make (Ad.grad fast) (Ops.scalar 0.) |> show_graph
;;

let branching () =
  title "while tracing there is no data to branch on";
  Ops.scalar 2. |> leaky |> print "leaky 2, eager";
  (match catch (fun () -> Ir.make leaky (Ops.scalar 2.)) with
   | Ok _ -> ()
   | Error (Concretization_error msg) -> Printf.printf "  Ir.make leaky: %s\n" msg
   | Error e -> failwith @@ error_to_string e);
  title "Control.cond traces both branches, so it stages";
  Ir.make staged_leaky (Ops.scalar 2.) |> show_graph;
  let fast = Compile.compile ~backend:Compile.jit staged_leaky in
  Ops.scalar (-3.) |> fast |> print "fast leaky -3";
  Ops.scalar (-3.) |> Ad.grad fast |> print "grad fast leaky -3"
;;

let run () =
  graphs ();
  jit ();
  branching ()
;;

let example = { name = "jit"; summary = "trace to a graph, compile with jit"; run }
