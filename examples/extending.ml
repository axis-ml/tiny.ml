(* A new primitive and a new interpreter, from outside the library. *)

open Tracer
open Expert
open Common

(* softplus x = log (1 + exp x) *)
type Prim.t += Softplus

let softplus x = Interp.bind1 Softplus [ x ]

let () =
  Prim.register
    Softplus
    ~name:"softplus"
    ~abstract_eval:(fun _ avals -> avals)
    ~impl:(fun _ -> List.map (Ndarray.map (fun x -> Float.log1p (Float.exp x))));
  (* One rule gives both jvp and vjp. *)
  Derivatives.register_elementwise Softplus ~partials:(fun _ xs _ ->
    List.map (fun x -> Some (fun t -> Ops.(t * logistic x))) xs)
;;

(* Prints every primitive, then passes it to the interpreter below. *)
let trace_ops f x =
  let logger : unit Interp.t =
    Interp.create ~name:"log" (fun _ prim args ->
      Printf.printf "    %s\n" (Prim.name prim);
      Interp.bind prim args)
  in
  Interp.handle logger (fun () -> f x)
;;

let run () =
  let xs = Ops.array [| -1.; 0.; 1. |] in
  title "softplus, a new primitive";
  print "softplus xs" (softplus xs);
  print "deriv softplus xs" (Ad.deriv softplus xs);
  print "logistic xs (by hand)" (Ops.logistic xs);
  print "grad (sum (softplus xs))" (Ad.grad1 (fun x -> Ops.reduce_sum (softplus x)) xs);
  title "the primitives run by deriv softplus";
  print "result" (trace_ops (Ad.deriv softplus) xs)
;;

let example =
  { name = "extending"; summary = "a new primitive and a new interpreter"; run }
;;
