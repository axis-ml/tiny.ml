open Tracer
open Expert
open Common

let softplus =
  Define.unary
    ~name:"softplus"
    ~impl:(fun x -> Float.log1p (Float.exp x))
    ~deriv:(fun x t -> Ops.(t * logistic x))
;;

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
  Ad.grad (fun x -> softplus x |> Ops.reduce_sum) xs |> print "grad (sum (softplus xs))";
  title "the primitives run by deriv softplus";
  trace_ops (Ad.deriv softplus) xs |> print "result"
;;

let example =
  { name = "extending"; summary = "a new primitive and a new interpreter"; run }
;;
