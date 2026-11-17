(* Tracer ids shift when examples change; [dune promote]. *)

open Tiny_ml_examples

let () = List.iter Registry.run Registry.all
