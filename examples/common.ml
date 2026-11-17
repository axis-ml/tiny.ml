open Tracer

type example =
  { name : string
  ; summary : string
  ; run : unit -> unit
  }

let title t = Printf.printf "\n-- %s --\n" t
let print label v = Printf.printf "  %-30s %s\n" label (Ops.to_string v)
