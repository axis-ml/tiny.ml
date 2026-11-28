open Tracer

type example =
  { name : string
  ; summary : string
  ; run : unit -> unit
  }

let title t = Printf.printf "\n-- %s --\n" t
let print label v = Printf.printf "  %-30s %s\n" label (Ops.to_string v)

let show_graph g =
  Ir.to_string g
  |> String.split_on_char '\n'
  |> List.filter (( <> ) "")
  |> List.iter (Printf.printf "    %s\n")
;;

let noisy_jit = Compile.jit_with ~on_trace:(fun _ -> print_endline "  [traced]")
