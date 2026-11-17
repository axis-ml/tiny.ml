let all : Common.example list =
  [ Basics.example; Training.example; Jit.example; Control.example; Extending.example ]
;;

let find name = List.find_opt (fun (e : Common.example) -> e.name = name) all

let run (e : Common.example) =
  Printf.printf "\n=== %s: %s ===\n" e.name e.summary;
  e.run ()
;;
