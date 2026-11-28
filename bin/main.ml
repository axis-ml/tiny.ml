open Tiny_ml_examples

let usage oc =
  Printf.fprintf oc "usage: tiny_ml [--list] [EXAMPLE ...]\n\nexamples:\n";
  List.iter
    (fun (e : Common.example) -> Printf.fprintf oc "  %-14s %s\n" e.name e.summary)
    Registry.all
;;

let run_named names =
  let examples, unknown =
    List.partition_map
      (fun n ->
         match Registry.find n with
         | Some e -> Either.Left e
         | None -> Either.Right n)
      names
  in
  match unknown with
  | [] -> List.iter Registry.run examples
  | _ ->
    Printf.eprintf "unknown example(s): %s\n\n" (String.concat ", " unknown);
    usage stderr;
    exit 2
;;

let () =
  match List.tl (Array.to_list Sys.argv) with
  | [] | [ "all" ] -> List.iter Registry.run Registry.all
  | [ ("-l" | "--list" | "-h" | "--help" | "help") ] -> usage stdout
  | names -> run_named names
;;
