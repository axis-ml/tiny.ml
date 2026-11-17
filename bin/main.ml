(* Usage: tiny_ml [--list | <example>...] *)

open Tiny_ml_examples

let usage oc =
  Printf.fprintf oc "usage: tiny_ml [--list] [EXAMPLE ...]\n\nexamples:\n";
  List.iter
    (fun (e : Common.example) -> Printf.fprintf oc "  %-14s %s\n" e.name e.summary)
    Registry.all
;;

let () =
  match List.tl (Array.to_list Sys.argv) with
  | [] | [ "all" ] -> List.iter Registry.run Registry.all
  | [ ("-l" | "--list" | "-h" | "--help" | "help") ] -> usage stdout
  | names ->
    let unknown = List.filter (fun n -> Option.is_none (Registry.find n)) names in
    if unknown <> []
    then (
      Printf.eprintf "unknown example(s): %s\n\n" (String.concat ", " unknown);
      usage stderr;
      exit 2);
    List.iter (fun n -> Registry.run (Option.get (Registry.find n))) names
;;
