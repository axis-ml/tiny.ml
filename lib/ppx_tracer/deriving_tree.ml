open Ppxlib
module B = Ast_builder.Default

let tree_name = function
  | "t" -> "tree"
  | name -> name ^ "_tree"
;;

let tree_fn ~loc name = B.evar ~loc ("Tracer.Tree." ^ name)

let rec nest ~unit ~tuple = function
  | [] -> unit
  | [ x ] -> x
  | [ _; _; _ ] as xs -> tuple xs
  | x :: rest -> tuple [ x; nest ~unit ~tuple rest ]
;;

let nest_tree ~loc =
  nest ~unit:(tree_fn ~loc "unit") ~tuple:(fun xs ->
    let combinator = if List.length xs = 3 then "triple" else "pair" in
    B.eapply ~loc (tree_fn ~loc combinator) xs)
;;

let nest_pat ~loc = nest ~unit:(B.punit ~loc) ~tuple:(B.ppat_tuple ~loc)
let nest_exp ~loc = nest ~unit:(B.eunit ~loc) ~tuple:(B.pexp_tuple ~loc)

let rec of_core_type (ty : core_type) =
  let loc = ty.ptyp_loc in
  match ty.ptyp_desc with
  | Ptyp_constr ({ txt = Lident "value" | Ldot (_, "value"); _ }, []) ->
    tree_fn ~loc "value"
  | Ptyp_constr ({ txt = Lident "unit"; _ }, []) -> tree_fn ~loc "unit"
  | Ptyp_constr ({ txt = Lident (("list" | "array") as c); _ }, [ a ]) ->
    [%expr [%e tree_fn ~loc c] [%e of_core_type a]]
  | Ptyp_constr ({ txt = id; _ }, args) ->
    let id =
      match id with
      | Lident n -> Lident (tree_name n)
      | Ldot (m, n) -> Ldot (m, tree_name n)
      | Lapply _ -> Location.raise_errorf ~loc "ppx_tree: functor application"
    in
    let f = B.pexp_ident ~loc { txt = id; loc } in
    if args = [] then f else List.map of_core_type args |> B.eapply ~loc f
  | Ptyp_tuple tys -> List.map of_core_type tys |> nest_tree ~loc
  | Ptyp_var v -> B.evar ~loc ("tree_" ^ v)
  | _ ->
    Location.raise_errorf
      ~loc
      "ppx_tree: no Tree for this type (supported: value, unit, list, array, tuples, \
       type variables, and types with their own _tree)"
;;

let of_record ~loc (fields : label_declaration list) =
  let names = List.map (fun (f : label_declaration) -> f.pld_name.txt) fields in
  let lid n = { txt = Lident n; loc } in
  let record_pat =
    B.ppat_record ~loc (List.map (fun n -> lid n, B.pvar ~loc n) names) Closed
  in
  let record_exp =
    B.pexp_record ~loc (List.map (fun n -> lid n, B.evar ~loc n) names) None
  in
  let tuple_pat = List.map (B.pvar ~loc) names |> nest_pat ~loc in
  let tuple_exp = List.map (B.evar ~loc) names |> nest_exp ~loc in
  let inner = List.map (fun f -> of_core_type f.pld_type) fields |> nest_tree ~loc in
  [%expr
    Tracer.Tree.conv
      ~to_tree:(fun [%p record_pat] -> [%e tuple_exp])
      ~of_tree:(fun [%p tuple_pat] -> [%e record_exp])
      [%e inner]]
;;

let of_type_decl (td : type_declaration) =
  let loc = td.ptype_loc in
  let body =
    match td.ptype_kind, td.ptype_manifest with
    | Ptype_record fields, _ -> of_record ~loc fields
    | Ptype_abstract, Some ty -> of_core_type ty
    | Ptype_variant _, _ ->
      Location.raise_errorf ~loc "ppx_tree: variants need a Tree.choice combinator"
    | _ -> Location.raise_errorf ~loc "ppx_tree: unsupported type"
  in
  let params =
    td.ptype_params
    |> List.map (fun (ty, _) ->
      match ty.ptyp_desc with
      | Ptyp_var v -> B.pvar ~loc ("tree_" ^ v)
      | _ -> Location.raise_errorf ~loc "ppx_tree: unnamed type parameter")
  in
  let body = B.eabstract ~loc params body in
  let name = tree_name td.ptype_name.txt in
  B.value_binding ~loc ~pat:(B.pvar ~loc name) ~expr:body
;;

let str_gen =
  Deriving.Generator.V2.make_noarg (fun ~ctxt (rec_flag, tds) ->
    let loc = Expansion_context.Deriver.derived_item_loc ctxt in
    if really_recursive rec_flag tds = Recursive
    then Location.raise_errorf ~loc "ppx_tree: recursive types are not supported";
    [ List.map of_type_decl tds |> B.pstr_value ~loc Nonrecursive ])
;;

let sig_gen =
  Deriving.Generator.V2.make_noarg (fun ~ctxt (_, tds) ->
    let loc = Expansion_context.Deriver.derived_item_loc ctxt in
    let value_of td =
      let self = core_type_of_type_declaration td in
      let tree t = [%type: [%t t] Tracer.Tree.t] in
      let arrow (p, _) acc = [%type: [%t tree p] -> [%t acc]] in
      let ty = List.fold_right arrow td.ptype_params (tree self) in
      let name = { txt = tree_name td.ptype_name.txt; loc } in
      B.value_description ~loc ~name ~type_:ty ~prim:[] |> B.psig_value ~loc
    in
    List.map value_of tds)
;;

let () =
  Deriving.add "tree" ~str_type_decl:str_gen ~sig_type_decl:sig_gen |> ignore;
  Driver.register_transformation
    "tree_extension"
    ~extensions:
      [ Extension.V3.declare
          "tree"
          Extension.Context.expression
          Ast_pattern.(ptyp __)
          (fun ~ctxt:_ ty -> of_core_type ty)
      ]
;;
