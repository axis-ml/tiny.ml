(* [%ops e] and [let%ops f x = e]: float syntax on traced values. Int
   operators stay Stdlib's. *)

open Ppxlib
module B = Ast_builder.Default

let ops ~loc name = B.evar ~loc ("Tracer.Ops." ^ name)
let app ~loc f args = B.eapply ~loc f args

let binops =
  [ "+.", "add"
  ; "-.", "sub"
  ; "*.", "mul"
  ; "/.", "div"
  ; ">.", "greater"
  ; "<.", "less"
  ; ">=.", "greater_equal"
  ; "<=.", "less_equal"
  ]
;;

(* Plain [abs] is int's, so it is left alone. *)
let funcs =
  List.concat_map
    (fun n -> [ Lident n, n; Ldot (Lident "Float", n), n ])
    [ "sin"
    ; "cos"
    ; "tan"
    ; "exp"
    ; "log"
    ; "sqrt"
    ; "sinh"
    ; "cosh"
    ; "tanh"
    ; "asinh"
    ; "acosh"
    ; "atanh"
    ]
  @ [ Lident "abs_float", "abs"; Ldot (Lident "Float", "abs"), "abs" ]
;;

let float_binfuncs = [ "max", "maximum"; "min", "minimum" ]
let stdlib_kept = [ "+"; "-"; "*"; "/"; "<"; ">"; "<="; ">="; "~-"; "abs" ]

let lift e =
  match e.pexp_desc with
  | Pexp_constant (Pconst_float _) ->
    let loc = e.pexp_loc in
    [%expr Tracer.Ops.scalar [%e e]]
  | _ -> e
;;

let float_literal e =
  match e.pexp_desc with
  | Pexp_constant (Pconst_float (s, None)) -> float_of_string_opt s
  | _ -> None
;;

let rewrite e =
  let loc = e.pexp_loc in
  match e.pexp_desc with
  | Pexp_apply
      ( { pexp_desc = Pexp_ident { txt = Lident "**"; _ }; _ }
      , [ (Nolabel, a); (Nolabel, b) ] ) ->
    (match float_literal b with
     | Some _ ->
       B.pexp_apply ~loc (ops ~loc "pow") [ Labelled "exponent", b; Nolabel, lift a ]
     | None ->
       [%expr Tracer.Ops.exp (Tracer.Ops.mul [%e lift b] (Tracer.Ops.log [%e lift a]))])
  | Pexp_apply
      ( { pexp_desc = Pexp_ident { txt = Lident op; _ }; _ }
      , [ (Nolabel, a); (Nolabel, b) ] )
    when List.mem_assoc op binops ->
    app ~loc (ops ~loc (List.assoc op binops)) [ lift a; lift b ]
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident "~-."; _ }; _ }, [ (Nolabel, a) ])
    -> app ~loc (ops ~loc "neg") [ a ]
  | Pexp_apply
      ( { pexp_desc = Pexp_ident { txt = Lident ("float_of_int" | "float"); _ }; _ }
      , [ (Nolabel, n) ] ) -> [%expr Tracer.Ops.scalar (Stdlib.float_of_int [%e n])]
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt = id; _ }; _ }, [ (Nolabel, a) ])
    when List.mem_assoc id funcs -> app ~loc (ops ~loc (List.assoc id funcs)) [ lift a ]
  | Pexp_apply
      ( { pexp_desc = Pexp_ident { txt = Ldot (Lident "Float", n); _ }; _ }
      , [ (Nolabel, a); (Nolabel, b) ] )
    when List.mem_assoc n float_binfuncs ->
    app ~loc (ops ~loc (List.assoc n float_binfuncs)) [ lift a; lift b ]
  | Pexp_ident { txt = Lident op; loc } when List.mem op stdlib_kept ->
    B.pexp_ident ~loc { txt = Ldot (Lident "Stdlib", op); loc }
  | _ -> e
;;

let mapper =
  object
    inherit Ast_traverse.map as super
    method! expression e = rewrite (super#expression e)
  end
;;

let region ~loc body =
  let body = mapper#expression body in
  let e =
    [%expr
      let open! Tracer.Ops in
      [%e body]]
  in
  { e with
    pexp_attributes =
      [ B.attribute
          ~loc
          ~name:{ txt = "ocaml.warning"; loc }
          ~payload:(PStr [ B.pstr_eval ~loc (B.estring ~loc "-33") [] ])
      ]
  }
;;

let rec under_fun ~loc e =
  match e.pexp_desc with
  | Pexp_fun (l, d, p, body) ->
    { e with pexp_desc = Pexp_fun (l, d, p, under_fun ~loc body) }
  | Pexp_newtype (t, body) -> { e with pexp_desc = Pexp_newtype (t, under_fun ~loc body) }
  | Pexp_constraint (inner, ty) ->
    { e with pexp_desc = Pexp_constraint (under_fun ~loc inner, ty) }
  | _ -> region ~loc e
;;

let under_vbs ~loc vbs =
  List.map (fun vb -> { vb with pvb_expr = under_fun ~loc vb.pvb_expr }) vbs
;;

let expr_ext =
  Extension.V3.declare
    "ops"
    Extension.Context.expression
    Ast_pattern.(single_expr_payload __)
    (fun ~ctxt e ->
       let loc = Expansion_context.Extension.extension_point_loc ctxt in
       match e.pexp_desc with
       | Pexp_let (rf, vbs, rest) -> B.pexp_let ~loc rf (under_vbs ~loc vbs) rest
       | _ -> region ~loc e)
;;

let str_ext =
  Extension.V3.declare
    "ops"
    Extension.Context.structure_item
    Ast_pattern.(pstr (pstr_value __ __ ^:: nil))
    (fun ~ctxt rf vbs ->
       let loc = Expansion_context.Extension.extension_point_loc ctxt in
       B.pstr_value ~loc rf (under_vbs ~loc vbs))
;;

let () = Driver.register_transformation "ops" ~extensions:[ expr_ext; str_ext ]
