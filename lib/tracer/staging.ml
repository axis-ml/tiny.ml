open Common
open Lang

module Ir = struct
  open Interp

  type var =
    { id : int
    ; aval : Aval.t
    }

  type eqn =
    { prim : Prim.t
    ; inputs : var list
    ; outputs : var list
    }

  type t =
    { params : var list
    ; captures : (var * value) list
    ; consts : (var * Ndarray.t) list
    ; eqns : eqn list
    ; outputs : var list
    ; num_vars : int
    }

  let avals xs = List.map (fun (x : var) -> x.aval) xs
  let ids xs = List.map (fun (x : var) -> x.id) xs

  let sub_graph_rules : (Prim.t -> (string * t) list) Prim.Table.t =
    Prim.Table.create "Ir.sub_graphs"
  ;;

  let register_sub_graphs p f = Prim.Table.add sub_graph_rules p f

  let sub_graphs p =
    match Prim.Table.find_opt sub_graph_rules p with
    | Some f -> f p
    | None -> []
  ;;

  exception Bad of string

  let rec check g =
    let fail fmt = Printf.ksprintf (fun s -> raise (Bad s)) fmt in
    let in_range (x : var) = x.id >= 0 && x.id < g.num_vars in
    let defined = Array.make (max g.num_vars 0) None in
    let define (x : var) =
      if not (in_range x) then fail "%%%d is out of range (%d vars)" x.id g.num_vars;
      if Option.is_some defined.(x.id) then fail "%%%d is defined twice" x.id;
      defined.(x.id) <- Some x.aval
    in
    let use (x : var) =
      match if in_range x then defined.(x.id) else None with
      | None -> fail "%%%d is used before it is defined" x.id
      | Some a ->
        if a <> x.aval
        then
          fail
            "%%%d is used as %s but defined as %s"
            x.id
            (Aval.to_string x.aval)
            (Aval.to_string a)
    in
    let holds what (x : var) a =
      if a <> x.aval
      then
        fail
          "%s %%%d is a %s holding a %s"
          what
          x.id
          (Aval.to_string x.aval)
          (Aval.to_string a)
    in
    List.iter define g.params;
    List.iter
      (fun (c, v) ->
         holds "capture" c (aval v);
         define c)
      g.captures;
    List.iter
      (fun (c, a) ->
         holds "const" c (Aval.of_ndarray a);
         define c)
      g.consts;
    List.iter
      (fun (e : eqn) ->
         List.iter use e.inputs;
         List.iter
           (fun (name, sub) ->
              try check sub with
              | Bad msg -> fail "%s.%s: %s" (Prim.name e.prim) name msg)
           (sub_graphs e.prim);
         let expected =
           try Prim.abstract_eval e.prim (avals e.inputs) with
           | Error.Type_error msg -> fail "ill-typed eqn: %s" msg
         in
         if expected <> avals e.outputs
         then
           fail "%s: outputs do not have the types abstract_eval gives" (Prim.name e.prim);
         List.iter define e.outputs)
      g.eqns;
    List.iter use g.outputs
  ;;

  let invariant g =
    try check g with
    | Bad msg -> failwith ("Ir.invariant: " ^ msg)
  ;;

  let trace_aux f in_avals =
    let ids = Graph.counter () in
    let fresh aval = { id = Graph.fresh ids; aval } in
    let captures = ref []
    and consts = ref []
    and eqns = ref []
    and seen = ref [] in
    let var_of stage v =
      match payload stage v with
      | Some x -> x
      | None ->
        (match List.assq_opt v !seen with
         | Some x -> x
         | None ->
           let x = fresh (aval v) in
           seen := (v, x) :: !seen;
           (match to_ndarray_opt v with
            | Some a -> consts := (x, a) :: !consts
            | None -> captures := (x, v) :: !captures);
           x)
    in
    let process self prim args =
      let inputs = List.map (var_of self) args in
      let outputs = List.map fresh (Prim.abstract_eval prim (List.map aval args)) in
      eqns := { prim; inputs; outputs } :: !eqns;
      List.map (fun x -> tracer self x.aval x) outputs
    in
    let stage = create ~name:"stage" process in
    let params = List.map fresh in_avals in
    let outs, aux =
      handle stage (fun () -> f (List.map (fun x -> tracer stage x.aval x) params))
    in
    let outputs = List.map (var_of stage) outs in
    let g =
      { params
      ; captures = List.rev !captures
      ; consts = List.rev !consts
      ; eqns = List.rev !eqns
      ; outputs
      ; num_vars = Graph.count ids
      }
    in
    if !Debug.enabled then invariant g;
    g, aux
  ;;

  let trace f in_avals = fst (trace_aux (fun xs -> f xs, ()) in_avals)

  let make in_tree out_tree f x =
    let leaves, s = Tree.flatten in_tree x in
    trace
      (fun tracers -> fst (Tree.flat_fn in_tree out_tree f s tracers))
      (List.map aval leaves)
  ;;

  let make1 f x = make Tree.value Tree.value f x

  let closure_convert g =
    ( { g with params = List.map fst g.captures @ g.params; captures = [] }
    , List.map snd g.captures )
  ;;

  let run g xs =
    if List.length xs <> List.length g.params
    then
      Error.type_error
        "Ir.run: expected %d input(s), got %d"
        (List.length g.params)
        (List.length xs);
    List.iter
      (fun (_, v) ->
         if not (alive v)
         then
           raise
             (Error.Escaped_tracer
                (Printf.sprintf
                   "Ir.run: the graph captured %s, whose transformation has returned; \
                    pass it as an argument instead"
                   (describe v))))
      g.captures;
    let env = Array.make g.num_vars None in
    let set (x : var) v = env.(x.id) <- Some v in
    List.iteri
      (fun i (p, x) ->
         if aval x <> p.aval
         then
           Error.type_error
             "Ir.run: input %d is %s, expected %s"
             i
             (Aval.to_string (aval x))
             (Aval.to_string p.aval);
         set p x)
      (List.combine g.params xs);
    List.iter (fun (c, v) -> set c v) g.captures;
    List.iter (fun (c, a) -> set c (of_ndarray a)) g.consts;
    let read x =
      match if x.id >= 0 && x.id < g.num_vars then env.(x.id) else None with
      | Some v -> v
      | None ->
        invalid_arg (Printf.sprintf "Ir.run: %%%d is used before it is defined" x.id)
    in
    List.iter
      (fun (e : eqn) ->
         let outs = bind e.prim (List.map read e.inputs) in
         if List.length outs <> List.length e.outputs
         then
           invalid_arg
             (Printf.sprintf
                "Ir.run: %s gave %d outputs, the eqn has %d"
                (Prim.name e.prim)
                (List.length outs)
                (List.length e.outputs));
         List.iter2 set e.outputs outs)
      g.eqns;
    List.map read g.outputs
  ;;

  let run1 g x =
    match run g [ x ] with
    | [ y ] -> y
    | ys -> Error.type_error "Ir.run1: expected 1 output, got %d" (List.length ys)
  ;;

  let producers g =
    let a = Array.make g.num_vars None in
    List.iter
      (fun (e : eqn) -> List.iter (fun (x : var) -> a.(x.id) <- Some e) e.outputs)
      g.eqns;
    a
  ;;

  let users g =
    let a = Array.make g.num_vars [] in
    List.iter
      (fun (e : eqn) ->
         List.iter
           (fun (x : var) -> if not (List.memq e a.(x.id)) then a.(x.id) <- e :: a.(x.id))
           e.inputs)
      (List.rev g.eqns);
    a
  ;;

  let live_mask g =
    List.rev
      (Graph.reaches
         ~num_ids:g.num_vars
         ~inputs:(fun (e : eqn) -> ids e.inputs)
         ~outputs:(fun (e : eqn) -> ids e.outputs)
         ~roots:(ids g.outputs)
         (List.rev g.eqns))
  ;;

  let dead g =
    List.combine (live_mask g) g.eqns
    |> List.filter_map (fun (live, e) -> if live then None else Some e)
  ;;

  let name x = Printf.sprintf "%%%d" x.id
  let typed x = name x ^ ":" ^ Aval.to_string x.aval
  let names xs = String.concat ", " (List.map name xs)

  let rec add_lines b indent g =
    let line fmt =
      Printf.ksprintf (fun s -> Buffer.add_string b (indent ^ s ^ "\n")) fmt
    in
    line "params: %s" (String.concat ", " (List.map typed g.params));
    List.iter (fun (c, v) -> line "%s = capture %s" (typed c) (describe v)) g.captures;
    List.iter
      (fun (c, a) -> line "%s = const %s" (typed c) (Ops.to_string (of_ndarray a)))
      g.consts;
    List.iter
      (fun (e : eqn) ->
         line
           "%s = %s(%s)"
           (String.concat ", " (List.map typed e.outputs))
           (Prim.name e.prim)
           (names e.inputs);
         List.iter
           (fun (n, sub) ->
              line "  %s:" n;
              add_lines b (indent ^ "    ") sub)
           (sub_graphs e.prim))
      g.eqns;
    line "outputs: %s" (names g.outputs)
  ;;

  let to_string g =
    let b = Buffer.create 256 in
    add_lines b "" g;
    Buffer.contents b
  ;;

  let pp fmt g = Format.pp_print_string fmt (to_string g)

  let to_dot g =
    let b = Buffer.create 1024 in
    let add fmt = Printf.ksprintf (Buffer.add_string b) fmt in
    let esc s =
      String.concat
        ""
        (List.map
           (function
             | '"' -> "\\\""
             | '\\' -> "\\\\"
             | c -> String.make 1 c)
           (List.of_seq (String.to_seq s)))
    in
    let clusters = ref 0 in
    let rec graph k g =
      let node x = Printf.sprintf "g%d_%s" k x in
      let src = Array.make g.num_vars "" in
      let leaf shape label (x : var) =
        let n = node (Printf.sprintf "v%d" x.id) in
        src.(x.id) <- n;
        add "  %s [shape=%s label=\"%s\"];\n" n shape (esc label)
      in
      List.iter (fun x -> leaf "ellipse" (typed x) x) g.params;
      List.iter (fun (x, _) -> leaf "diamond" (typed x ^ " capture") x) g.captures;
      List.iter (fun (x, _) -> leaf "note" (typed x ^ " const") x) g.consts;
      List.iteri
        (fun i (live, (e : eqn)) ->
           let n = node (Printf.sprintf "e%d" i) in
           add
             "  %s [shape=box label=\"%s\"%s];\n"
             n
             (esc (Prim.name e.prim))
             (if live then "" else " style=dashed");
           List.iter
             (fun (x : var) -> add "  %s -> %s [label=\"%s\"];\n" src.(x.id) n (name x))
             e.inputs;
           List.iter (fun (x : var) -> src.(x.id) <- n) e.outputs;
           List.iter
             (fun (sub_name, sub) ->
                incr clusters;
                let k' = !clusters in
                add
                  "  subgraph cluster_%d {\n  label=\"%s.%s\"; style=dotted;\n"
                  k'
                  (esc (Prim.name e.prim))
                  (esc sub_name);
                graph k' sub;
                add "  }\n";
                add "  g%d_out -> %s [style=dotted];\n" k' n)
             (sub_graphs e.prim))
        (List.combine (live_mask g) g.eqns);
      let out = node "out" in
      add "  %s [shape=plaintext label=\"outputs\"];\n" out;
      List.iter
        (fun (x : var) -> add "  %s -> %s [label=\"%s\"];\n" src.(x.id) out (name x))
        g.outputs
    in
    add "digraph ir {\n  node [fontname=\"monospace\"];\n";
    graph 0 g;
    add "}\n";
    Buffer.contents b
  ;;
end

module Compile = struct
  open Interp

  module type Backend = sig
    val name : string
    val compile : (value list -> value list * 'aux) -> value list -> value list * 'aux
  end

  type backend = (module Backend)

  module Eager = struct
    let name = "eager"
    let compile f = f
  end

  let eager : backend = (module Eager)

  let jit_with ~on_trace : backend =
    (module struct
      let name = "jit"

      let compile f =
        let cache = Hashtbl.create 8 in
        fun xs ->
          let avals = List.map aval xs in
          let g, aux =
            match Hashtbl.find_opt cache avals with
            | Some entry -> entry
            | None ->
              let ((g, _) as entry) = Ir.trace_aux f avals in
              on_trace g;
              (match g.Ir.captures with
               | [] -> Hashtbl.add cache avals entry
               | _ :: _ -> ());
              entry
          in
          Ir.run g xs, aux
      ;;
    end)
  ;;

  let jit = jit_with ~on_trace:ignore

  (* One [B.compile] per input structure. *)
  let compile ?(backend = eager) in_tree out_tree f =
    let module B = (val backend : Backend) in
    let instances = Hashtbl.create 4 in
    fun x ->
      let xs, s = Tree.flatten in_tree x in
      let compiled =
        match Hashtbl.find_opt instances s with
        | Some c -> c
        | None ->
          let c = B.compile (Tree.flat_fn in_tree out_tree f s) in
          Hashtbl.add instances s c;
          c
      in
      let ys, out = compiled xs in
      if List.length ys <> Tree.num_leaves out
      then
        invalid_arg
          (Printf.sprintf
             "Compile: backend %s returned %d leaves, the output has %d"
             B.name
             (List.length ys)
             (Tree.num_leaves out));
      Tree.unflatten out_tree out ys
  ;;

  let compile1 ?backend f = compile ?backend Tree.value Tree.value f
end

module Control = struct
  open Interp
  module Z = Autodiff.Maybe_zero
  module Jvp = Autodiff.Jvp
  module Vjp = Autodiff.Vjp

  type Prim.t +=
    | Cond of
        { then_ : Ir.t
        ; else_ : Ir.t
        }
    | While of
        { cond : Ir.t
        ; body : Ir.t
        }

  let rec split n l =
    if n <= 0
    then [], l
    else (
      match l with
      | x :: rest ->
        let a, b = split (n - 1) rest in
        x :: a, b
      | [] -> invalid_arg "Control: too few operands")
  ;;

  let avals = List.map aval
  let avals_str l = String.concat ", " (List.map Aval.to_string l)
  let is_scalar (a : Aval.t) = Array.length a.shape = 0

  let empty =
    { Ir.params = []; captures = []; consts = []; eqns = []; outputs = []; num_vars = 0 }
  ;;

  let closed f in_avals = Ir.closure_convert (Ir.trace f in_avals)

  let traced f in_avals =
    match closed f in_avals with
    | g, [] -> g
    | _, _ :: _ -> failwith "Control: a rule's graph captured a tracer"
  ;;

  let run_arrays g arrays = List.map to_ndarray (Ir.run g (List.map of_ndarray arrays))

  let expect what ~graph ~operands =
    if graph <> operands
    then
      Error.type_error
        "%s: operands are (%s), the graph takes (%s)"
        what
        (avals_str operands)
        (avals_str graph)
  ;;

  let cond_abstract_eval p avals =
    match p, avals with
    | Cond { then_; else_ }, pred :: rest ->
      if not (is_scalar pred)
      then
        Error.type_error
          "cond: the predicate must be a scalar, got %s"
          (Aval.to_string pred);
      let nt = List.length then_.params
      and ne = List.length else_.params in
      if List.length rest <> nt + ne then Prim.arity_error p ~expected:(1 + nt + ne) avals;
      let at, ae = split nt rest in
      expect "cond then" ~graph:(Ir.avals then_.params) ~operands:at;
      expect "cond else" ~graph:(Ir.avals else_.params) ~operands:ae;
      let ot = Ir.avals then_.outputs
      and oe = Ir.avals else_.outputs in
      if ot <> oe
      then
        Error.type_error
          "cond: the branches return (%s) and (%s)"
          (avals_str ot)
          (avals_str oe);
      ot
    | _ -> Prim.arity_error p ~expected:1 avals
  ;;

  let cond_impl p arrays =
    match p, arrays with
    | Cond { then_; else_ }, pred :: rest ->
      let at, ae = split (List.length then_.params) rest in
      if Ndarray.to_float pred > 0. then run_arrays then_ at else run_arrays else_ ae
    | _ -> Prim.arity_error p ~expected:1 arrays
  ;;

  let while_layout cond (body : Ir.t) =
    let n = List.length body.outputs in
    List.length cond.Ir.params - n, List.length body.params - n, n
  ;;

  let while_abstract_eval p avals =
    match p with
    | While { cond; body } ->
      let nc, nb, n = while_layout cond body in
      if nc < 0 || nb < 0 || List.length avals <> nc + nb + n
      then Prim.arity_error p ~expected:(nc + nb + n) avals;
      let ac, rest = split nc avals in
      let ab, carry = split nb rest in
      expect "while cond" ~graph:(Ir.avals cond.params) ~operands:(ac @ carry);
      expect "while body" ~graph:(Ir.avals body.params) ~operands:(ab @ carry);
      (match Ir.avals cond.outputs with
       | [ a ] when is_scalar a -> ()
       | l ->
         Error.type_error "while_loop: cond must return a scalar, got (%s)" (avals_str l));
      if Ir.avals body.outputs <> carry
      then
        Error.type_error
          "while_loop: body returns (%s) for a carry of (%s)"
          (avals_str (Ir.avals body.outputs))
          (avals_str carry);
      carry
    | _ -> Prim.arity_error p ~expected:0 avals
  ;;

  let while_impl p arrays =
    match p with
    | While { cond; body } ->
      let nc, nb, _ = while_layout cond body in
      let ac, rest = split nc arrays in
      let ab, carry = split nb rest in
      let rec loop carry =
        match run_arrays cond (ac @ carry) with
        | [ c ] when Ndarray.to_float c > 0. -> loop (run_arrays body (ab @ carry))
        | _ -> carry
      in
      loop carry
    | _ -> Prim.arity_error p ~expected:0 arrays
  ;;

  let nonzero = List.map (fun d -> Z.Nonzero d)
  let instantiate = List.map Z.instantiate

  let jvp_graph (g : Ir.t) =
    let n = List.length g.params in
    let a = Ir.avals g.params in
    traced
      (fun l ->
         let xs, dxs = split n l in
         let ys, dys, () = Jvp.jvp_flat (fun xs -> Ir.run g xs, ()) xs (nonzero dxs) in
         ys @ instantiate dys)
      (a @ a)
  ;;

  let vjp_graph (g : Ir.t) ~before ~after =
    let n = List.length g.params in
    let zeros = List.map (fun (a : Aval.t) -> Ops.zeros a.shape) in
    traced
      (fun l ->
         let xs, cts = split n l in
         let _, (), back = Vjp.vjp_flat (fun xs -> Ir.run g xs, ()) xs in
         zeros before @ back cts @ zeros after)
      (Ir.avals g.params @ Ir.avals g.outputs)
  ;;

  let () =
    let cond_key = Cond { then_ = empty; else_ = empty }
    and while_key = While { cond = empty; body = empty } in
    Prim.register cond_key ~name:"cond" ~abstract_eval:cond_abstract_eval ~impl:cond_impl;
    Prim.register
      while_key
      ~name:"while"
      ~abstract_eval:while_abstract_eval
      ~impl:while_impl;
    Ir.register_sub_graphs cond_key (function
      | Cond { then_; else_ } -> [ "then", then_; "else", else_ ]
      | _ -> []);
    Ir.register_sub_graphs while_key (function
      | While { cond; body } -> [ "cond", cond; "body", body ]
      | _ -> []);
    Jvp.register_rule cond_key (fun p xs dxs ->
      match p, xs, dxs with
      | Cond { then_; else_ }, pred :: rest, _ :: drest ->
        let nt = List.length then_.params in
        let at, ae = split nt rest
        and dt, de = split nt drest in
        let outs =
          bind
            (Cond { then_ = jvp_graph then_; else_ = jvp_graph else_ })
            ((pred :: (at @ instantiate dt)) @ ae @ instantiate de)
        in
        let ys, dys = split (List.length then_.outputs) outs in
        ys, nonzero dys
      | _ -> Prim.arity_error p ~expected:1 xs);
    Vjp.register_rule cond_key (fun p xs _ ->
      match p, xs with
      | Cond { then_; else_ }, pred :: rest ->
        let at, ae = split (List.length then_.params) rest in
        fun cts ->
          let cts = instantiate cts in
          let back =
            Cond
              { then_ = vjp_graph then_ ~before:[] ~after:(avals ae)
              ; else_ = vjp_graph else_ ~before:(avals at) ~after:[]
              }
          in
          Z.zero_like pred :: nonzero (bind back ((pred :: (at @ cts)) @ ae @ cts))
      | _ -> Prim.arity_error p ~expected:1 xs);
    Jvp.register_rule while_key (fun p xs dxs ->
      match p with
      | While { cond; body } ->
        let nc, nb, n = while_layout cond body in
        let ac, rest = split nc xs in
        let ab, carry = split nb rest in
        let _, drest = split nc dxs in
        let dab, dcarry = split nb drest in
        let ca = avals carry in
        let cond' =
          traced
            (fun l ->
               let ac, l = split nc l in
               let carry, _ = split n l in
               Ir.run cond (ac @ carry))
            (avals ac @ ca @ ca)
        in
        let body' =
          traced
            (fun l ->
               let ab, l = split nb l in
               let dab, l = split nb l in
               let carry, dcarry = split n l in
               let ys, dys, () =
                 Jvp.jvp_flat
                   (fun xs -> Ir.run body xs, ())
                   (ab @ carry)
                   (nonzero (dab @ dcarry))
               in
               ys @ instantiate dys)
            (avals ab @ avals ab @ ca @ ca)
        in
        let outs =
          bind
            (While { cond = cond'; body = body' })
            (ac @ (ab @ instantiate dab) @ carry @ instantiate dcarry)
        in
        let ys, dys = split n outs in
        ys, nonzero dys
      | _ -> Prim.arity_error p ~expected:0 xs);
    Vjp.register_rule while_key (fun _ _ _ ->
      raise
        (Error.Missing_rule
           "vjp of while_loop: reverse mode needs a fixed trip count; use forward mode \
            (Ad.jvp, Ad.jacfwd) or an OCaml loop"))
  ;;

  let cond_flat pred then_ else_ xs =
    let a = avals xs in
    let gt, ct = closed then_ a
    and ge, ce = closed else_ a in
    bind (Cond { then_ = gt; else_ = ge }) ((pred :: (ct @ xs)) @ ce @ xs)
  ;;

  let cond in_tree out_tree pred ~then_ ~else_ x =
    let xs, s = Tree.flatten in_tree x in
    let out = ref None in
    let branch f leaves =
      let ys, o = Tree.flat_fn in_tree out_tree f s leaves in
      (match !out with
       | None -> out := Some o
       | Some o' ->
         if o <> o' then Error.type_error "cond: the branches return different structures");
      ys
    in
    let ys = cond_flat pred (branch then_) (branch else_) xs in
    match !out with
    | Some o -> Tree.unflatten out_tree o ys
    | None -> assert false
  ;;

  let cond1 pred ~then_ ~else_ x = cond Tree.value Tree.value pred ~then_ ~else_ x

  let while_loop tree ~cond ~body init =
    let xs, s = Tree.flatten tree init in
    let a = avals xs in
    let of_leaves leaves = Tree.unflatten tree s leaves in
    let gc, cc = closed (fun leaves -> [ cond (of_leaves leaves) ]) a in
    let gb, bc =
      closed
        (fun leaves ->
           let ys, s' = Tree.flatten tree (body (of_leaves leaves)) in
           if s' <> s
           then Error.type_error "while_loop: body changed the structure of the carry";
           ys)
        a
    in
    of_leaves (bind (While { cond = gc; body = gb }) (cc @ bc @ xs))
  ;;

  let while_loop1 ~cond ~body x = while_loop Tree.value ~cond ~body x
end
